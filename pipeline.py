#!/usr/bin/env python3
"""
Pipeline completo: CSV bruto → canais endêmicos → canais por faixa etária → index.html

Uso:
    python pipeline.py dados_ids.csv --pop 210000 --output index.html

Este script:
1. Roda compute_channels.py para gerar channel_data.json (58 canais)
2. Agrega dados por faixa etária para 20 agravos prioritários
3. Computa canais Gamma-Poisson por faixa etária
4. Gera boletim enriquecido
5. Monta o index.html final com todos os dados embutidos
"""

import json
import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

STEP 0 — Pré-processamento de CSV bruto Maestro 187
====================================================
Cole este bloco no pipeline.py ANTES da função step1_compute_channels().
Também adicione a chamada no main() conforme indicado no final deste arquivo.
 
Entrada:  CSV bruto do Maestro 187
          Data;Código US;Unidade de Saúde;Usuário;Sexo;Faixa etária;C.E.P.;C.I.D.;Descrição - C.I.D.;Quantidade
 
Saída:    CSV processado compatível com o pipeline
          ano_epi;semana_epi;faixa_etaria;cid_codigo;cid_descricao;quantidade
 
Aplica:
  - Detecção automática de formato (bruto vs agregado)
  - Pseudonimização SHA-256 + salt (LGPD Art.13 §4°)
  - Decodificação de idade IDS → faixa etária (200+M=meses, 300+D=dias)
  - Cálculo de semana epidemiológica MS/OMS (dom-sáb)
  - Remoção de duplicatas (data+unidade+pseudo_id+cid)
  - Filtragem de linhas totalizadoras
  - Agregação final: ano_epi × semana_epi × faixa_etaria × cid → quantidade
"""
 
import hashlib
import datetime
import re
from pathlib import Path
 
import numpy as np
import pandas as pd
 
# ── Configuração ──────────────────────────────────────────────────────────────
PSEUDO_SALT = 'upa-rc-2026'   # mesmo salt do GAS — resultados idênticos
 
# ── Funções auxiliares ────────────────────────────────────────────────────────
 
def _pseudo_id(uid: str) -> str:
    """SHA-256 + salt → 12 chars hex. Idêntico ao GAS pseudoId_()."""
    key = PSEUDO_SALT + ':' + str(uid).strip()
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]
 
 
def _decode_age(raw) -> str:
    """
    Decodifica idade codificada do IDS:
      0-199  = anos
      200-299 = meses (200 + M)
      300+    = dias (300 + D)
    Retorna faixa etária string (ex: '30-39').
    """
    try:
        v = int(str(raw).strip())
        if v >= 300:   years = (v - 300) / 365.25
        elif v >= 200: years = (v - 200) / 12
        else:          years = float(v)
    except (ValueError, TypeError):
        return 'NI'
 
    if years < 0:   return 'NI'
    if years < 1:   return '<1'
    if years < 5:   return '1-4'
    if years < 10:  return '5-9'
    if years < 15:  return '10-14'
    if years < 20:  return '15-19'
    if years < 30:  return '20-29'
    if years < 40:  return '30-39'
    if years < 50:  return '40-49'
    if years < 60:  return '50-59'
    if years < 70:  return '60-69'
    if years < 80:  return '70-79'
    return '80+'
 
 
def _epi_week(date_str: str):
    """
    Semana epidemiológica MS/OMS (domingo–sábado).
    Idêntica ao epi_week() do compute_channels.py.
    Retorna (ano_epi, se) ou (None, None) se inválido.
    """
    try:
        s = str(date_str).strip()
        if '/' in s:
            p = s.split('/')
            d = datetime.date(int(p[2]), int(p[1]), int(p[0]))
        elif '-' in s:
            p = s.split('-')
            d = datetime.date(int(p[0]), int(p[1]), int(p[2]))
        else:
            return None, None
 
        dow_sun  = (d.weekday() + 1) % 7          # dom=0 … sáb=6
        sun      = d - datetime.timedelta(days=dow_sun)
        wed      = sun + datetime.timedelta(days=3)
        year     = wed.year
        jan1     = datetime.date(year, 1, 1)
        j1d      = (jan1.weekday() + 1) % 7
        first_sun = (jan1 - datetime.timedelta(days=j1d)
                     if j1d <= 3
                     else jan1 + datetime.timedelta(days=7 - j1d))
        se = (sun - first_sun).days // 7 + 1
        return year, max(1, min(53, se))
    except Exception:
        return None, None
 
 
def _decode_sexo(code: str) -> str:
    s = str(code).strip()
    if s == '1': return 'MASCULINO'
    if s == '2': return 'FEMININO'
    return s or 'NI'
 
 
def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Localiza coluna pelo nome (tolerante a acentos e variações)."""
    for c in df.columns:
        cl = c.lower().strip()
        for cand in candidates:
            if cand in cl:
                return c
    return None
 
 
def _is_maestro187_raw(df: pd.DataFrame) -> bool:
    """
    Detecta se o CSV é formato bruto Maestro 187.
    Critério: tem coluna 'data' + ('usuário' ou 'código us').
    """
    cols = [c.lower().strip() for c in df.columns]
    has_data  = any('data' == c for c in cols)
    has_user  = any('usu' in c for c in cols)
    has_unit  = any('código us' in c or 'codigo us' in c for c in cols)
    # Formato agregado tem ano_epi — não é bruto
    has_agg   = any('ano_epi' in c for c in cols)
    return has_data and (has_user or has_unit) and not has_agg
 
 
# ── STEP 0 ────────────────────────────────────────────────────────────────────
 
def step0_preprocess_if_needed(csv_path: str) -> str:
    """
    Verifica se o CSV de entrada é formato bruto Maestro 187.
    Se sim: pré-processa e salva versão processada, retorna novo caminho.
    Se não: retorna o caminho original (já é formato agregado).
 
    O arquivo processado é salvo como:
      {original_stem}_processed.csv
 
    Parameters
    ----------
    csv_path : str
        Caminho do CSV de entrada.
 
    Returns
    -------
    str
        Caminho do CSV a usar nas etapas seguintes.
    """
    print("\n" + "=" * 60)
    print("STEP 0: Verificando formato do CSV de entrada")
    print("=" * 60)
 
    path = Path(csv_path)
 
    # Tentar ler com diferentes encodings
    df = None
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, sep=';', dtype=str,
                             encoding=enc, skipinitialspace=True)
            print(f"  Encoding: {enc}")
            break
        except Exception:
            continue
 
    if df is None:
        print("  ERRO: não foi possível ler o CSV. Prosseguindo sem Step 0.")
        return csv_path
 
    # Limpar nomes de colunas
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
 
    print(f"  Colunas: {list(df.columns)}")
 
    if not _is_maestro187_raw(df):
        print("  → Formato já processado (agregado). Pulando Step 0.")
        return csv_path
 
    print("  → Formato bruto Maestro 187 detectado. Pré-processando...")
 
    # ── Filtrar linhas válidas ────────────────────────────────────────
    col_data = _find_col(df, ['data'])
    if not col_data:
        print("  ERRO: coluna 'Data' não encontrada.")
        return csv_path
 
    df = df[df[col_data].str.match(r'\d{2}/\d{2}/\d{4}', na=False)].copy()
    print(f"  Linhas com data válida: {len(df)}")
 
    if len(df) == 0:
        print("  ERRO: nenhuma linha válida encontrada.")
        return csv_path
 
    # ── Mapear colunas ────────────────────────────────────────────────
    col_unit_cd = _find_col(df, ['código us', 'codigo us', 'cod us'])
    col_unit_nm = _find_col(df, ['unidade de saúde', 'unidade de saude',
                                  'unidade saude', 'unidade'])
    col_user    = _find_col(df, ['usuário', 'usuario'])
    col_sexo    = _find_col(df, ['sexo'])
    col_faixa   = _find_col(df, ['faixa etária', 'faixa etaria', 'faixa'])
    col_cep     = _find_col(df, ['c.e.p', 'cep'])
    col_cid     = _find_col(df, ['c.i.d.', 'cid_codigo', 'cid'])
    col_desc    = _find_col(df, ['descrição - c.i.d', 'descricao - c.i.d',
                                  'descrição', 'descricao', 'cid_descricao'])
 
    print(f"  Mapeamento detectado:")
    print(f"    data={col_data}, unid_cd={col_unit_cd}, unid_nm={col_unit_nm}")
    print(f"    user={col_user}, sexo={col_sexo}, faixa={col_faixa}")
    print(f"    cep={col_cep}, cid={col_cid}, desc={col_desc}")
 
    # ── Processar ─────────────────────────────────────────────────────
    rows = []
    erros = 0
 
    for _, r in df.iterrows():
        data_str = r[col_data].strip()
        if not re.match(r'\d{2}/\d{2}/\d{4}', data_str):
            continue
 
        ano_epi, se_epi = _epi_week(data_str)
        if not ano_epi:
            erros += 1
            continue
 
        unit_cd  = r[col_unit_cd].strip()  if col_unit_cd else ''
        unit_nm  = r[col_unit_nm].strip()  if col_unit_nm else ''
        user     = r[col_user].strip()     if col_user    else ''
        sexo_raw = r[col_sexo].strip()     if col_sexo    else ''
        faixa_raw= r[col_faixa].strip()    if col_faixa   else ''
        cep_raw  = r[col_cep].strip()      if col_cep     else ''
        cid_cd   = r[col_cid].strip()      if col_cid     else ''
        cid_desc = r[col_desc].strip()     if col_desc    else ''
 
        rows.append([
            data_str,
            ano_epi,
            se_epi,
            unit_cd,
            unit_nm,
            _pseudo_id(user) if user else '',
            _decode_sexo(sexo_raw),
            _decode_age(faixa_raw),
            ''.join(c for c in cep_raw if c.isdigit()),
            cid_cd,
            cid_desc,
            1
        ])
 
    HEADERS = ['data', 'ano_epi', 'semana_epi', 'unidade_codigo', 'unidade_nome',
               'pseudo_id', 'sexo', 'faixa_etaria', 'cep', 'cid_codigo',
               'cid_descricao', 'quantidade']
 
    out = pd.DataFrame(rows, columns=HEADERS)
    print(f"  Linhas processadas: {len(out)} (erros SE: {erros})")
 
    # ── Remover duplicatas ────────────────────────────────────────────
    before = len(out)
    out['_key'] = (out['data'] + '|' + out['unidade_codigo'] + '|' +
                   out['pseudo_id'] + '|' + out['cid_descricao'])
    out = out[~out.duplicated('_key', keep='first')].drop(columns=['_key'])
    dupes = before - len(out)
    print(f"  Duplicatas removidas: {dupes}")
    print(f"  Linhas únicas: {len(out)}")
 
    # ── Agregar para formato do pipeline ─────────────────────────────
    # O pipeline espera: ano_epi;semana_epi;faixa_etaria;cid_codigo;cid_descricao;quantidade
    agg = (out.groupby(['ano_epi', 'semana_epi', 'faixa_etaria',
                        'cid_codigo', 'cid_descricao'], as_index=False)
              .agg(quantidade=('quantidade', 'sum')))
 
    print(f"  Linhas após agregação: {len(agg)}")
 
    # ── Estatísticas ──────────────────────────────────────────────────
    anos = sorted(out['ano_epi'].unique())
    print(f"  Anos presentes: {anos}")
    se_range = f"SE {out['semana_epi'].min()} a SE {out['semana_epi'].max()}"
    print(f"  Semanas: {se_range}")
    print(f"  Atendimentos únicos: {(out['data']+'|'+out['pseudo_id']).nunique()}")
 
    # ── Salvar arquivo processado ─────────────────────────────────────
    out_path = path.parent / (path.stem + '_processed.csv')
    agg.to_csv(out_path, sep=';', index=False, encoding='utf-8')
    size_kb = out_path.stat().st_size / 1024
    print(f"  → Salvo: {out_path} ({size_kb:.1f} KB)")
 
    return str(out_path)
 
 

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

MC_SAMPLES = 30_000
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
RNG_SEED = 2026
MAX_SE = 52
TRAINING_YEARS_DEFAULT = ['2023', '2024', '2025']  # 2021/2022 excluídos (implantação)
MONITOR_YEAR = '2026'
FALLBACK_SHAPE = 0.1
FALLBACK_RATE = 1.0

AGE_GROUPS = {
    "Bebês (0-1)": (0, 1),
    "Crianças (2-11)": (2, 11),
    "Adolescentes (12-17)": (12, 17),
    "Adultos jovens (18-39)": (18, 39),
    "Meia-idade (40-59)": (40, 59),
    "Idosos (60+)": (60, 200),
}

# 20 agravos prioritários para canais por faixa etária
KEY_AGRAVOS_AGE = [
    "Todos os atendimentos",
    "X - Aparelho respiratório",
    "SINAN: Influenza NE",
    "SINAN: Dengue",
    "SINAN: Diarréia/gastroenterite",
    "SINAN: Pneumonia NE",
    "Sínd. Gripal (IVAS+Febre)",
    "Sínd. Respiratória (J09-J22)",
    "Gastroenterites (A09+K52)",
    "Saúde Mental",
    "Dor Osteomuscular",
    "Cardiovascular Aguda",
    "XIII - Sistema osteomuscular",
    "V - Transtornos mentais",
    "XIX - Lesões e causas externas",
    "XVIII - Sintomas e sinais",
    "Acidentes de Trânsito",
    "Agressões",
    "Tentativas de Suicídio",
    "Dermatológica Aguda",
]

# Mapeamento descrição → CID (mesmo do compute_channels.py)
# Este dicionário é importado de compute_channels.py via exec
DESC_TO_CID = {}

# Definições de síndromes (mesmo do compute_channels.py)
SYNDROME_DEFS = {}


# ══════════════════════════════════════════════════════════════════════
# Gamma-Poisson (simplificado para faixas etárias)
# ══════════════════════════════════════════════════════════════════════

from math import lgamma, log, exp, inf

def estimate_params_mom_simple(cases_arr):
    x = np.array(cases_arr, dtype=float)
    m = np.mean(x)
    v = np.var(x, ddof=1) if len(x) > 1 else m + 1
    if m <= 0:
        return FALLBACK_SHAPE, FALLBACK_RATE
    if v <= m:
        v = m + 0.01
    denom = v - m
    if denom <= 0:
        denom = 0.01
    a_hat = max(m * m / denom, 0.01)
    b_hat = max(m / denom, 0.001)
    return a_hat, b_hat


def nb_loglik(x_arr, shape, rate):
    r, p = shape, rate / (rate + 1.0)
    ll = 0.0
    for x in x_arr:
        x = int(x)
        ll += lgamma(x + r) - lgamma(r) - lgamma(x + 1)
        ll += r * log(p + 1e-300) + x * log(1 - p + 1e-300)
    return ll


def estimate_params_mle_simple(cases_arr, a0=None, b0=None):
    x = np.array(cases_arr, dtype=float)
    if a0 is None or b0 is None:
        a0, b0 = estimate_params_mom_simple(cases_arr)
    best_a, best_b, best_ll = a0, b0, -inf
    for scale in [2.0, 0.5, 0.1]:
        a_grid = np.exp(np.linspace(log(max(best_a * exp(-scale), 0.001)),
                                     log(best_a * exp(scale)), 15))
        b_grid = np.exp(np.linspace(log(max(best_b * exp(-scale), 0.0001)),
                                     log(best_b * exp(scale)), 15))
        for a in a_grid:
            for b in b_grid:
                ll = nb_loglik(x, a, b)
                if ll > best_ll:
                    best_ll, best_a, best_b = ll, a, b
    return best_a, best_b


def mc_quantiles(shape, rate, n_samples=MC_SAMPLES, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    lam = rng.gamma(shape, 1.0 / rate, size=n_samples)
    x = rng.poisson(lam)
    return [int(np.quantile(x, q)) for q in QUANTILES]


def classify_zone(value, thresholds):
    p10, p25, p50, p75, p90 = thresholds
    if value <= p25: return 'sucesso'
    elif value <= p50: return 'seguranca'
    elif value <= p75: return 'alerta'
    elif value <= p90: return 'epidemico'
    else: return 'emergencia'

# ANTES do step1:
input_csv = step0_preprocess_if_needed(args.input)

# Step 1 usa input_csv em vez de args.input:
channel_data = step1_compute_channels(input_csv, args.pop)
age_data = step2_age_group_data(input_csv, channel_data)
# ══════════════════════════════════════════════════════════════════════
# Step 1: Rodar compute_channels.py
# ══════════════════════════════════════════════════════════════════════

def step1_compute_channels(csv_path, pop, channel_json='channel_data.json'):
    """Roda compute_channels.py para gerar os 58 canais principais."""
    print("\n" + "=" * 60)
    print("STEP 1: Computando canais endêmicos principais")
    print("=" * 60)

    cmd = [
        sys.executable, 'compute_channels.py',
        csv_path,
        '--pop', str(pop),
        '--output', channel_json,
        '--agravos', 'all',
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("ERRO: compute_channels.py falhou!")
        sys.exit(1)

    with open(channel_json, 'r') as f:
        data = json.load(f)
    print(f"  → {len(data['channels'])} canais gerados")
    return data


# ══════════════════════════════════════════════════════════════════════
# Step 2: Agregar dados por faixa etária
# ══════════════════════════════════════════════════════════════════════

def step2_age_group_data(csv_path, channel_data):
    """Agrega dados por faixa etária para os 20 agravos prioritários.

    Suporta 4 formatos de CSV:
      A) GAS agregado v2: ano_epi;semana_epi;faixa_etaria;cid_codigo;cid_descricao;quantidade
      B) GAS agregado v1: ano_epi;semana_epi;faixa_etaria;cid_descricao;quantidade
      C) GAS bruto:       data;unidade;sexo;faixa_etaria;cep;cid_descricao
      D) CSV bruto:       data;unidade;usuario;sexo;idade;cep;cid_descricao
    """
    print("\n" + "=" * 60)
    print("STEP 2: Agregando dados por faixa etária")
    print("=" * 60)

    # Importar DESC_TO_CID e funções de compute_channels.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("compute_channels", "compute_channels.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)

    # Ler CSV
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, sep=';', encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    # Detectar colunas
    cols = {c.lower().strip(): c for c in df.columns}
    col_cid = cols.get('cid_descricao', cols.get('cid', 'cid_descricao'))
    has_faixa_etaria = 'faixa_etaria' in cols
    has_idade = 'idade' in cols
    has_ano_epi = 'ano_epi' in cols
    has_data = 'data' in cols

    # ── Determinar ano_epi e se_epi ──────────────────────────────────
    if has_ano_epi and 'semana_epi' in cols:
        # Formato A (GAS agregado): já tem ano_epi e semana_epi
        print("  → Detectado formato GAS agregado (ano_epi + semana_epi)")
        df['ano_epi'] = df[cols['ano_epi']]
        df['se_epi'] = df[cols['semana_epi']]
    elif has_data:
        # Formatos B/C: tem coluna data → calcular SE
        col_date = cols['data']
        df[col_date] = pd.to_datetime(df[col_date], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[col_date])
        if hasattr(cc, 'epi_week'):
            epi = df[col_date].apply(cc.epi_week)
            df['ano_epi'] = epi.apply(lambda x: x[0])
            df['se_epi'] = epi.apply(lambda x: x[1])
        else:
            # Fallback: SE padrão MS/OMS (dom–sáb, quarta define ano)
            from datetime import timedelta as _td
            def _epi_week_fallback(dt):
                dow = dt.isoweekday() % 7       # dom=0 .. sab=6
                sun = dt - _td(days=dow)
                wed = sun + _td(days=3)
                ano = wed.year
                jan1 = pd.Timestamp(ano, 1, 1)
                j1d = jan1.isoweekday() % 7
                if j1d <= 3:
                    fs = jan1 - _td(days=j1d)
                else:
                    fs = jan1 + _td(days=7 - j1d)
                se = (sun - fs).days // 7 + 1
                return (ano, max(1, se))
            epi_fb = df[col_date].apply(_epi_week_fallback)
            df['ano_epi'] = epi_fb.apply(lambda x: x[0])
            df['se_epi'] = epi_fb.apply(lambda x: x[1])
    else:
        print("  ⚠ Sem coluna 'data' nem 'ano_epi' — skip age channels")
        return {}

    # ── Coluna de quantidade ─────────────────────────────────────────
    col_qty = 'quantidade'
    if col_qty not in cols:
        df['quantidade'] = 1
    else:
        col_qty = cols['quantidade']

    # ── Mapear CID ───────────────────────────────────────────────────
    if 'cid_codigo' in cols:
        # Nova coluna cid_codigo disponível — usar diretamente
        df['cid_code'] = df[cols['cid_codigo']].astype(str).str.strip().str.upper()
        df.loc[df['cid_code'].isin(['', 'NAN', 'NONE']), 'cid_code'] = pd.NA
        print(f"  → Coluna cid_codigo encontrada: {df['cid_code'].notna().mean():.0%} com código")
    else:
        # Fallback: mapear descrição → código via DESC_TO_CID
        if hasattr(cc, 'DESC_TO_CID'):
            desc_to_cid = cc.DESC_TO_CID
        else:
            desc_to_cid = {}
        df['cid_code'] = df[col_cid].astype(str).str.strip().str.upper().map(desc_to_cid)

    # ── Classificar faixa etária (dashboard age groups) ──────────────
    # Mapeamento faixas GAS → age groups do dashboard
    FAIXA_TO_AGE_GROUP = {
        '<1':    'Bebês (0-1)',
        '1-4':   'Bebês (0-1)',     # 1 ano → ainda Bebê
        '5-9':   'Crianças (2-11)',
        '10-14': 'Crianças (2-11)', # 10-11 → Crianças, 12-14 → misto
        '15-19': 'Adolescentes (12-17)',  # 15-17 → Adolescentes, 18-19 → misto
        '20-29': 'Adultos jovens (18-39)',
        '30-39': 'Adultos jovens (18-39)',
        '40-49': 'Meia-idade (40-59)',
        '50-59': 'Meia-idade (40-59)',
        '60-69': 'Idosos (60+)',
        '70-79': 'Idosos (60+)',
        '80+':   'Idosos (60+)',
        'NI':    None,
    }
    # Nota: faixas '1-4' inclui idades 1 (Bebê) e 2-4 (Criança);
    # '10-14' inclui 10-11 (Criança) e 12-14 (Adolescente);
    # '15-19' inclui 15-17 (Adolescente) e 18-19 (Adulto jovem).
    # A aproximação é aceitável — as faixas GAS são mais largas que os
    # age groups do dashboard. O resultado é conservador.

    if has_faixa_etaria:
        # Formatos A/B: faixa_etaria é string
        col_age = cols['faixa_etaria']
        print("  → Classificando por faixa_etaria (string)")
        df['faixa'] = df[col_age].astype(str).str.strip().map(FAIXA_TO_AGE_GROUP)
    elif has_idade:
        # Formato C: idade é numérica
        col_age = cols['idade']
        print("  → Classificando por idade (numérica)")
        def parse_age(val):
            try:
                v = str(val).strip()
                num = ''.join(c for c in v.split()[0] if c.isdigit() or c == '.')
                return int(float(num)) if num else None
            except:
                return None

        df['idade_num'] = df[col_age].apply(parse_age)
        df = df.dropna(subset=['idade_num'])
        df['idade_num'] = df['idade_num'].astype(int)

        def get_age_group(age):
            for name, (lo, hi) in AGE_GROUPS.items():
                if lo <= age <= hi:
                    return name
            return None

        df['faixa'] = df['idade_num'].apply(get_age_group)
    else:
        print("  ⚠ Nenhuma coluna de idade/faixa_etaria encontrada — skip age channels")
        return {}

    df = df.dropna(subset=['faixa'])

    # ── Preparar mapeamentos para filtragem ─────────────────────
    # Mapeamento reverso SINAN: disease_name → set(CID codes)
    sinan_reverse = {}
    if hasattr(cc, 'SINAN_MAP'):
        for code, disease in cc.SINAN_MAP.items():
            sinan_reverse.setdefault(disease, set()).add(code)

    # Mapeamento reverso capítulo: chapter_name → first_letter(s)
    chapter_reverse = {}
    if hasattr(cc, 'CID_CHAPTERS'):
        for prefix, chapter_name in cc.CID_CHAPTERS.items():
            chapter_reverse.setdefault(chapter_name, set()).add(prefix)

    # Classificar cid_code → SINAN disease e cid_code → chapter
    if 'cid_code' in df.columns:
        df['_sinan'] = df['cid_code'].apply(
            lambda x: cc.cid_to_sinan(x) if hasattr(cc, 'cid_to_sinan') else 'Outros')
        df['_chapter'] = df['cid_code'].apply(
            lambda x: cc.cid_to_chapter(x) if hasattr(cc, 'cid_to_chapter') else None)
    else:
        df['_sinan'] = 'Outros'
        df['_chapter'] = None

    # ── Agregar por agravo × faixa × ano × SE ────────────────────
    available_agravos = list(channel_data['channels'].keys())
    selected = [a for a in KEY_AGRAVOS_AGE if a in available_agravos]

    age_data = {}
    for agravo in selected:
        age_data[agravo] = {}

        for faixa in AGE_GROUPS:
            df_f = df[df['faixa'] == faixa]

            # Filtrar por agravo (lógica depende do tipo)
            if agravo == "Todos os atendimentos":
                df_a = df_f
            elif agravo.startswith("SINAN: "):
                # SINAN: filtrar por doença SINAN
                sinan_disease = agravo[7:]  # strip "SINAN: "
                df_a = df_f[df_f['_sinan'] == sinan_disease]
            elif any(agravo.startswith(f"{rom} -") for rom in
                     ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII',
                      'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV',
                      'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', 'XXII']):
                # Capítulo CID
                df_a = df_f[df_f['_chapter'] == agravo]
            elif hasattr(cc, 'SYNDROME_DEFS') and agravo in cc.SYNDROME_DEFS:
                # Síndrome: filtrar por códigos CID da definição
                syn_codes = cc.SYNDROME_DEFS[agravo]
                if 'cid_code' in df.columns:
                    mask = df_f['cid_code'].apply(
                        lambda x: any(str(x).startswith(c) for c in syn_codes)
                        if pd.notna(x) else False)
                    df_a = df_f[mask]
                else:
                    df_a = df_f
            else:
                # CID individual: extrair código do nome (ex: "A90 - DENGUE...")
                cid_code_match = agravo.split(' - ')[0].strip() if ' - ' in agravo else agravo
                if 'cid_code' in df.columns:
                    df_a = df_f[df_f['cid_code'] == cid_code_match]
                else:
                    df_a = df_f

            if len(df_a) == 0:
                continue

            # Somar quantidade por ano × SE (suporta dados pré-agregados)
            counts = df_a.groupby(['ano_epi', 'se_epi'])[col_qty].sum().reset_index(name='n')
            year_se = {}
            for _, row in counts.iterrows():
                yr = str(int(row['ano_epi']))
                se = str(int(row['se_epi']))
                if int(se) > MAX_SE:
                    continue
                if yr not in year_se:
                    year_se[yr] = {}
                year_se[yr][se] = int(row['n'])

            if year_se:
                age_data[agravo][faixa] = year_se

    print(f"  → {len(age_data)} agravos × faixas etárias agregados")
    return age_data


# ══════════════════════════════════════════════════════════════════════
# Step 3: Computar canais por faixa etária
# ══════════════════════════════════════════════════════════════════════

def step3_age_channels(age_data):
    """Computa canais Gamma-Poisson por (agravo, faixa etária)."""
    print("\n" + "=" * 60)
    print("STEP 3: Computando canais por faixa etária")
    print("=" * 60)

    rng = np.random.default_rng(RNG_SEED)
    results = {}
    total = sum(len(age_data[a]) for a in age_data)
    done = 0

    for agravo in sorted(age_data.keys()):
        results[agravo] = {}
        for age_group in sorted(age_data[agravo].keys()):
            done += 1
            grp = age_data[agravo][age_group]

            channels = {}
            raw = {}
            classifications = {}

            all_years = sorted(set(TRAINING_YEARS_DEFAULT + [MONITOR_YEAR]))
            for yr in all_years:
                raw[yr] = {}
                if yr in grp:
                    for se_s, count in grp[yr].items():
                        se = int(se_s)
                        if 1 <= se <= MAX_SE:
                            raw[yr][se] = int(count)

            for se in range(1, MAX_SE + 1):
                train = [int(grp.get(yr, {}).get(str(se), 0)) for yr in TRAINING_YEARS_DEFAULT]
                if sum(train) == 0:
                    thresholds = [0, 0, 0, 1, 2]
                else:
                    a0, b0 = estimate_params_mom_simple(train)
                    try:
                        a, b = estimate_params_mle_simple(train, a0, b0)
                    except:
                        a, b = a0, b0
                    thresholds = mc_quantiles(a, b, n_samples=MC_SAMPLES, rng=rng)

                channels[se] = {
                    'p10': thresholds[0], 'p25': thresholds[1],
                    'p50': thresholds[2], 'p75': thresholds[3], 'p90': thresholds[4]
                }

            for yr in all_years:
                classifications[yr] = {}
                for se in range(1, MAX_SE + 1):
                    val = raw.get(yr, {}).get(se, 0)
                    if se in channels:
                        th = [channels[se]['p10'], channels[se]['p25'],
                              channels[se]['p50'], channels[se]['p75'], channels[se]['p90']]
                        classifications[yr][str(se)] = classify_zone(val, th)

            raw_str = {yr: {str(k): v for k, v in raw[yr].items()} for yr in raw}
            channels_str = {str(k): v for k, v in channels.items()}

            results[agravo][age_group] = {
                'channels': channels_str,
                'raw': raw_str,
                'classifications': classifications
            }

            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] {agravo} / {age_group}")

    print(f"  → {total} combinações computadas")
    return results


# ══════════════════════════════════════════════════════════════════════
# Step 4: Gerar boletim enriquecido
# ══════════════════════════════════════════════════════════════════════

def step4_boletim(channel_data):
    """Gera BOLETIM_DATA enriquecido a partir dos canais."""
    print("\n" + "=" * 60)
    print("STEP 4: Gerando boletim enriquecido")
    print("=" * 60)

    channels = channel_data['channels']

    # Selecionar agravos prioritários para o boletim
    priority_agravos = [
        ("SINAN: Dengue", "ALTA"),
        ("SINAN: Diarréia/gastroenterite", "ALTA"),
        ("SINAN: Pneumonia NE", "ALTA"),
        ("Todos os atendimentos", "MODERADA"),
        ("I - Doenças infecciosas e parasitárias", "ALTA"),
        ("X - Aparelho respiratório", "MODERADA"),
        ("XVIII - Sintomas e sinais", "MODERADA"),
        ("XIII - Sistema osteomuscular", "MODERADA"),
        ("SINAN: Influenza NE", "MODERADA"),
        ("V - Transtornos mentais", "MODERADA"),
        ("IX - Aparelho circulatório", "BAIXA"),
        ("XIV - Aparelho geniturinário", "BAIXA"),
        ("XII - Pele e tecido subcutâneo", "BAIXA"),
        ("SINAN: Sífilis NE", "BAIXA"),
    ]

    # Busca flexível: tenta match exato, depois parcial (case-insensitive)
    def find_channel(name, channels):
        if name in channels:
            return name, channels[name]
        name_upper = name.upper()
        for key in channels:
            if name_upper in key.upper() or key.upper() in name_upper:
                return key, channels[key]
        # Tentar sem prefixo "SINAN: "
        if name.startswith("SINAN: "):
            short = name[7:]
            for key in channels:
                if short.upper() in key.upper():
                    return key, channels[key]
        return None, None

    boletim = []
    for name, prio in priority_agravos:
        matched_name, ch = find_channel(name, channels)
        if not ch:
            continue
        name = matched_name  # usar o nome real do canal

        se_list = ch['se_list']
        raw = ch.get('raw', [])
        years = ch['years']

        # Totals
        total_2025 = sum(r.get('c2025', 0) for r in raw)
        total_2026 = sum(r.get('c2026', 0) for r in raw)
        hist_years = [y for y in years if 2022 <= y <= 2024]
        hist_totals = [sum(r.get(f'c{y}', 0) for r in raw) for y in hist_years]
        media_hist = int(np.mean(hist_totals)) if hist_totals else 0

        var_pct = round((total_2025 - media_hist) / max(media_hist, 1) * 100, 1)

        # Peak
        vals_2025 = [(r.get('c2025', 0), se) for r, se in zip(raw, se_list)]
        pico_val, pico_se = max(vals_2025, key=lambda x: x[0])

        # SE > P90
        cls_2025 = ch.get('classifications', {}).get('2025', [])
        se_p90 = sum(1 for z in cls_2025 if z == 'emergencia')

        # Last SE 2026 with data
        last_se = 0
        for r, se in zip(raw, se_list):
            if r.get('c2026', 0) > 0:
                last_se = se

        # Classifications
        zone_counts_2025 = {'sucesso': 0, 'seguranca': 0, 'alerta': 0, 'epidemico': 0, 'emergencia': 0}
        for z in cls_2025:
            if z in zone_counts_2025:
                zone_counts_2025[z] += 1

        cls_2026 = ch.get('classifications', {}).get('2026', [])
        zone_counts_2026 = {'sucesso': 0, 'seguranca': 0, 'alerta': 0, 'epidemico': 0, 'emergencia': 0}
        for z in cls_2026:
            if z in zone_counts_2026:
                zone_counts_2026[z] += 1

        # Sazonalidade
        sazon_se = pico_se

        # Tendência text
        if var_pct > 10:
            tend = f"Aumento de {var_pct}% em 2025 vs média 2022-2024."
        elif var_pct < -10:
            tend = f"Redução de {abs(var_pct)}% em 2025 vs média 2022-2024."
        else:
            tend = f"Estável de {var_pct}% em 2025 vs média 2022-2024."

        # Ação
        if prio == "ALTA":
            acao = "Manter vigilância ativa."
        elif prio == "MODERADA":
            acao = "Manter vigilância ativa."
        else:
            acao = "Monitoramento de rotina."

        # Última SE zona
        ultima_zona = cls_2026[last_se - 1] if last_se > 0 and last_se <= len(cls_2026) else 'sem dados'

        # Obs/P90 última SE
        obs_ult = raw[last_se - 1].get('c2026', 0) if last_se > 0 else 0
        ch_2026 = ch.get('channels', {}).get('2026', [])
        p90_ult = ch_2026[last_se - 1][4] if ch_2026 and last_se > 0 else 1
        p50_ult = ch_2026[last_se - 1][2] if ch_2026 and last_se > 0 else 1

        boletim.append({
            'name': name,
            'prioridade': prio,
            'tendencia': tend,
            'sazonalidade': f"Pico na SE {sazon_se}.",
            'acao': acao,
            'total_2025': total_2025,
            'se_p90_2025': se_p90,
            'total_2026': total_2026,
            'se_2026': last_se,
            'variacao_pct': var_pct,
            'media_hist': media_hist,
            'classificacao_2025': zone_counts_2025,
            'classificacao_2026': zone_counts_2026,
            'pico_val_2025': pico_val,
            'pico_se_2025': pico_se,
            'pico_val_2026': max((r.get('c2026', 0) for r in raw), default=0),
            'pico_se_2026': max(((r.get('c2026', 0), se) for r, se in zip(raw, se_list)),
                                key=lambda x: x[0], default=(0, 1))[1],
            'ultima_se_zona': ultima_zona,
            'ultima_se_obs': obs_ult,
            'ultima_se_p90': p90_ult,
            'ultima_se_p50': p50_ult,
            'media_semanal_2026': round(total_2026 / max(last_se, 1), 1),
            'media_semanal_2025': round(total_2025 / 52, 1),
        })

    print(f"  → {len(boletim)} agravos no boletim")
    return boletim


# ══════════════════════════════════════════════════════════════════════
# Step 5: Montar HTML final
# ══════════════════════════════════════════════════════════════════════

def step5_generate_html(channel_data, age_data, age_channels, boletim,
                        template_html, output_html):
    """Atualiza o HTML com os dados computados e timestamps atuais."""
    print("\n" + "=" * 60)
    print("STEP 5: Gerando HTML final")
    print("=" * 60)

    with open(template_html, 'r') as f:
        html = f.read()

    now = datetime.now()
    now_br = now.strftime('%d/%m/%Y')

    # Detectar última data dos dados
    # Procurar último SE com dados em 2026
    ch_total = channel_data['channels'].get('Total de atendimentos', {})
    raw = ch_total.get('raw', [])
    se_list = ch_total.get('se_list', [])
    last_se = 0
    for r, se in zip(raw, se_list):
        if r.get('c2026', 0) > 0:
            last_se = se

    # 1. Atualizar DATA
    data_json = json.dumps(channel_data, separators=(',', ':'), ensure_ascii=False)
    import re
    html = re.sub(r'const DATA = \{.*?\};\s*\n', '', html, count=1, flags=re.DOTALL)
    # Insert after the opening script tag
    script_pos = html.find('<script type="text/babel">') + len('<script type="text/babel">')
    html = html[:script_pos] + f"\nconst DATA = {data_json};\n" + html[script_pos:]

    # 2. Atualizar AGE_GROUP_DATA
    age_json = json.dumps(age_data, separators=(',', ':'), ensure_ascii=False)
    html = re.sub(r'const AGE_GROUP_DATA = \{.*?\};\s*\n', '', html, count=1, flags=re.DOTALL)
    insert_after = html.find('const DATA = ')
    insert_after = html.find(';\n', insert_after) + 2
    html = html[:insert_after] + f"const AGE_GROUP_DATA = {age_json};\n" + html[insert_after:]

    # 3. Atualizar AGE_CHANNELS (formato compacto)
    ac_compact = build_age_channels_compact(age_channels)
    ac_json = json.dumps(ac_compact, separators=(',', ':'), ensure_ascii=False)
    html = re.sub(r'const AGE_CHANNELS = \{.*?\};\s*\n', '', html, count=1, flags=re.DOTALL)
    insert_after = html.find('const AGE_COLORS = ')
    insert_after = html.find('};\n', insert_after) + 3
    html = html[:insert_after] + f"const AGE_CHANNELS = {ac_json};\n" + html[insert_after:]

    # 4. Atualizar BOLETIM_DATA
    bol_json = json.dumps(boletim, separators=(',', ':'), ensure_ascii=False)
    html = re.sub(r'const BOLETIM_DATA = \[.*?\];\s*\n', '', html, count=1, flags=re.DOTALL)
    insert_after = html.find('const AGE_CHANNELS = ')
    insert_after = html.find(';\n', insert_after) + 2
    html = html[:insert_after] + f"const BOLETIM_DATA = {bol_json};\n" + html[insert_after:]

    # 5. Atualizar datas no HTML
    # Dashboard header
    html = re.sub(
        r'Última extração: <b[^>]*>[\d/]+</b>',
        f'Última extração: <b style={{{{ color: "#fbbf24" }}}}>{now_br}</b>',
        html
    )
    html = re.sub(
        r'Dados até: <b[^>]*>[\d/]+</b> \(SE \d+\)',
        f'Dados até: <b style={{{{ color: "#60a5fa" }}}}>{now_br}</b> (SE {last_se})',
        html
    )
    html = re.sub(
        r'Gerado em: <b[^>]*>[\d/]+</b>',
        f'Gerado em: <b style={{{{ color: "#a5b4fc" }}}}>{now_br}</b>',
        html
    )

    # Boletim header
    html = re.sub(
        r'SE \d+/2026',
        f'SE {last_se}/2026',
        html
    )

    with open(output_html, 'w') as f:
        f.write(html)

    size_mb = os.path.getsize(output_html) / (1024 * 1024)
    print(f"  → {output_html}: {size_mb:.1f} MB")
    print(f"  → Datas atualizadas: extração={now_br}, SE={last_se}/2026")


def build_age_channels_compact(ac_data):
    """Converte age_channels para formato compacto do dashboard."""
    result = {}
    for agravo in ac_data:
        result[agravo] = {}
        for age_group in ac_data[agravo]:
            agd = ac_data[agravo][age_group]
            ch = agd['channels']
            raw = agd['raw']
            se_list = list(range(1, 53))
            years = sorted(raw.keys(), key=lambda x: int(x))

            channels_compact = {}
            for yr in years:
                yr_ch = []
                for se in range(1, 53):
                    se_s = str(se)
                    c = ch.get(se_s, {'p10': 0, 'p25': 0, 'p50': 0, 'p75': 0, 'p90': 0})
                    yr_ch.append([c['p10'], c['p25'], c['p50'], c['p75'], c['p90']])
                channels_compact[yr] = yr_ch

            raw_compact = []
            for se in range(1, 53):
                entry = {}
                for yr in years:
                    entry[f'c{yr}'] = raw.get(yr, {}).get(str(se), 0)
                raw_compact.append(entry)

            result[agravo][age_group] = {
                'years': [int(y) for y in years],
                'se_list': se_list,
                'channels': channels_compact,
                'raw': raw_compact,
            }
    return result


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline completo: CSV → Dashboard HTML atualizado')
    parser.add_argument('input', help='CSV de entrada (dados brutos IDS Saúde)')
    parser.add_argument('--pop', type=int, default=210000,
                        help='População do município (default: 210000)')
    parser.add_argument('--output', '-o', default='index.html',
                        help='HTML de saída (default: index.html)')
    parser.add_argument('--template', default='index.html',
                        help='HTML template (default: index.html existente)')
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Pipeline Canal Endêmico — Atualização Automática     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  CSV: {args.input}")
    print(f"  Pop: {args.pop}")
    print(f"  Output: {args.output}")

    # Step 1
    channel_data = step1_compute_channels(args.input, args.pop)

    # Step 2
    age_data = step2_age_group_data(args.input, channel_data)

    # Step 3
    age_channels = step3_age_channels(age_data)

    # Step 4
    boletim = step4_boletim(channel_data)

    # Step 5
    step5_generate_html(
        channel_data, age_data, age_channels, boletim,
        args.template, args.output
    )

    print("\n✓ Pipeline concluído com sucesso!")
    print(f"  Dashboard atualizado: {args.output}")


if __name__ == '__main__':
    main()
