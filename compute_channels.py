#!/usr/bin/env python3
"""
Canal Endêmico Bayesiano Hierárquico — Gamma-Poisson
=====================================================
Implementação pura numpy (sem scipy).

Modelo:
    X_si | λ_si ~ Poisson(λ_si · e_i)
    λ_si ~ Gamma(a_s, b_s)

    Marginalização → X_si ~ BinNeg(a_s, p_s)  com  p_s = b_s/(b_s + e_i)

Estimação: Método dos Momentos (MoM) com refinamento por grid-search
           em log-verossimilhança da Binomial Negativa.

Quantis: Monte Carlo (500 000 amostras) — preciso e robusto sem scipy.

Saída: JSON compacto para dashboard React/Recharts.

Autor: Pipeline epidemiológico UPAs Rio Claro
"""

import json
import sys
import argparse
import warnings
from math import lgamma, log, exp, inf
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constantes ────────────────────────────────────────────────────────
MC_SAMPLES = 500_000
QUANTILES  = [0.10, 0.25, 0.50, 0.75, 0.90]
ZONE_NAMES = ['sucesso', 'seguranca', 'alerta', 'epidemico', 'emergencia']
RNG_SEED   = 2026
MAX_SE     = 52          # SE 53 excluída por padrão (poucos dados)
FALLBACK_SHAPE = 0.1     # para SE com todos os anos = 0
FALLBACK_RATE  = 1.0

# ── Funções auxiliares ────────────────────────────────────────────────

def epi_week(date):
    """Semana epidemiológica MS/OMS (domingo–sábado).
    Retorna (ano_epi, se).
    """
    from datetime import timedelta
    # Ajusta para domingo = dia 0 da semana
    dow = date.isoweekday() % 7  # dom=0, seg=1, ..., sab=6
    # Início da semana epi (domingo)
    sun = date - timedelta(days=dow)
    # Dia de referência: quarta-feira da mesma SE
    wed = sun + timedelta(days=3)
    ano_epi = wed.year
    # Primeiro domingo do ano epi
    jan1 = pd.Timestamp(ano_epi, 1, 1)
    jan1_dow = jan1.isoweekday() % 7
    if jan1_dow <= 3:
        first_sun = jan1 - timedelta(days=jan1_dow)
    else:
        first_sun = jan1 + timedelta(days=7 - jan1_dow)
    se = (sun - first_sun).days // 7 + 1
    if se < 1:
        return epi_week(date - timedelta(days=7))
    return ano_epi, se


def nb_loglik(x_arr, shape, rate, exposure):
    """Log-verossimilhança da Binomial Negativa marginal.

    X ~ NB(r, p) com r=shape, p=rate/(rate+exposure).
    """
    r = shape
    p = rate / (rate + exposure)
    ll = 0.0
    for x in x_arr:
        x = int(x)
        ll += lgamma(x + r) - lgamma(r) - lgamma(x + 1)
        ll += r * log(p) + x * log(1 - p + 1e-300)
    return ll


def estimate_params_mom(cases_arr, exposures_arr):
    """Método dos Momentos para (shape, rate) da Gamma prior.

    Se x_i ~ NB(a, b/(b+e_i)):
        E[X_i] = a * e_i / b
        Var[X_i] = a * e_i / b * (1 + e_i / b)

    Com exposição constante (simplificação):
        mean ≈ a * e_bar / b
        var  ≈ mean * (1 + e_bar / b)
    """
    x = np.array(cases_arr, dtype=float)
    e = np.array(exposures_arr, dtype=float)

    # Taxa observada por 100k
    rates = x / np.maximum(e, 1e-10)
    m = np.mean(rates)
    v = np.var(rates, ddof=1) if len(rates) > 1 else m + 1

    if m <= 0 or v <= 0:
        return FALLBACK_SHAPE, FALLBACK_RATE

    # MoM: shape = m^2 / (v - m), rate = m / (v - m)
    # Mas var da taxa = a/b^2, mean da taxa = a/b
    # → b = m / (v - 0), a = m * b ...
    # Para Gamma(a,b) com E=a/b, Var=a/b^2:
    #   a = m^2 / v,  b = m / v
    if v < 1e-10:
        v = m  # Poisson assumption

    b_hat = m / v
    a_hat = m * b_hat

    # Clamp
    a_hat = max(a_hat, 0.01)
    b_hat = max(b_hat, 0.001)

    return a_hat, b_hat


def estimate_params_mle(cases_arr, exposures_arr, a0=None, b0=None):
    """MLE via grid refinement em torno do MoM.

    Sem scipy, fazemos busca em grade 2D em log-space.
    """
    x = np.array(cases_arr, dtype=float)
    e = np.array(exposures_arr, dtype=float)

    if a0 is None or b0 is None:
        a0, b0 = estimate_params_mom(cases_arr, exposures_arr)

    best_a, best_b = a0, b0
    best_ll = -inf

    # Busca em 3 escalas
    for scale in [2.0, 0.5, 0.1]:
        a_grid = np.exp(np.linspace(
            log(max(best_a * exp(-scale), 0.001)),
            log(best_a * exp(scale)),
            21
        ))
        b_grid = np.exp(np.linspace(
            log(max(best_b * exp(-scale), 0.0001)),
            log(best_b * exp(scale)),
            21
        ))

        for a in a_grid:
            for b in b_grid:
                # Usar exposição média para simplificar
                e_mean = np.mean(e)
                ll = nb_loglik(x, a, b, e_mean)
                if ll > best_ll:
                    best_ll = ll
                    best_a, best_b = a, b

    return best_a, best_b


def mc_quantiles(shape, rate, exposure, quantiles=QUANTILES, n_samples=MC_SAMPLES, rng=None):
    """Quantis preditivos via Monte Carlo Gamma-Poisson."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    # λ ~ Gamma(shape, 1/rate)
    lam = rng.gamma(shape, 1.0 / rate, size=n_samples)
    # X ~ Poisson(λ * exposure)
    x = rng.poisson(lam * exposure)

    return [int(np.quantile(x, q)) for q in quantiles]


def classify_zone(value, thresholds):
    """Classifica valor em zona epidêmica.
    thresholds = [p10, p25, p50, p75, p90]
    """
    p10, p25, p50, p75, p90 = thresholds
    if value <= p25:
        return 'sucesso'
    elif value <= p50:
        return 'seguranca'
    elif value <= p75:
        return 'alerta'
    elif value <= p90:
        return 'epidemico'
    else:
        return 'emergencia'


# ── Detecção de SE incompleta ─────────────────────────────────────────

def detectar_se_incompleta(df, col_se, col_ano, col_casos, col_data=None):
    """Detecta se a última SE do ano mais recente está incompleta."""
    ultimo_ano = int(df[col_ano].max())
    df_ua = df[df[col_ano] == ultimo_ano]
    ultima_se = int(df_ua[col_se].max())
    df_use = df_ua[df_ua[col_se] == ultima_se]

    vol_ultima = df_use[col_casos].sum()
    se_ant = ultima_se - 1
    df_ant = df_ua[df_ua[col_se] == se_ant]
    vol_ant = df_ant[col_casos].sum() if len(df_ant) > 0 and se_ant >= 1 else vol_ultima
    ratio = vol_ultima / max(vol_ant, 1)

    criterios_ok = 0
    dias = 7  # default se não temos data individual
    dia_sem = 7

    if col_data and col_data in df.columns:
        dias = df_use[col_data].dt.date.nunique()
        dia_sem = pd.to_datetime(df_use[col_data]).max().isoweekday()
        if dias >= 6:
            criterios_ok += 1
        if dia_sem >= 5:
            criterios_ok += 1
    else:
        criterios_ok += 2  # sem data individual, assume OK para 2 critérios

    if ratio >= 0.50:
        criterios_ok += 1

    completa = criterios_ok >= 2
    return {
        'ultima_se': ultima_se,
        'ano': ultimo_ano,
        'completa': completa,
        'dias': dias,
        'ratio': round(ratio, 2),
        'decisao': 'INCLUIR' if completa else 'EXCLUIR'
    }


# ── Pipeline principal ────────────────────────────────────────────────

def compute_endemic_channel(
    df_agg,
    populations,
    agravo_name="Todos",
    leave_one_out=True,
    use_mle=True,
    monitor_year=None
):
    """
    Computa canal endêmico para um agravo.

    Parâmetros:
        df_agg: DataFrame com colunas [ano, se, casos]
                Já agregado (uma linha por ano×SE).
        populations: dict {ano: população}
        agravo_name: nome do agravo (para metadados)
        leave_one_out: se True, exclui o ano monitorado da estimação
        use_mle: se True, usa MLE; senão, MoM
        monitor_year: ano específico a monitorar (None = todos)

    Retorna:
        dict com estrutura para JSON do dashboard
    """
    rng = np.random.default_rng(RNG_SEED)

    years = sorted(df_agg['ano'].unique())
    all_se = sorted(df_agg['se'].unique())
    all_se = [s for s in all_se if s <= MAX_SE]

    if monitor_year:
        monitor_years = [monitor_year]
    else:
        monitor_years = years

    # Construir matriz ano × SE
    matrix = {}
    for _, row in df_agg.iterrows():
        a, s, c = int(row['ano']), int(row['se']), int(row['casos'])
        if s > MAX_SE:
            continue
        matrix[(a, s)] = c

    # Preencher zeros para SE×ano ausentes
    for y in years:
        for s in all_se:
            if (y, s) not in matrix:
                matrix[(y, s)] = 0

    # Exposições (pop / 100_000)
    exposures = {}
    for y in years:
        pop = populations.get(y, populations.get(str(y), 100_000))
        exposures[y] = pop / 100_000

    # RAW data para o dashboard
    raw = []
    for s in all_se:
        entry = {'se': s}
        for y in years:
            entry[f'c{y}'] = matrix.get((y, s), 0)
        raw.append(entry)

    # Canal por ano monitorado (leave-one-out)
    channels = {}
    params = {}

    for mon_year in monitor_years:
        if leave_one_out:
            train_years = [y for y in years if y != mon_year]
        else:
            train_years = years

        if len(train_years) < 2:
            train_years = years  # fallback: usa todos se poucos anos

        channel_se = []
        params_se = []

        for s in all_se:
            cases_train = [matrix.get((y, s), 0) for y in train_years]
            exp_train = [exposures.get(y, 1.0) for y in train_years]

            # Verificar se todos são zero
            if sum(cases_train) == 0:
                a_s, b_s = FALLBACK_SHAPE, FALLBACK_RATE
            elif use_mle:
                a_s, b_s = estimate_params_mle(cases_train, exp_train)
            else:
                a_s, b_s = estimate_params_mom(cases_train, exp_train)

            # Exposição do ano monitorado
            e_mon = exposures.get(mon_year, 1.0)

            # Quantis preditivos
            qs = mc_quantiles(a_s, b_s, e_mon, rng=rng)
            channel_se.append(qs)
            params_se.append({'shape': round(a_s, 4), 'rate': round(b_s, 4)})

        channels[str(mon_year)] = channel_se
        params[str(mon_year)] = params_se

    # Classificações por ano monitorado
    classifications = {}
    for mon_year in monitor_years:
        ch = channels[str(mon_year)]
        clf = []
        for s_idx, s in enumerate(all_se):
            obs = matrix.get((mon_year, s), 0)
            zone = classify_zone(obs, ch[s_idx])
            clf.append(zone)
        classifications[str(mon_year)] = clf

    # Exceedance ratio (obs / P90)
    exceedance = {}
    for mon_year in monitor_years:
        ch = channels[str(mon_year)]
        exc = []
        for s_idx, s in enumerate(all_se):
            obs = matrix.get((mon_year, s), 0)
            p90 = ch[s_idx][4]  # index 4 = P90
            ratio = obs / max(p90, 1)
            exc.append(round(ratio, 3))
        exceedance[str(mon_year)] = exc

    # KPIs por ano
    kpis = {}
    for mon_year in monitor_years:
        ch = channels[str(mon_year)]
        cases_year = [matrix.get((mon_year, s), 0) for s in all_se]
        se_above_p90 = sum(
            1 for s_idx, s in enumerate(all_se)
            if matrix.get((mon_year, s), 0) > ch[s_idx][4]
        )
        kpis[str(mon_year)] = {
            'total': sum(cases_year),
            'pico': max(cases_year),
            'pico_se': all_se[cases_year.index(max(cases_year))],
            'se_acima_p90': se_above_p90,
        }

    return {
        'agravo': agravo_name,
        'years': [int(y) for y in years],
        'se_list': [int(s) for s in all_se],
        'populations': {str(k): int(v) for k, v in populations.items()},
        'raw': raw,
        'channels': channels,
        'params': params,
        'classifications': classifications,
        'exceedance': exceedance,
        'kpis': kpis,
    }


# ── Agregação de dados brutos ────────────────────────────────────────

def aggregate_raw_data(df, col_date, col_cid, col_qty='quantidade',
                       group_by='chapter', sinan_only=False):
    """
    Agrega dados brutos em formato SE × ano × agravo.

    Parâmetros:
        df: DataFrame com dados de atendimentos
        col_date: nome da coluna de data
        col_cid: nome da coluna de CID (código ou descrição)
        col_qty: nome da coluna de quantidade
        group_by: 'chapter' (capítulo CID), 'cid' (CID individual),
                  'sinan' (agravos SINAN), 'all' (todos atendimentos)
        sinan_only: se True, filtra apenas CIDs de notificação SINAN

    Retorna:
        dict {agravo_name: DataFrame[ano, se, casos]}
    """
    # Detectar SE
    df = df.copy()
    if 'ano_epi' not in df.columns or 'semana_epi' not in df.columns:
        dates = pd.to_datetime(df[col_date], dayfirst=True, errors='coerce')
        df['ano_epi'] = 0
        df['semana_epi'] = 0
        for idx in df.index:
            if pd.notna(dates[idx]):
                ae, se = epi_week(dates[idx])
                df.at[idx, 'ano_epi'] = ae
                df.at[idx, 'semana_epi'] = se

    # Mapear CID para grupo
    if group_by == 'all':
        df['_grupo'] = 'Todos os atendimentos'
    elif group_by == 'chapter':
        df['_grupo'] = df[col_cid].apply(cid_to_chapter)
    elif group_by == 'sinan':
        df['_grupo'] = df[col_cid].apply(cid_to_sinan)
        df = df[df['_grupo'] != 'Outros']
    else:
        df['_grupo'] = df[col_cid]

    # Agregar
    result = {}
    for grupo, gdf in df.groupby('_grupo'):
        if pd.isna(grupo) or grupo is None or str(grupo).strip() == '':
            continue
        agg = gdf.groupby(['ano_epi', 'semana_epi'])[col_qty].sum().reset_index()
        agg.columns = ['ano', 'se', 'casos']
        agg = agg[agg['se'] <= MAX_SE]
        result[str(grupo)] = agg

    return result


# ── Mapeamento CID ───────────────────────────────────────────────────

CID_CHAPTERS = {
    'A': 'I - Doenças infecciosas e parasitárias',
    'B': 'I - Doenças infecciosas e parasitárias',
    'C': 'II - Neoplasias',
    'D0': 'II - Neoplasias',
    'D1': 'II - Neoplasias',
    'D2': 'II - Neoplasias',
    'D3': 'II - Neoplasias',
    'D4': 'II - Neoplasias',
    'D5': 'III - Sangue e órgãos hematopoéticos',
    'D6': 'III - Sangue e órgãos hematopoéticos',
    'D7': 'III - Sangue e órgãos hematopoéticos',
    'D8': 'III - Sangue e órgãos hematopoéticos',
    'D9': 'III - Sangue e órgãos hematopoéticos',  # D80-D89 = IV, simplificação
    'E': 'IV - Endócrinas, nutricionais e metabólicas',
    'F': 'V - Transtornos mentais',
    'G': 'VI - Sistema nervoso',
    'H0': 'VII - Olho e anexos',
    'H1': 'VII - Olho e anexos',
    'H2': 'VII - Olho e anexos',
    'H3': 'VII - Olho e anexos',
    'H4': 'VII - Olho e anexos',
    'H5': 'VII - Olho e anexos',
    'H6': 'VIII - Ouvido e apófise mastóide',
    'H7': 'VIII - Ouvido e apófise mastóide',
    'H8': 'VIII - Ouvido e apófise mastóide',
    'H9': 'VIII - Ouvido e apófise mastóide',
    'I': 'IX - Aparelho circulatório',
    'J': 'X - Aparelho respiratório',
    'K': 'XI - Aparelho digestivo',
    'L': 'XII - Pele e tecido subcutâneo',
    'M': 'XIII - Sistema osteomuscular',
    'N': 'XIV - Aparelho geniturinário',
    'O': 'XV - Gravidez, parto e puerpério',
    'P': 'XVI - Afecções perinatais',
    'Q': 'XVII - Malformações congênitas',
    'R': 'XVIII - Sintomas e sinais',
    'S': 'XIX - Lesões e causas externas',
    'T': 'XIX - Lesões e causas externas',
    'U': 'XXII - Códigos para propósitos especiais',
    'V': 'XX - Causas externas de morbidade',
    'W': 'XX - Causas externas de morbidade',
    'X': 'XX - Causas externas de morbidade',
    'Y': 'XX - Causas externas de morbidade',
    'Z': 'XXI - Fatores que influenciam o estado de saúde',
}

def cid_to_chapter(cid_str):
    """Mapeia código ou descrição CID para capítulo."""
    if not cid_str or not isinstance(cid_str, str):
        return None

    code = cid_str.strip().upper()

    # Se é descrição com código entre parênteses: "Dengue (A90)"
    import re
    m = re.search(r'\b([A-Z]\d{2})', code)
    if m:
        code = m.group(1)

    if len(code) < 2 or not code[0].isalpha():
        return None

    # Tentar mapeamento com 2 chars, depois 1
    key2 = code[:2]
    key1 = code[0]

    if key2 in CID_CHAPTERS:
        return CID_CHAPTERS[key2]
    if key1 in CID_CHAPTERS:
        return CID_CHAPTERS[key1]

    return None


# Agravos de notificação SINAN (CIDs de interesse)
SINAN_MAP = {
    'A90': 'Dengue', 'A91': 'Dengue hemorrágica',
    'A92.0': 'Chikungunya', 'A92.8': 'Zika',
    'A01': 'Febre tifóide', 'A09': 'Diarréia/gastroenterite',
    'A15': 'Tuberculose respiratória', 'A16': 'Tuberculose respiratória',
    'A17': 'Tuberculose SNC', 'A18': 'Tuberculose outros órgãos',
    'A19': 'Tuberculose miliar',
    'A27': 'Leptospirose',
    'A30': 'Hanseníase',
    'A33': 'Tétano neonatal', 'A34': 'Tétano obstétrico', 'A35': 'Tétano acidental',
    'A37': 'Coqueluche',
    'A50': 'Sífilis congênita', 'A51': 'Sífilis precoce', 'A52': 'Sífilis tardia', 'A53': 'Sífilis NE',
    'A69.2': 'Doença de Lyme',
    'A77': 'Febre maculosa',
    'A78': 'Febre Q',
    'A82': 'Raiva',
    'A95': 'Febre amarela',
    'B05': 'Sarampo',
    'B06': 'Rubéola',
    'B15': 'Hepatite A', 'B16': 'Hepatite B', 'B17': 'Hepatite C/outras',
    'B18': 'Hepatite crônica viral',
    'B19': 'Hepatite viral NE',
    'B20': 'HIV/AIDS', 'B21': 'HIV/AIDS', 'B22': 'HIV/AIDS', 'B23': 'HIV/AIDS', 'B24': 'HIV NE',
    'B26': 'Caxumba',
    'B50': 'Malária P.falciparum', 'B51': 'Malária P.vivax', 'B52': 'Malária P.malariae', 'B53': 'Malária outras',
    'B54': 'Malária NE',
    'B55': 'Leishmaniose',
    'B57': 'Doença de Chagas',
    'B65': 'Esquistossomose',
    'J09': 'Influenza pandêmica', 'J10': 'Influenza identificada', 'J11': 'Influenza NE',
    'J12': 'Pneumonia viral', 'J18': 'Pneumonia NE',
    'P35.0': 'Rubéola congênita',
    'U04': 'SRAG', 'U07.1': 'COVID-19', 'U07.2': 'COVID-19 suspeito',
}

def cid_to_sinan(cid_str):
    """Mapeia CID para agravo SINAN. Retorna 'Outros' se não for SINAN."""
    if not cid_str or not isinstance(cid_str, str):
        return 'Outros'

    import re
    code = cid_str.strip().upper()
    m = re.search(r'\b([A-Z]\d{2}(?:\.\d{1,2})?)', code)
    if not m:
        return 'Outros'

    code = m.group(1)

    # Tentar match exato, depois sem decimal, depois só 3 chars
    if code in SINAN_MAP:
        return SINAN_MAP[code]
    base = code.split('.')[0]
    if base in SINAN_MAP:
        return SINAN_MAP[base]

    return 'Outros'


def extract_cid_code(desc):
    """Extrai código CID de uma descrição como 'A90 - DENGUE [DENGUE CLÁSSICO]'."""
    import re
    if not desc or not isinstance(desc, str):
        return None
    m = re.match(r'^([A-Z]\d{2}(?:\.\d{1,2})?)\b', desc.strip().upper())
    if m:
        return m.group(1)
    m = re.search(r'\(([A-Z]\d{2}(?:\.\d{1,2})?)\)', desc.strip().upper())
    if m:
        return m.group(1)
    return None


# ── Função principal ──────────────────────────────────────────────────

def run_pipeline(input_file, populations, output_file,
                 agravos='all', col_date='data', col_cid='cid_descricao',
                 col_qty='quantidade', monitor_year=None):
    """
    Pipeline completo: CSV → canais endêmicos → JSON.

    Parâmetros:
        input_file: caminho para CSV com dados brutos ou pré-agregados
        populations: dict {ano: pop} ou int (pop constante)
        output_file: caminho para JSON de saída
        agravos: 'all', 'chapters', 'sinan', 'top_N' (ex: 'top_20'), ou lista de nomes
        col_date, col_cid, col_qty: nomes das colunas
        monitor_year: ano para monitorar (None = todos)
    """
    print(f"[1/5] Lendo dados de {input_file}...")

    # Ler CSV
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(input_file, sep=';', encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        df = pd.read_csv(input_file, encoding='latin-1')

    print(f"   {len(df)} registros, colunas: {list(df.columns)}")

    # Detectar colunas
    cols_lower = {c.lower().strip(): c for c in df.columns}

    if col_date not in df.columns:
        for key in ['data', 'dt_atendimento', 'date', 'dt_notif']:
            if key in cols_lower:
                col_date = cols_lower[key]
                break

    if col_cid not in df.columns:
        for key in ['cid_descricao', 'cid_codigo', 'cid', 'hipotese']:
            if key in cols_lower:
                col_cid = cols_lower[key]
                break

    if col_qty not in df.columns:
        if 'quantidade' in cols_lower:
            col_qty = cols_lower['quantidade']
        else:
            df['quantidade'] = 1
            col_qty = 'quantidade'

    # Extrair código CID se só temos descrição
    if 'cid_codigo' not in df.columns:
        df['cid_codigo'] = df[col_cid].apply(extract_cid_code)

    # Normalizar populações
    if isinstance(populations, (int, float)):
        pop_dict = {}
        for y in df['ano_epi'].unique() if 'ano_epi' in df.columns else range(2020, 2027):
            pop_dict[int(y)] = int(populations)
        populations = pop_dict
    else:
        populations = {int(k): int(v) for k, v in populations.items()}

    print(f"[2/5] Agregando dados por agravo...")

    # Determinar agrupamentos
    results = {}

    # Sempre incluir "Todos"
    agg_all = aggregate_raw_data(df, col_date, col_cid, col_qty, group_by='all')
    results.update(agg_all)

    if agravos in ('all', 'chapters') or (isinstance(agravos, str) and agravos.startswith('top')):
        agg_ch = aggregate_raw_data(df, col_date, 'cid_codigo', col_qty, group_by='chapter')
        results.update(agg_ch)

    if agravos in ('all', 'sinan'):
        agg_sinan = aggregate_raw_data(df, col_date, 'cid_codigo', col_qty, group_by='sinan')
        results.update(agg_sinan)

    if agravos == 'all' or (isinstance(agravos, str) and agravos.startswith('top')):
        # Top N CIDs mais prevalentes
        n = 20
        if isinstance(agravos, str) and agravos.startswith('top_'):
            n = int(agravos.split('_')[1])

        top_cids = (df.groupby('cid_codigo')[col_qty].sum()
                    .sort_values(ascending=False)
                    .head(n))

        for cid_code in top_cids.index:
            if pd.isna(cid_code) or cid_code is None:
                continue
            df_cid = df[df['cid_codigo'] == cid_code].copy()
            desc = df_cid[col_cid].mode().iloc[0] if len(df_cid) > 0 else cid_code
            name = f"{cid_code} - {desc}" if cid_code != desc else cid_code
            agg = df_cid.groupby(['ano_epi', 'semana_epi'])[col_qty].sum().reset_index()
            agg.columns = ['ano', 'se', 'casos']
            agg = agg[agg['se'] <= MAX_SE]
            results[name] = agg

    print(f"   {len(results)} agravos/grupos identificados")

    # Verificar SE incompleta
    print(f"[3/5] Verificando completude da última SE...")
    for name, agg_df in results.items():
        info_se = detectar_se_incompleta(agg_df, 'se', 'ano', 'casos')
        if info_se['decisao'] == 'EXCLUIR':
            print(f"   ⚠ {name}: SE {info_se['ultima_se']}/{info_se['ano']} EXCLUÍDA "
                  f"(ratio={info_se['ratio']})")
            mask = (agg_df['ano'] == info_se['ano']) & (agg_df['se'] == info_se['ultima_se'])
            agg_df.loc[mask, 'casos'] = 0

    print(f"[4/5] Computando canais endêmicos (Gamma-Poisson)...")

    all_channels = {}
    for i, (name, agg_df) in enumerate(results.items()):
        years_available = sorted(agg_df['ano'].unique())
        if len(years_available) < 1:
            continue

        ch = compute_endemic_channel(
            agg_df, populations,
            agravo_name=name,
            leave_one_out=(len(years_available) >= 3),
            use_mle=True,
            monitor_year=monitor_year
        )
        all_channels[name] = ch

        if (i + 1) % 5 == 0 or i == len(results) - 1:
            print(f"   {i+1}/{len(results)} agravos processados...")

    print(f"[5/5] Exportando JSON para {output_file}...")

    output = {
        'metadata': {
            'generated': pd.Timestamp.now().isoformat(),
            'model': 'Gamma-Poisson (hierárquico bayesiano)',
            'estimation': 'MLE com grid-search + Monte Carlo quantiles',
            'mc_samples': MC_SAMPLES,
            'quantiles': QUANTILES,
            'zones': ZONE_NAMES,
            'max_se': MAX_SE,
            'n_agravos': len(all_channels),
            'source': str(input_file),
        },
        'channels': all_channels,
    }

class NumpyEncoder(json.JSONEncoder):
           def default(self, obj):
               if isinstance(obj, (np.integer,)):
                   return int(obj)
               if isinstance(obj, (np.floating,)):
                   return float(obj)
               if isinstance(obj, np.ndarray):
                   return obj.tolist()
               return super().default(obj)
   
       with open(output_file, 'w', encoding='utf-8') as f:
           json.dump(output, f, ensure_ascii=False, indent=None, separators=(',', ':'), cls=NumpyEncoder)

    size_kb = Path(output_file).stat().st_size / 1024
    print(f"\n✓ Concluído! {len(all_channels)} canais → {output_file} ({size_kb:.1f} KB)")

    return output


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Canal Endêmico Bayesiano Hierárquico (Gamma-Poisson)')
    parser.add_argument('input', help='CSV de entrada (dados brutos ou agregados)')
    parser.add_argument('--pop', required=True,
                        help='Populações por ano (JSON dict) ou valor único')
    parser.add_argument('--output', '-o', default='channel_data.json',
                        help='JSON de saída (default: channel_data.json)')
    parser.add_argument('--agravos', default='all',
                        help='Agrupamento: all, chapters, sinan, top_N')
    parser.add_argument('--monitor-year', type=int, default=None,
                        help='Ano específico para monitorar')
    parser.add_argument('--col-date', default='data')
    parser.add_argument('--col-cid', default='cid_descricao')
    parser.add_argument('--col-qty', default='quantidade')

    args = parser.parse_args()

    # Parse populations
    try:
        pop = json.loads(args.pop)
        if isinstance(pop, (int, float)):
            pop = int(pop)
    except json.JSONDecodeError:
        pop = int(args.pop)

    run_pipeline(
        args.input, pop, args.output,
        agravos=args.agravos,
        col_date=args.col_date,
        col_cid=args.col_cid,
        col_qty=args.col_qty,
        monitor_year=args.monitor_year,
    )
