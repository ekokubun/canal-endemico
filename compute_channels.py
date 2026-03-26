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


class NumpyEncoder(json.JSONEncoder):
    """Converte tipos numpy para tipos nativos Python (JSON serializável)."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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


# ── Mapeamento CID descrição → código (tabela CID-10 DATASUS) ────────
# Cobre os CIDs mais prevalentes em UPAs/emergências + todos SINAN

CID_DESC_TO_CODE = {
    # Cap I — Doenças infecciosas e parasitárias (A00-B99)
    'COLERA': 'A00', 'COLERA NAO ESPECIFICADA': 'A00.9',
    'FEBRES TIFOIDE E PARATIFOIDE': 'A01',
    'FEBRE TIFOIDE': 'A01.0', 'FEBRE PARATIFOIDE NAO ESPECIFICADA': 'A01.4',
    'INTOXICACOES ALIMENTARES BACTERIANAS': 'A05',
    'OUTRAS INTOXICACOES ALIMENTARES BACTERIANAS': 'A05.8',
    'AMEBIASE': 'A06', 'AMEBIASE NAO ESPECIFICADA': 'A06.9',
    'DIARREIA E GASTROENTERITE DE ORIGEM INFECCIOSA PRESUMIVEL': 'A09',
    'TUBERCULOSE RESPIRATORIA, COM CONFIRMACAO BACTERIOLOGICA E HISTOLOGICA': 'A15',
    'TUBERCULOSE PULMONAR, COM CONFIRMACAO POR EXAME MICROSCOPICO DE ESCARRO': 'A15.0',
    'TUBERCULOSE RESPIRATORIA, SEM CONFIRMACAO BACTERIOLOGICA OU HISTOLOGICA': 'A16',
    'PESTE': 'A20',
    'BRUCELOSE NAO ESPECIFICADA': 'A23.9',
    'ERISIPELA': 'A46',
    'LEPTOSPIROSE NAO ESPECIFICADA': 'A27.9', 'LEPTOSPIROSE ICTEROHEMMORRAGICA': 'A27.0',
    'HANSENIASE [LEPRA] NAO ESPECIFICADA': 'A30.9',
    'TETANO NEONATAL': 'A33', 'TETANO OBSTETRICO': 'A34',
    'OUTROS TIPOS DE TETANO': 'A35',
    'COQUELUCHE NAO ESPECIFICADA': 'A37.9', 'COQUELUCHE POR BORDETELLA PERTUSSIS': 'A37.0',
    'ESCARLATINA': 'A38',
    'SEPTICEMIA NAO ESPECIFICADA': 'A41.9',
    'SEPTICEMIA NAO ESPECIFICADA (SEPSIS)': 'A41.9',
    'SIFILIS CONGENITA PRECOCE, NAO ESPECIFICADA': 'A50.2',
    'SIFILIS CONGENITA': 'A50',
    'SIFILIS PRECOCE, NAO ESPECIFICADA': 'A51.9',
    'OUTRAS FORMAS DE SIFILIS TARDIA': 'A52.7',
    'SIFILIS NAO ESPECIFICADA': 'A53.9',
    'OUTRAS FORMAS DE SIFILIS E AS NAO ESPECIFICADAS': 'A53',
    'INFECCAO GONOCOCICA NAO ESPECIFICADA': 'A54.9',
    'HERPES ZOSTER SEM COMPLICACOES': 'B02.9',
    'HERPES ZOSTER': 'B02',
    'VARICELA SEM COMPLICACOES': 'B01.9', 'VARICELA [CATAPORA]': 'B01',
    'SARAMPO SEM COMPLICACOES': 'B05.9',
    'RUBEOLA SEM COMPLICACOES': 'B06.9',
    'HEPATITE AGUDA A': 'B15', 'HEPATITE AGUDA A SEM COMA HEPATICO': 'B15.9',
    'HEPATITE AGUDA B': 'B16', 'HEPATITE AGUDA B SEM AGENTE DELTA': 'B16.9',
    'OUTRAS HEPATITES VIRAIS AGUDAS': 'B17',
    'HEPATITE VIRAL CRONICA': 'B18',
    'HEPATITE VIRAL NAO ESPECIFICADA': 'B19.9',
    'DOENCA PELO VIRUS DA IMUNODEFICIENCIA HUMANA [HIV]': 'B24',
    'PAROTIDITE EPIDEMCIA NAO COMPLICADA': 'B26.9', 'CAXUMBA': 'B26',
    'CONJUNTIVITE VIRAL': 'B30',
    'MICOSE NAO ESPECIFICADA': 'B49',
    'MALARIA POR PLASMODIUM FALCIPARUM': 'B50', 'MALARIA POR PLASMODIUM VIVAX': 'B51',
    'LEISHMANIOSE VISCERAL': 'B55.0', 'LEISHMANIOSE CUTANEA': 'B55.1',
    'DOENCA DE CHAGAS AGUDA': 'B57.0', 'DOENCA DE CHAGAS': 'B57',
    'TOXOPLASMOSE NAO ESPECIFICADA': 'B58.9',
    'ESQUISTOSSOMOSE': 'B65',
    'DENGUE [DENGUE CLASSICO]': 'A90', 'DENGUE': 'A90',
    'FEBRE HEMORRAGICA DEVIDA AO VIRUS DO DENGUE': 'A91',
    'FEBRE DE CHIKUNGUNYA': 'A92.0', 'CHIKUNGUNYA': 'A92.0',
    'DOENCA PELO VIRUS ZIKA': 'A92.8', 'ZIKA': 'A92.8',
    'FEBRE AMARELA NAO ESPECIFICADA': 'A95.9', 'FEBRE AMARELA': 'A95',
    'RAIVA': 'A82',
    'FEBRE MACULOSA': 'A77',
    'MENINGITE BACTERIANA NAO ESPECIFICADA': 'G00.9',
    'MENINGITE VIRAL': 'A87',
    # Cap II — Neoplasias (C00-D48)
    'NEOPLASIA MALIGNA DA MAMA NAO ESPECIFICADA': 'C50.9',
    'LEIOMIOMA DO UTERO NAO ESPECIFICADO': 'D25.9',
    # Cap III — Sangue (D50-D89)
    'ANEMIA POR DEFICIENCIA DE FERRO NAO ESPECIFICADA': 'D50.9',
    'ANEMIA NAO ESPECIFICADA': 'D64.9',
    # Cap IV — Endócrinas (E00-E90)
    'DIABETES MELLITUS NAO ESPECIFICADO': 'E14',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM CETOACIDOSE': 'E14.1',
    'DIABETES MELLITUS NAO ESPECIFICADO - SEM COMPLICACOES': 'E14.9',
    'DIABETES MELLITUS INSULINO-DEPENDENTE': 'E10',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE': 'E11',
    'DESIDRATACAO': 'E86', 'DEPLEÇÃO DE VOLUME': 'E86',
    'HIPOGLICEMIA NAO ESPECIFICADA': 'E16.2',
    'HIPOPOTASSEMIA': 'E87.6',
    # Cap V — Transtornos mentais (F00-F99)
    'TRANSTORNOS MENTAIS E COMPORTAMENTAIS DEVIDOS AO USO DE ALCOOL': 'F10',
    'TRANSTORNOS MENTAIS DEVIDOS AO USO DE ALCOOL - SINDROME DE DEPENDENCIA': 'F10.2',
    'TRANSTORNOS MENTAIS DEVIDOS AO USO DE ALCOOL - INTOXICACAO AGUDA': 'F10.0',
    'EPISODIO DEPRESSIVO NAO ESPECIFICADO': 'F32.9',
    'TRANSTORNO ANSIOSO NAO ESPECIFICADO': 'F41.9',
    'TRANSTORNO AFETIVO BIPOLAR NAO ESPECIFICADO': 'F31.9',
    'ESQUIZOFRENIA NAO ESPECIFICADA': 'F20.9',
    # Cap VI — Sistema nervoso (G00-G99)
    'MENINGITE BACTERIANA NAO CLASSIFICADA EM OUTRA PARTE': 'G00',
    'EPILEPSIA NAO ESPECIFICADA': 'G40.9',
    'ENXAQUECA NAO ESPECIFICADA': 'G43.9', 'ENXAQUECA': 'G43',
    'OUTRAS CEFALEIAS': 'G44',
    'PARALISIA DE BELL': 'G51.0',
    'VERTIGEM PAROXISTICA BENIGNA': 'H81.1',
    # Cap VII — Olho (H00-H59)
    'CONJUNTIVITE NAO ESPECIFICADA': 'H10.9', 'CONJUNTIVITE AGUDA': 'H10.3',
    'HORDEOLO E CALAZIO': 'H00',
    'CORPO ESTRANHO NA CORNEA': 'T15.0',
    # Cap VIII — Ouvido (H60-H95)
    'OTITE MEDIA NAO ESPECIFICADA': 'H66.9', 'OTITE MEDIA AGUDA NAO ESPECIFICADA': 'H66.9',
    'OTITE EXTERNA NAO ESPECIFICADA': 'H60.9',
    'OUTRAS OTITES EXTERNAS INFECCIOSAS': 'H60.3',
    'CERUME IMPACTADO': 'H61.2',
    'OTALGIA': 'H92.0',
    # Cap IX — Aparelho circulatório (I00-I99)
    'HIPERTENSAO ESSENCIAL (PRIMARIA)': 'I10',
    'DOENCA CARDIACA HIPERTENSIVA': 'I11',
    'ANGINA PECTORIS NAO ESPECIFICADA': 'I20.9', 'ANGINA PECTORIS': 'I20',
    'INFARTO AGUDO DO MIOCARDIO NAO ESPECIFICADO': 'I21.9',
    'INFARTO AGUDO DO MIOCARDIO': 'I21',
    'INSUFICIENCIA CARDIACA CONGESTIVA': 'I50.0',
    'INSUFICIENCIA CARDIACA NAO ESPECIFICADA': 'I50.9',
    'ACIDENTE VASCULAR CEREBRAL, NAO ESPECIFICADO COMO HEMORRAGICO OU ISQUEMICO': 'I64',
    'FIBRILACAO E FLUTTER ATRIAL': 'I48',
    'TAQUICARDIA SUPRAVENTRICULAR': 'I47.1',
    'TROMBOSE VENOSA PROFUNDA': 'I80.2',
    'VARIZES DOS MEMBROS INFERIORES': 'I83',
    'HEMORROIDAS NAO ESPECIFICADAS': 'I84.9', 'HEMORROIDAS': 'I84',
    'HIPOTENSAO NAO ESPECIFICADA': 'I95.9',
    'EMBOLIA PULMONAR': 'I26',
    # Cap X — Aparelho respiratório (J00-J99)
    'NASOFARINGITE AGUDA [RESFRIADO COMUM]': 'J00',
    'SINUSITE AGUDA NAO ESPECIFICADA': 'J01.9',
    'FARINGITE AGUDA NAO ESPECIFICADA': 'J02.9', 'FARINGITE AGUDA': 'J02',
    'AMIGDALITE AGUDA NAO ESPECIFICADA': 'J03.9',
    'AMIGDALITE AGUDA DEVIDA A OUTROS MICROORGANISMOS ESPECIFICADOS': 'J03.8',
    'AMIGDALITE AGUDA ESTREPTOCOCICA': 'J03.0',
    'LARINGITE AGUDA': 'J04.0',
    'LARINGITE E TRAQUEITE AGUDAS': 'J04',
    'INFECCAO AGUDA DAS VIAS AEREAS SUPERIORES NAO ESPECIFICADA': 'J06.9',
    'INFECCOES AGUDAS DAS VIAS AEREAS SUPERIORES DE LOCALIZACOES MULTIPLAS E NAO ESPECIFICADAS': 'J06',
    'INFLUENZA [GRIPE] DEVIDA A VIRUS NAO IDENTIFICADO': 'J11',
    'INFLUENZA [GRIPE] DEVIDA A VIRUS IDENTIFICADO DA GRIPE': 'J10',
    'PNEUMONIA VIRAL NAO ESPECIFICADA': 'J12.9', 'PNEUMONIA VIRAL': 'J12',
    'PNEUMONIA BACTERIANA NAO CLASSIFICADA EM OUTRA PARTE': 'J15',
    'PNEUMONIA POR MICROORGANISMO NAO ESPECIFICADA': 'J18.9',
    'PNEUMONIA NAO ESPECIFICADA': 'J18.9',
    'BRONQUITE AGUDA NAO ESPECIFICADA': 'J20.9',
    'BRONQUIOLITE AGUDA NAO ESPECIFICADA': 'J21.9',
    'ASMA NAO ESPECIFICADA': 'J45.9', 'ASMA': 'J45',
    'ESTADO DE MAL ASMATICO': 'J46',
    'BRONQUITE NAO ESPECIFICADA COMO AGUDA OU CRONICA': 'J40',
    'DOENCA PULMONAR OBSTRUTIVA CRONICA NAO ESPECIFICADA': 'J44.9',
    'DOENCA PULMONAR OBSTRUTIVA CRONICA': 'J44',
    'OUTRAS DOENCAS DO TRATO RESPIRATORIO SUPERIOR': 'J39',
    'TOSSE': 'R05',
    # Cap XI — Aparelho digestivo (K00-K93)
    'CARIE DENTARIA': 'K02',
    'OUTRAS DOENCAS DOS TECIDOS MOLES DA BOCA': 'K13',
    'DOENCA DE REFLUXO GASTROESOFAGICO COM ESOFAGITE': 'K21.0',
    'DOENCA DO REFLUXO GASTROESOFAGICO': 'K21',
    'ULCERA GASTRICA': 'K25', 'ULCERA DUODENAL': 'K26',
    'GASTRITE NAO ESPECIFICADA': 'K29.7', 'GASTRITE': 'K29',
    'OUTRAS GASTRITES AGUDAS': 'K29.1',
    'DISPEPSIA': 'K30',
    'GASTROENTERITE E COLITE NAO-INFECCIOSAS, NAO ESPECIFICADAS': 'K52.9',
    'SINDROME DO COLON IRRITAVEL': 'K58',
    'CONSTIPACAO': 'K59.0',
    'APENDICITE AGUDA NAO ESPECIFICADA': 'K35.9', 'APENDICITE AGUDA': 'K35',
    'HERNIA INGUINAL': 'K40',
    'DOENCA DIVERTICULAR DO INTESTINO GROSSO SEM PERFURACAO OU ABSCESSO': 'K57.3',
    'COLELITIASE NAO ESPECIFICADA': 'K80.2', 'COLELITIASE': 'K80',
    'COLECISTITE NAO ESPECIFICADA': 'K81.9', 'COLECISTITE AGUDA': 'K81.0',
    'PANCREATITE AGUDA': 'K85',
    'OUTRAS DOENCAS DO ANUS E DO RETO': 'K62',
    'DOENCAS DO FIGADO': 'K76',
    'HEMORRAGIA GASTROINTESTINAL NAO ESPECIFICADA': 'K92.2',
    # Cap XII — Pele (L00-L99)
    'ABSCESSO CUTANEO, FURUNCULO E CARBUNCULO': 'L02',
    'ABSCESSO CUTANEO, FURUNCULO E CARBUNCULO NAO ESPECIFICADOS': 'L02.9',
    'CELULITE NAO ESPECIFICADA': 'L03.9', 'CELULITE': 'L03',
    'IMPETIGO': 'L01',
    'DERMATITE ATOPICA NAO ESPECIFICADA': 'L20.9',
    'DERMATITE DE CONTATO NAO ESPECIFICADA': 'L25.9',
    'URTICARIA NAO ESPECIFICADA': 'L50.9', 'URTICARIA': 'L50',
    'PIODERMITE': 'L08.0',
    # Cap XIII — Sistema osteomuscular (M00-M99)
    'DORSALGIA NAO ESPECIFICADA': 'M54.9', 'DORSALGIA': 'M54',
    'CERVICALGIA': 'M54.2',
    'LUMBAGO COM CIATICA': 'M54.4',
    'DOR LOMBAR BAIXA': 'M54.5', 'LUMBAGO NAO ESPECIFICADO': 'M54.5',
    'MIALGIA': 'M79.1',
    'DOR EM MEMBRO': 'M79.6',
    'ARTRALGIA': 'M25.5',
    'CONTRATURA DE MUSCULO': 'M62.4',
    'GOTA NAO ESPECIFICADA': 'M10.9', 'GOTA': 'M10',
    'ARTRITE REUMATOIDE NAO ESPECIFICADA': 'M06.9',
    'TRANSTORNOS DE DISCOS LOMBARES E DE OUTROS DISCOS INTERVERTEBRAIS COM RADICULOPATIA': 'M51.1',
    'SINOVITE E TENOSSINOVITE NAO ESPECIFICADAS': 'M65.9',
    'EPICONDILITE LATERAL': 'M77.1',
    # Cap XIV — Aparelho geniturinário (N00-N99)
    'INFECCAO DO TRATO URINARIO DE LOCALIZACAO NAO ESPECIFICADA': 'N39.0',
    'CISTITE NAO ESPECIFICADA': 'N30.9', 'CISTITE AGUDA': 'N30.0',
    'CALCULO RENAL': 'N20.0', 'CALCULO DO RIM E DO URETER': 'N20',
    'COLICA RENAL NAO ESPECIFICADA': 'N23',
    'HIPERPLASIA DA PROSTATA': 'N40',
    'MENSTRUACAO EXCESSIVA E FREQUENTE COM CICLO REGULAR': 'N92.0',
    'HEMORRAGIA VAGINAL E UTERINA ANORMAL, NAO ESPECIFICADA': 'N93.9',
    'DOENCA INFLAMATORIA DO UTERO NAO ESPECIFICADA': 'N71.9',
    'VAGINITE AGUDA': 'N76.0',
    # Cap XV — Gravidez (O00-O99)
    'ABORTO ESPONTANEO': 'O03',
    'TRABALHO DE PARTO PREMATURO': 'O60',
    'ECLAMPSIA': 'O15',
    'HIPERTENSAO GESTACIONAL': 'O13',
    # Cap XVIII — Sintomas e sinais (R00-R99)
    'DOR ABDOMINAL E PELVICA': 'R10',
    'DOR ABDOMINAL NAO ESPECIFICADA': 'R10.4',
    'DOR LOCALIZADA NO ABDOME SUPERIOR': 'R10.1',
    'DOR LOCALIZADA EM OUTRAS PARTES DO ABDOME INFERIOR': 'R10.3',
    'DOR PELVICA E PERINEAL': 'R10.2',
    'NAUSEA E VOMITOS': 'R11', 'NAUSEA': 'R11',
    'PIROSE': 'R12',
    'CEFALEIA': 'R51', 'DOR DE CABECA': 'R51',
    'FEBRE NAO ESPECIFICADA': 'R50.9', 'FEBRE': 'R50',
    'SINCOPE E COLAPSO': 'R55',
    'CONVULSOES NAO CLASSIFICADAS EM OUTRA PARTE': 'R56',
    'DOR NAO CLASSIFICADA EM OUTRA PARTE': 'R52',
    'DOR AGUDA': 'R52.0',
    'DOR TORACICA NAO ESPECIFICADA': 'R07.4',
    'DOR DE GARGANTA E NO PEITO': 'R07',
    'DISPNEIA': 'R06.0', 'FALTA DE AR': 'R06.0',
    'EPISTAXE': 'R04.0',
    'HEMOPTISE': 'R04.2',
    'DIFICULDADE DE DEGLUTICAO': 'R13',
    'ERUPCAO CUTANEA E OUTRAS NAO ESPECIFICADAS': 'R21',
    'EDEMA LOCALIZADO': 'R60.0',
    'TONTURA E INSTABILIDADE': 'R42', 'VERTIGEM': 'R42',
    'MAL ESTAR E FADIGA': 'R53', 'MAL-ESTAR': 'R53',
    'CAUSAS DESCONHECIDAS E NAO ESPECIFICADAS DE MORBIDADE': 'R69',
    'HEMORRAGIA NAO CLASSIFICADA EM OUTRA PARTE': 'R58',
    'RETENCAO URINARIA': 'R33',
    'ANOREXIA': 'R63.0',
    'ACHADOS ANORMAIS DE EXAMES DE SANGUE': 'R79',
    # Cap XIX — Lesões e envenenamentos (S00-T98)
    'TRAUMATISMO SUPERFICIAL DA CABECA NAO ESPECIFICADO': 'S00.9',
    'TRAUMATISMO NAO ESPECIFICADO DA CABECA': 'S09.9',
    'FERIMENTO DA CABECA': 'S01',
    'FRATURA DO RADIO DISTAL': 'S52.5',
    'FRATURA DA PERNA, INCLUINDO TORNOZELO': 'S82',
    'FRATURA DO FEMUR': 'S72',
    'ENTORSE E DISTENSAO DO TORNOZELO': 'S93.4',
    'ENTORSE E DISTENSAO DO JOELHO': 'S83',
    'CONTUSAO DO JOELHO': 'S80.0',
    'CONTUSAO DE OUTRAS PARTES DO PUNHO E DA MAO': 'S60.2',
    'CONTUSAO DE DEDO(S) DO PE SEM LESAO DA UNHA': 'S90.1',
    'FERIMENTO DO DEDO(S) DA MAO SEM LESAO DA UNHA': 'S61.0',
    'FERIMENTO DE OUTRAS PARTES DO ANTEBRACO': 'S51.8',
    'FERIMENTO DO PENIS': 'S31.2',
    'LUXACAO DO OMBRO': 'S43.0',
    'CORPO ESTRANHO NO OUVIDO': 'T16',
    'CORPO ESTRANHO NO TRATO RESPIRATORIO': 'T17',
    'EFEITOS DO CALOR E DA LUZ': 'T67',
    'QUEIMADURA NAO ESPECIFICADA': 'T30',
    'INTOXICACAO POR OUTRAS DROGAS, MEDICAMENTOS E SUBSTANCIAS BIOLOGICAS E AS NAO ESPECIFICADAS': 'T50.9',
    'EFEITO TOXICO DE ALCOOL': 'T51',
    'EFEITOS TOXICOS DE SUBSTANCIAS DE ORIGEM PREDOMINANTEMENTE NAO-MEDICINAL': 'T65',
    'MORDEDURA OU PICADA DE INSETO NAO-VENENOSO': 'W57',
    'CONTATO COM ANIMAIS VENENOSOS': 'T63',
    # Cap XX — Causas externas (V01-Y98)
    'QUEDA MESMO NIVEL POR ESCORR., TROP. OU PASSO FALSO - RESIDENCIA': 'W01.0',
    'MOTOCICLISTA TRAUM. EM COL. C/CARRO, PICK-UP OU CAMINHON. - CONDUTOR TRAUM. EM ACID. ñ-TRANSITO': 'V43.0',
    'AGRESSAO POR MEIO DE OBJETO CORTANTE OU PENETRANTE': 'X99',
    'AGRESSAO POR DISPARO DE ARMA DE FOGO': 'X95',
    # Cap XXI — Fatores de saúde (Z00-Z99)
    'EXAME MEDICO GERAL': 'Z00.0', 'EXAME GERAL E INVESTIGACAO': 'Z00',
    'PESSOA EM CONTATO COM SERVICOS DE SAUDE PARA INVESTIGACAO E EXAMES': 'Z00',
    'NECESSIDADE DE VACINACAO': 'Z23',
    'SUPERVISAO DE GRAVIDEZ NORMAL NAO ESPECIFICADA': 'Z34.9',
    # COVID e SRAG
    'COVID-19, VIRUS IDENTIFICADO': 'U07.1',
    'COVID-19, VIRUS NAO IDENTIFICADO': 'U07.2',
    'SINDROME RESPIRATORIA AGUDA GRAVE [SARS]': 'U04',
}

# ── Mapeamento por palavras-chave para capítulo (fallback) ───────────
# Quando não há código CID e a descrição não está no dicionário acima,
# tenta determinar o capítulo por palavras-chave na descrição.
CHAPTER_KEYWORDS = {
    'I - Doenças infecciosas e parasitárias': [
        'DENGUE', 'CHIKUNGUNYA', 'ZIKA', 'MALARIA', 'TUBERCULOSE',
        'HANSENIASE', 'LEPTOSPIROSE', 'HEPATITE', 'HIV', 'AIDS',
        'SIFILIS', 'GONOCOCIC', 'HERPES', 'VARICELA', 'SARAMPO',
        'RUBEOLA', 'CAXUMBA', 'MENINGITE', 'TETANO', 'COQUELUCHE',
        'DIARREIA', 'GASTROENTERITE DE ORIGEM INFECCIOSA', 'FEBRE TIFOIDE',
        'COLERA', 'SEPTICEMIA', 'ERISIPELA', 'ESQUISTOSSOMOSE',
        'LEISHMANIOSE', 'CHAGAS', 'TOXOPLASMOSE', 'RAIVA', 'PESTE',
        'FEBRE AMARELA', 'FEBRE MACULOSA', 'FEBRE HEMORRAGICA',
    ],
    'IX - Aparelho circulatório': [
        'HIPERTENSAO', 'INFARTO', 'ANGINA', 'INSUFICIENCIA CARDIACA',
        'ACIDENTE VASCULAR CEREBRAL', 'AVC', 'FIBRILACAO', 'FLUTTER',
        'TAQUICARDIA', 'ARRITMIA', 'EMBOLIA PULMONAR', 'TROMBOSE',
        'VARIZES', 'HEMORROIDAS', 'HIPOTENSAO', 'ENDOCARDITE',
        'MIOCARDITE', 'PERICARDITE', 'DOENCA CARDIACA',
    ],
    'X - Aparelho respiratório': [
        'PNEUMONIA', 'BRONQUITE', 'BRONQUIOLITE', 'ASMA', 'GRIPE',
        'INFLUENZA', 'SINUSITE', 'FARINGITE', 'AMIGDALITE', 'LARINGITE',
        'TRAQUEITE', 'NASOFARINGITE', 'RESFRIADO', 'INFECCAO.*VIAS AEREAS',
        'PULMONAR OBSTRUTIVA', 'RINITE', 'OTITE',
    ],
    'XI - Aparelho digestivo': [
        'GASTRITE', 'ULCERA GASTRICA', 'ULCERA DUODENAL', 'APENDICITE',
        'COLELITIASE', 'COLECISTITE', 'PANCREATITE', 'DISPEPSIA',
        'REFLUXO GASTRO', 'HERNIA INGUINAL', 'HERNIA UMBILICAL',
        'CONSTIPACAO', 'CARIE DENTARIA', 'DIVERTICULAR', 'HEMORRAGIA GASTROINTESTINAL',
    ],
    'XIII - Sistema osteomuscular': [
        'DORSALGIA', 'CERVICALGIA', 'LUMBAGO', 'LOMBALGIA', 'ARTRITE',
        'ARTROSE', 'GOTA', 'MIALGIA', 'TENDINITE', 'BURSITE',
        'CONTRATURA', 'EPICONDILITE', 'FIBROMIALGIA', 'ESPONDIL',
        'RADICULOPATIA', 'DOR LOMBAR',
    ],
    'XIX - Lesões e causas externas': [
        'TRAUMATISMO', 'FRATURA', 'LUXACAO', 'ENTORSE', 'CONTUSAO',
        'FERIMENTO', 'QUEIMADURA', 'CORPO ESTRANHO', 'INTOXICACAO',
        'ENVENENAMENTO', 'MORDEDURA', 'PICADA', 'QUEDA',
        'MOTOCICLISTA', 'PEDESTRE', 'CICLISTA', 'AGRESSAO',
        'LESAO', 'EFEITO TOXICO',
    ],
    'XIV - Aparelho geniturinário': [
        'INFECCAO.*URINARI', 'CISTITE', 'CALCULO RENAL', 'COLICA RENAL',
        'PROSTAT', 'VAGINITE', 'MENSTRUACAO', 'HEMORRAGIA VAGINAL',
    ],
    'XVIII - Sintomas e sinais': [
        'DOR ABDOMINAL', 'NAUSEA', 'VOMITO', 'FEBRE NAO ESPECIFICADA',
        'CEFALEIA', 'SINCOPE', 'CONVULS', 'DOR TORACICA',
        'DISPNEIA', 'EPISTAXE', 'ERUPCAO CUTANEA', 'TONTURA',
        'MAL ESTAR', 'VERTIGEM', 'DOR AGUDA', 'DOR NAO CLASSIF',
        'RETENCAO URINARIA', 'HEMOPTISE',
    ],
    'IV - Endócrinas, nutricionais e metabólicas': [
        'DIABETES', 'HIPOGLICEMIA', 'DESIDRATACAO', 'HIPOPOTASSEMIA',
        'OBESIDADE', 'DESNUTRICAO', 'TIREOIDE', 'HIPOTIROID', 'HIPERTIROID',
    ],
    'XII - Pele e tecido subcutâneo': [
        'ABSCESSO CUTANEO', 'FURUNCULO', 'CELULITE', 'IMPETIGO',
        'DERMATITE', 'URTICARIA', 'PIODERMITE', 'ECZEMA',
    ],
    'V - Transtornos mentais': [
        'TRANSTORNO.*MENTAL', 'DEPRESSIVO', 'ANSIOS', 'BIPOLAR',
        'ESQUIZOFRENIA', 'USO DE ALCOOL', 'USO DE DROGA', 'PANICO',
    ],
    'XXI - Fatores que influenciam o estado de saúde': [
        'EXAME MEDICO', 'EXAME GERAL', 'VACINACAO', 'SUPERVISAO DE GRAVIDEZ',
        'PESSOA EM CONTATO', 'ACOMPANHAMENTO',
    ],
}


def desc_to_cid_code(desc):
    """Mapeia descrição CID (DATASUS) para código CID-10.

    Usa: (1) dicionário exato, (2) busca parcial no dicionário.
    Retorna o código ou None se não encontrar.
    """
    if desc is None or not isinstance(desc, str) or desc.strip() == '':
        return None

    d = desc.strip().upper()

    # 1. Busca exata
    if d in CID_DESC_TO_CODE:
        return CID_DESC_TO_CODE[d]

    # 2. Busca parcial — a descrição pode conter texto extra
    for key, code in CID_DESC_TO_CODE.items():
        if key in d or d in key:
            return code

    return None


def desc_to_chapter(desc):
    """Determina capítulo CID a partir da descrição usando palavras-chave.

    Fallback para quando não há código CID nem match no dicionário.
    """
    import re
    if desc is None or not isinstance(desc, str) or desc.strip() == '':
        return None

    d = desc.strip().upper()

    for chapter, keywords in CHAPTER_KEYWORDS.items():
        for kw in keywords:
            if '.*' in kw:
                if re.search(kw, d):
                    return chapter
            elif kw in d:
                return chapter

    return None


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
    if cid_str is None or not isinstance(cid_str, str) or cid_str == '':
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
    if cid_str is None or not isinstance(cid_str, str) or cid_str == '':
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
    if desc is None or not isinstance(desc, str) or desc == '':
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

    # Extrair código CID — 3 estratégias em cascata:
    # 1. Coluna cid_codigo (se existir e tiver dados)
    # 2. extract_cid_code() — regex em descrições tipo "A90 - Dengue"
    # 3. desc_to_cid_code() — dicionário DATASUS descrição→código
    col_desc = None
    for key in ['cid_descricao', 'cid_desc']:
        if key in cols_lower:
            col_desc = cols_lower[key]
            break
    if col_desc is None:
        col_desc = col_cid  # fallback

    if 'cid_codigo' in df.columns:
        # Coluna cid_codigo existe no CSV → limpar valores vazios
        df['cid_codigo'] = df['cid_codigo'].astype(str).str.strip()
        df.loc[df['cid_codigo'].isin(['', 'nan', 'None', 'NaN']), 'cid_codigo'] = pd.NA
        # Estratégia 2: regex na descrição
        mask_no_code = df['cid_codigo'].isna()
        if mask_no_code.any() and col_desc in df.columns:
            df.loc[mask_no_code, 'cid_codigo'] = (
                df.loc[mask_no_code, col_desc].apply(extract_cid_code))
        # Estratégia 3: dicionário DATASUS
        mask_no_code = df['cid_codigo'].isna()
        if mask_no_code.any() and col_desc in df.columns:
            df.loc[mask_no_code, 'cid_codigo'] = (
                df.loc[mask_no_code, col_desc].apply(desc_to_cid_code))
        print(f"   Coluna cid_codigo: {df['cid_codigo'].notna().mean():.0%} com código")
    else:
        # Estratégia 2
        df['cid_codigo'] = df[col_cid].apply(extract_cid_code)
        # Estratégia 3
        mask_no_code = df['cid_codigo'].isna()
        if mask_no_code.any() and col_desc in df.columns:
            df.loc[mask_no_code, 'cid_codigo'] = (
                df.loc[mask_no_code, col_desc].apply(desc_to_cid_code))

    # Verificar cobertura
    cid_coverage = df['cid_codigo'].notna().mean()
    print(f"   Cobertura CID: {cid_coverage:.0%}")

    # Se cobertura ainda baixa, usar desc_to_chapter como fallback para capítulos
    use_desc_fallback = cid_coverage < 0.2
    if use_desc_fallback:
        print(f"   ⚠ Cobertura CID baixa ({cid_coverage:.0%}).")
        print(f"   → Usando desc_to_chapter() por palavras-chave para capítulos.")
        # Criar coluna auxiliar de capítulo por descrição
        if col_desc in df.columns:
            df['_chapter_from_desc'] = df[col_desc].apply(desc_to_chapter)
            chapter_coverage = df['_chapter_from_desc'].notna().mean()
            print(f"   → Capítulos por palavras-chave: {chapter_coverage:.0%} cobertos")

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

    if not use_desc_fallback:
        # Caminho normal: tem códigos CID → capítulos e SINAN
        if agravos in ('all', 'chapters') or (isinstance(agravos, str) and agravos.startswith('top')):
            agg_ch = aggregate_raw_data(df, col_date, 'cid_codigo', col_qty, group_by='chapter')
            results.update(agg_ch)

        if agravos in ('all', 'sinan'):
            agg_sinan = aggregate_raw_data(df, col_date, 'cid_codigo', col_qty, group_by='sinan')
            # Prefixar com "SINAN: " para identificação no dashboard
            agg_sinan = {f"SINAN: {k}": v for k, v in agg_sinan.items()}
            results.update(agg_sinan)

        if agravos == 'all' or (isinstance(agravos, str) and agravos.startswith('top')):
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

    else:
        # Fallback: poucos códigos CID → usar descrições + palavras-chave
        print(f"   Usando fallback por descrição...")

        # Capítulos via desc_to_chapter()
        if agravos in ('all', 'chapters') or (isinstance(agravos, str) and agravos.startswith('top')):
            if '_chapter_from_desc' in df.columns:
                df_ch = df[df['_chapter_from_desc'].notna()].copy()
                if len(df_ch) > 0:
                    for ch, gdf in df_ch.groupby('_chapter_from_desc'):
                        agg_c = gdf.groupby(['ano_epi', 'semana_epi'])[col_qty].sum().reset_index()
                        agg_c.columns = ['ano', 'se', 'casos']
                        agg_c = agg_c[agg_c['se'] <= MAX_SE]
                        if len(agg_c) > 0:
                            results[str(ch)] = agg_c
                    print(f"   → {len([k for k in results if k not in ['Todos os atendimentos']])} capítulos gerados")

        # SINAN via palavras-chave na descrição
        if agravos in ('all', 'sinan'):
            # Mapear descrições para códigos CID via dicionário, depois checar SINAN
            if col_desc in df.columns:
                df['_sinan_from_desc'] = df[col_desc].apply(
                    lambda d: cid_to_sinan(desc_to_cid_code(d)) if desc_to_cid_code(d) else 'Outros'
                )
                df_sinan = df[df['_sinan_from_desc'] != 'Outros']
                if len(df_sinan) > 0:
                    for sinan_name, gdf in df_sinan.groupby('_sinan_from_desc'):
                        agg_s = gdf.groupby(['ano_epi', 'semana_epi'])[col_qty].sum().reset_index()
                        agg_s.columns = ['ano', 'se', 'casos']
                        agg_s = agg_s[agg_s['se'] <= MAX_SE]
                        if len(agg_s) > 0:
                            results[f"SINAN: {sinan_name}"] = agg_s
                    sinan_count = len([k for k in results if k.startswith('SINAN:')])
                    print(f"   → {sinan_count} agravos SINAN detectados")

        # Top N descrições como CIDs individuais
        n = 30
        df['_desc_clean'] = df[col_desc].astype(str).str.strip()
        df = df[df['_desc_clean'] != '']
        df = df[df['_desc_clean'] != 'nan']

        top_descs = (df.groupby('_desc_clean')[col_qty].sum()
                     .sort_values(ascending=False)
                     .head(n))

        print(f"   Top {len(top_descs)} descrições por volume:")
        for desc_name in top_descs.index:
            if pd.isna(desc_name) or not desc_name:
                continue
            # Tentar obter código CID para nome mais limpo
            cid_code = desc_to_cid_code(str(desc_name))
            display_name = f"{cid_code} - {desc_name}" if cid_code else str(desc_name)
            df_desc = df[df['_desc_clean'] == desc_name].copy()
            agg = df_desc.groupby(['ano_epi', 'semana_epi'])[col_qty].sum().reset_index()
            agg.columns = ['ano', 'se', 'casos']
            agg = agg[agg['se'] <= MAX_SE]
            if len(agg) > 0:
                results[display_name] = agg
                print(f"     {display_name}: {int(agg['casos'].sum())} atendimentos")

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
