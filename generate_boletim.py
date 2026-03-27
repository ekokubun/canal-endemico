#!/usr/bin/env python3
"""
generate_boletim.py Ñ Gera boletim HTML da semana epidemiologica encerrada no sabado.

Uso:
    python generate_boletim.py dados_ids.csv           # SE encerrada no ultimo sabado
    python generate_boletim.py dados_ids.csv --se 12  # SE especifica

Gera: boletins/SE_XX_YYYY.html
"""

import json, os, sys, argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd


# ?? Semana epidemiologica (padrao MS/OMS: dom-sab, quarta define o ano) ?????
def epi_week(dt):
    if pd.isna(dt):
        return (None, None)
    dt = pd.Timestamp(dt)
    dow = dt.isoweekday() % 7          # dom=0 .. sab=6
    sun = dt - timedelta(days=dow)
    wed = sun + timedelta(days=3)
    ano = wed.year
    jan1 = pd.Timestamp(ano, 1, 1)
    j1d  = jan1.isoweekday() % 7
    fs   = jan1 - timedelta(days=j1d) if j1d <= 3 else jan1 + timedelta(days=7 - j1d)
    se   = (sun - fs).days // 7 + 1
    return (ano, max(1, se))


def last_saturday_se():
    """Retorna (ano, se) da SE que encerrou no ultimo sabado."""
    today = datetime.today()
    dow   = today.isoweekday() % 7     # dom=0
    # Sabado anterior (inclusive hoje se hoje eh sabado)
    days_to_sat = (dow - 6) % 7
    last_sat = today - timedelta(days=days_to_sat)
    return epi_week(last_sat)


# ?? Carregar dados ????????????????????????????????????????????????????????????
def load_data(csv_path, target_year, target_se):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, sep=';', encoding=enc)
            break
        except Exception:
            continue

    cols = {c.lower().strip(): c for c in df.columns}

    # Calcular SE epidemiologica
    if 'data' in cols:
        df['_dt'] = pd.to_datetime(df[cols['data']], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['_dt'])
        epi = df['_dt'].apply(epi_week)
        df['_ano'] = epi.apply(lambda x: x[0])
        df['_se']  = epi.apply(lambda x: x[1])
    elif 'ano_epi' in cols and 'semana_epi' in cols:
        df['_ano'] = df[cols['ano_epi']]
        df['_se']  = df[cols['semana_epi']]
    else:
        raise ValueError('CSV sem coluna data nem ano_epi/semana_epi')

    # Coluna quantidade
    col_qty = cols.get('quantidade', None)
    if col_qty is None:
        df['_qty'] = 1
        col_qty = '_qty'

    return df, col_qty


# ?? Carregar canais fixos ????????????????????????????????????????????????????
def load_channels(channel_json='channel_data.json'):
    if not os.path.exists(channel_json):
        return None
    with open(channel_json, 'r') as f:
        return json.load(f)


# ?? Zona de classificacao ????????????????????????????????????????????????????
ZONE_LABELS = {
    'sucesso':   ('Zona de Sucesso',   '#22c55e'),
    'seguranca': ('Zona de Seguranca', '#86efac'),
    'alerta':    ('Zona de Alerta',    '#facc15'),
    'epidemico': ('Zona Epidemica',    '#f97316'),
    'emergencia':('Zona de Emergencia','#ef4444'),
    'sem dados': ('Sem dados',         '#94a3b8'),
}

def classify(value, p25, p50, p75, p90):
    if value is None: return 'sem dados'
    if value <= p25:  return 'sucesso'
    if value <= p50:  return 'seguranca'
    if value <= p75:  return 'alerta'
    if value <= p90:  return 'epidemico'
    return 'emergencia'


# ?? Gerar boletim ????????????????????????????????????????????????????????????
def generate(csv_path, target_se=None):
    # Determinar SE alvo
    if target_se is None:
        ano_alvo, se_alvo = last_saturday_se()
    else:
        from datetime import date
        ano_alvo = date.today().year
        se_alvo  = target_se

    print(f'Gerando boletim SE {se_alvo}/{ano_alvo}...')

    # Carregar dados
    df, col_qty = load_data(csv_path, ano_alvo, se_alvo)
    channels    = load_channels()

    # Filtrar SE alvo
    df_se = df[(df['_ano'] == ano_alvo) & (df['_se'] == se_alvo)]
    total_se = int(df_se[col_qty].sum()) if len(df_se) > 0 else 0

    # Montar linhas da tabela
    rows = []
    if channels:
        for agravo, ch in channels['channels'].items():
            raw_se = ch.get('raw', [])
            se_list = ch.get('se_list', [])
            ch_thresholds = ch.get('channels', {})

            # Observado na SE alvo
            obs = 0
            for entry, se in zip(raw_se, se_list):
                if se == se_alvo:
                    obs = entry.get(f'c{ano_alvo}', 0)
                    break

            # Limiar da SE alvo
            year_ch = ch_thresholds.get(str(ano_alvo), [])
            thr = year_ch[se_alvo - 1] if year_ch and se_alvo <= len(year_ch) else None
            if thr:
                p25, p50, p75, p90 = thr[1], thr[2], thr[3], thr[4]
                zona = classify(obs, p25, p50, p75, p90)
                p50_val, p90_val = p50, p90
            else:
                zona, p50_val, p90_val = 'sem dados', '-', '-'

            rows.append({
                'agravo': agravo,
                'obs': obs,
                'p50': p50_val,
                'p90': p90_val,
                'zona': zona,
            })

    # Ordenar por zona (critico primeiro)
    zona_order = {'emergencia': 0, 'epidemico': 1, 'alerta': 2, 'seguranca': 3, 'sucesso': 4, 'sem dados': 5}
    rows.sort(key=lambda r: (zona_order.get(r['zona'], 9), -r['obs']))

    # Gerar HTML
    now_str = datetime.now().strftime('%d/%m/%Y %H:%M')
    html = generate_html(rows, se_alvo, ano_alvo, total_se, now_str)

    # Salvar
    Path('boletins').mkdir(exist_ok=True)
    out = f'boletins/SE_{se_alvo:02d}_{ano_alvo}.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Boletim salvo: {out}')
    return out


# ?? Template HTML ?????????????????????????????????????????????????????????????
def zona_badge(zona):
    label, color = ZONE_LABELS.get(zona, ('?', '#94a3b8'))
    return f'<span style="background:{color};color:#000;padding:2px 8px;border-radius:4px;font-size:0.85em;">{label}</span>'


def generate_html(rows, se, ano, total_se, now_str):
    table_rows = ''
    for r in rows:
        badge = zona_badge(r['zona'])
        table_rows += (
            f'<tr>'
            f'<td style="padding:6px 10px">{r["agravo"]}</td>'
            f'<td style="padding:6px 10px;text-align:center">{r["obs"]}</td>'
            f'<td style="padding:6px 10px;text-align:center">{r["p50"]}</td>'
            f'<td style="padding:6px 10px;text-align:center">{r["p90"]}</td>'
            f'<td style="padding:6px 10px">{badge}</td>'
            f'</tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Boletim Epidemiologico SE {se:02d}/{ano}</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:20px}}
  h1{{color:#60a5fa;margin-bottom:4px}}
  h2{{color:#94a3b8;font-size:1rem;font-weight:400;margin:0 0 20px}}
  .card{{background:#1e293b;border-radius:8px;padding:20px;margin-bottom:20px}}
  table{{width:100%;border-collapse:collapse}}
  th{{background:#334155;padding:8px 10px;text-align:left;color:#94a3b8;font-size:0.85em;text-transform:uppercase}}
  tr:nth-child(even){{background:#253347}}
  .stat{{display:inline-block;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px 24px;margin:8px}}
  .stat-val{{font-size:2em;font-weight:700;color:#60a5fa}}
  .stat-label{{color:#94a3b8;font-size:0.85em}}
  footer{{color:#475569;font-size:0.8em;margin-top:30px}}
  @media print{{body{{background:#fff;color:#000}}h1{{color:#1d4ed8}}th{{background:#e2e8f0}}}}
</style>
</head>
<body>
<h1>Boletim Epidemiologico</h1>
<h2>UPA Rio Claro/SP &mdash; Semana Epidemiologica {se:02d}/{ano}</h2>

<div class="card">
  <div class="stat">
    <div class="stat-val">{se:02d}/{ano}</div>
    <div class="stat-label">Semana Epidemiologica</div>
  </div>
  <div class="stat">
    <div class="stat-val">{total_se:,}</div>
    <div class="stat-label">Total de atendimentos na SE</div>
  </div>
  <div class="stat">
    <div class="stat-val">{len(rows)}</div>
    <div class="stat-label">Sindromes monitoradas</div>
  </div>
</div>

<div class="card">
  <table>
    <thead>
      <tr>
        <th>Agravo / Sindrome</th>
        <th>Observado</th>
        <th>P50 (canal)</th>
        <th>P90 (limiar)</th>
        <th>Classificacao</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</div>

<footer>
  Gerado automaticamente em {now_str} &bull;
  Canais endemicoS: modelo Gamma-Poisson hierarquico &bull;
  <a href="../index.html" style="color:#60a5fa">Ver dashboard interativo</a>
</footer>
</body>
</html>
"""


# ?? Main ??????????????????????????????????????????????????????????????????????
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera boletim da SE encerrada no sabado')
    parser.add_argument('input', help='CSV de entrada (dados_ids.csv)')
    parser.add_argument('--se', type=int, default=None,
                        help='Numero da SE (omitir = SE encerrada no ultimo sabado)')
    args = parser.parse_args()
    generate(args.input, target_se=args.se)
