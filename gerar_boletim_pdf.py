#!/usr/bin/env python3
"""
Gerador de Boletim Sindromico - Sala de Situacao/CIEVS - FMS Rio Claro/SP
Gera HTML formatado para impressao/PDF a partir de boletim_data.json + channel_data.json.

Uso:
    python3 gerar_boletim_pdf.py [--output-dir boletins/]
"""

import json, argparse
from datetime import datetime, timedelta, date
from pathlib import Path

ZONA_RANK  = {"sucesso":0,"seguranca":1,"alerta":2,"epidemico":3,"emergencia":4}
ZONA_NOME  = {"sucesso":"Sucesso","seguranca":"Seguranca","alerta":"Alerta","epidemico":"Epidemico","emergencia":"Emergencia"}
ZONA_COR   = {"sucesso":"#16a34a","seguranca":"#2563eb","alerta":"#d97706","epidemico":"#ea580c","emergencia":"#dc2626"}
ZONA_EMOJI = {"sucesso":"v","seguranca":"B","alerta":"!","epidemico":"O","emergencia":"X"}
MESES_ABREV = ["jan","fev","mar","abr","mai","jun","jul","ago","set","out","nov","dez"]
MESES_LONG  = ["janeiro","fevereiro","marco","abril","maio","junho","julho","agosto","setembro","outubro","novembro","dezembro"]

def se_para_datas(se, ano):
    jan4 = date(ano, 1, 4)
    jan4_isowd = jan4.isoweekday()
    se1 = jan4 - timedelta(days=(jan4_isowd % 7))
    inicio = se1 + timedelta(weeks=se - 1)
    return inicio, inicio + timedelta(days=6)

def se_atual_calc():
    hoje = date.today()
    ano = hoje.year
    jan4 = date(ano, 1, 4)
    se1 = jan4 - timedelta(days=(jan4.isoweekday() % 7))
    dias = (hoje - se1).days
    se_curso = dias // 7 + 1
    inicio_se = se1 + timedelta(weeks=se_curso - 1)
    fim_se = inicio_se + timedelta(days=6)
    se_completa = se_curso if hoje > fim_se else se_curso - 1
    return se_curso, se_completa, ano

def fmt_abrev(d):
    return f"{d.day:02d}{MESES_ABREV[d.month-1]}"

def fmt_longa(d):
    return f"{d.day} de {MESES_LONG[d.month-1]} de {d.year}"

def nome_arquivo(se, ano):
    inicio, fim = se_para_datas(se, ano)
    return f"Boletim_Sindromico_{ano}_SE{se:02d}_{fmt_abrev(inicio)}_{fmt_abrev(fim)}"

def pior_zona(zonas):
    if not zonas: return "sucesso"
    return max(zonas, key=lambda z: ZONA_RANK.get(z, 0))

def limpa_nome(nome):
    return nome.replace("SINAN: ","").replace("Sind. ","Sindrome ").strip()

def enriquecer(boletim_data, channels, se_n):
    result = []
    for item in boletim_data:
        ch = channels.get(item["name"])
        seN = min(item.get("se_2026", 0), se_n)
        zonas, obs, p50 = [], 0, None
        if ch:
            zonas = ch.get("classifications",{}).get("2026",[])[:seN]
            obs   = sum(r.get("c2026",0) for r in ch.get("raw",[]) if r.get("se",0) <= seN)
            chs25 = ch.get("channels",{}).get("2025",[])
            if chs25 and seN > 0:
                p50 = sum(row[2] for row in chs25[:seN] if len(row) > 2)
        status  = pior_zona(zonas[-2:]) if zonas else "sucesso"
        var_pct = ((obs - p50) / p50 * 100) if p50 and p50 > 0 else None
        spark   = "".join(f'<span style="color:{ZONA_COR.get(z,"#cbd5e1")}">&#9679;</span>' for z in zonas[-4:]) or '<span style="color:#cbd5e1">&#8212;</span>'
        result.append({**item,"status":status,"obs_ytd":obs,"p50_ytd":p50,
                       "var_pct":var_pct,"sparkline":spark,"seN":seN,
                       "nome_limpo":limpa_nome(item["name"])})
    result.sort(key=lambda x: -ZONA_RANK.get(x["status"],0))
    return result

def row_html(item):
    cor = ZONA_COR[item["status"]]
    var_str = (f'<span style="color:{"#dc2626" if item["var_pct"]>=10 else "#16a34a" if item["var_pct"]<=-10 else "#6b7280"};font-weight:700">'
               f'{"+" if item["var_pct"]>0 else ""}{item["var_pct"]:.1f}%</span>'
               if item["var_pct"] is not None else "-")
    obs_fmt = f"{item['obs_ytd']:,}".replace(",",".")
    return f"""<tr style="border-bottom:1px solid #e5e7eb">
  <td style="padding:7px 9px;font-size:12px;font-weight:600">{item['nome_limpo']}</td>
  <td style="padding:7px 9px"><span style="background:{cor}22;color:{cor};padding:3px 9px;border-radius:5px;font-size:11px;font-weight:700;border:1px solid {cor}44">{ZONA_NOME[item['status']]}</span></td>
  <td style="padding:7px 9px;text-align:right;font-family:monospace;font-size:12px">{obs_fmt}</td>
  <td style="padding:7px 9px;text-align:right;font-size:12px">{var_str}</td>
  <td style="padding:7px 9px;text-align:center;font-size:15px;letter-spacing:4px;white-space:nowrap">{item['sparkline']}</td>
  <td style="padding:7px 9px;font-size:11px;color:#6b7280">{item.get('acao','Monitoramento de rotina.')}</td>
</tr>"""

TBL_HEAD = """<table style="width:100%;border-collapse:collapse;margin-top:8px">
<thead><tr style="background:#f1f5f9;border-bottom:2px solid #cbd5e1">
  <th style="padding:7px 9px;text-align:left;font-size:11px;color:#475569;font-weight:700">Agravo / Sindrome</th>
  <th style="padding:7px 9px;text-align:left;font-size:11px;color:#475569;font-weight:700">Status</th>
  <th style="padding:7px 9px;text-align:right;font-size:11px;color:#475569;font-weight:700">Atend. 2026</th>
  <th style="padding:7px 9px;text-align:right;font-size:11px;color:#475569;font-weight:700">Var% esp.</th>
  <th style="padding:7px 9px;text-align:center;font-size:11px;color:#475569;font-weight:700">Ult. 4 SEs</th>
  <th style="padding:7px 9px;text-align:left;font-size:11px;color:#475569;font-weight:700">Recomendacao</th>
</tr></thead><tbody>"""

def secao(titulo, cor, items):
    if not items: return ""
    rows = "\n".join(row_html(i) for i in items)
    return f"""<div style="margin-bottom:18px;page-break-inside:avoid">
  <div style="font-size:13px;font-weight:700;color:{cor};padding:6px 10px;background:{cor}11;border-left:4px solid {cor};border-radius:0 4px 4px 0;margin-bottom:4px">
    {titulo} <span style="font-size:11px;opacity:0.7">({len(items)} agravo{'s' if len(items)>1 else ''})</span></div>
  {TBL_HEAD}{rows}</tbody></table></div>"""

def gerar_html(items, se, ano):
    inicio, fim = se_para_datas(se, ano)
    # Referência = data de FIM da SE (determinística), não date.today(): evita que o
    # PDF mude todo dia só pela data de geração (churn de commits no git). O boletim
    # reporta uma SE completa, então a data de encerramento é a referência correta.
    ref = fim
    emerg = [i for i in items if i["status"]=="emergencia"]
    epid  = [i for i in items if i["status"]=="epidemico"]
    alert = [i for i in items if i["status"]=="alerta"]
    ok    = [i for i in items if i["status"] in ("sucesso","seguranca")]
    n_crit = len(emerg)+len(epid)
    nomes_e = ", ".join(i["nome_limpo"] for i in emerg+epid) or "-"
    nomes_a = ", ".join(i["nome_limpo"] for i in alert) or "-"
    return f"""<!DOCTYPE html><html lang="pt-BR"><head>
<meta charset="UTF-8">
<title>Boletim Sindromico {ano} SE{se:02d} - FMS Rio Claro</title>
<style>
@page{{size:A4;margin:16mm 14mm}}
@media print{{.np{{display:none!important}}body{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
body{{font-family:"Segoe UI",Arial,sans-serif;color:#1f2937;font-size:13px;line-height:1.5;margin:0;background:#fff}}
h1{{margin:0;font-size:20px;color:#1e3a5f}}
h2{{font-size:12px;font-weight:700;color:#374151;margin:14px 0 5px;padding-bottom:4px;border-bottom:2px solid #e5e7eb;text-transform:uppercase;letter-spacing:.5px}}
.hdr{{border-bottom:3px solid #1e3a5f;padding-bottom:12px;margin-bottom:14px}}
.sem{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:18px}}
.sc{{border-radius:8px;padding:12px;text-align:center}}
.ftr{{margin-top:14px;padding-top:8px;border-top:1px solid #e5e7eb;display:flex;justify-content:space-between;font-size:9.5px;color:#9ca3af}}
.nota{{margin-top:14px;padding:9px 12px;background:#f8fafc;border-left:3px solid #94a3b8;font-size:10px;color:#6b7280;line-height:1.6}}
</style></head><body>
<div class="np" style="background:#1e3a5f;color:#fff;padding:10px 20px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:99">
  <span style="font-weight:600;font-size:13px">Boletim Sindromico SE {se:02d}/{ano} - {inicio.strftime('%d/%m/%Y')} a {fim.strftime('%d/%m/%Y')} - FMS Rio Claro/SP</span>
  <button onclick="window.print()" style="background:#6366f1;color:#fff;border:none;padding:7px 18px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:700">Imprimir / Salvar PDF</button>
</div>
<div style="max-width:740px;margin:0 auto;padding:20px 16px">
<div class="hdr"><div style="display:flex;justify-content:space-between;align-items:flex-start">
  <div><div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">Sala de Situacao / CIEVS - Vigilancia Epidemiologica - FMS Rio Claro/SP</div>
  <h1>Boletim Sindromico de Atendimentos nas UPAs</h1>
  <div style="font-size:12px;color:#6b7280;margin-top:4px">SE {se:02d}/{ano} - {inicio.strftime('%d/%m/%Y')} a {fim.strftime('%d/%m/%Y')} - Hipoteses diagnosticas nas UPAs da FMS</div></div>
  <div style="text-align:right;font-size:10px;color:#9ca3af">N {se:02d}/{ano}<br>Encerramento: {ref.strftime('%d/%m/%Y')}<br><span style="font-size:9px">ekokubun.github.io/canal-endemico</span></div>
</div></div>
<h2>Quadro de Situacao - SE {se:02d}/{ano}</h2>
<div class="sem">
  <div class="sc" style="background:#fef2f2;border:2px solid #ef4444"><div style="font-size:36px;font-weight:700;color:#dc2626;line-height:1">{n_crit}</div><div style="font-size:11px;font-weight:700;color:#dc2626;text-transform:uppercase;margin:3px 0 5px">Emergencia / Epidemico</div><div style="font-size:9.5px;color:#6b7280">{nomes_e}</div></div>
  <div class="sc" style="background:#fffbeb;border:2px solid #f59e0b"><div style="font-size:36px;font-weight:700;color:#d97706;line-height:1">{len(alert)}</div><div style="font-size:11px;font-weight:700;color:#d97706;text-transform:uppercase;margin:3px 0 5px">Alerta</div><div style="font-size:9.5px;color:#6b7280">{nomes_a}</div></div>
  <div class="sc" style="background:#f0fdf4;border:2px solid #22c55e"><div style="font-size:36px;font-weight:700;color:#16a34a;line-height:1">{len(ok)}</div><div style="font-size:11px;font-weight:700;color:#16a34a;text-transform:uppercase;margin:3px 0 5px">Controlado</div><div style="font-size:9.5px;color:#6b7280">Dentro do esperado historico</div></div>
</div>
{secao("EMERGENCIA - Acao imediata necessaria","#dc2626",emerg)}
{secao("EPIDEMICO - Mobilizacao reforcada","#ea580c",epid)}
{secao("ALERTA - Atencao aumentada","#d97706",alert)}
{secao("CONTROLADO - Monitoramento de rotina","#16a34a",ok)}
<div class="nota"><b>Fonte e metodo:</b> Atendimentos por hipotese diagnostica nas UPAs da FMS de Rio Claro/SP (Sistema Maestro/IDS Saude), que estimam indiretamente o numero de casos ocorridos na populacao. As ocorrencias semanais sao classificadas segundo a probabilidade esperada de ocorrencia para cada semana epidemiologica, com base no padrao historico de atendimentos 2023-2025. Status = pior zona das ultimas 2 SEs completas. Var% = atendimentos 2026 acumulados vs frequencia esperada historica nas mesmas SEs de 2025.<br><b>Ult. 4 SEs:</b> uma bolinha por semana, cor = zona &#8212; <span style="color:#16a34a">&#9679;</span>&nbsp;Sucesso&nbsp;&nbsp;<span style="color:#2563eb">&#9679;</span>&nbsp;Seguranca&nbsp;&nbsp;<span style="color:#d97706">&#9679;</span>&nbsp;Alerta&nbsp;&nbsp;<span style="color:#ea580c">&#9679;</span>&nbsp;Epidemico&nbsp;&nbsp;<span style="color:#dc2626">&#9679;</span>&nbsp;Emergencia.</div>
<div class="ftr"><span>Sala de Situacao/CIEVS - FMS Rio Claro/SP - Vigilancia Epidemiologica</span><span>{fmt_longa(ref)}</span></div>
</div></body></html>"""

def atualizar_manifest(boletins_dir, se, ano, nome_pdf):
    mpath = boletins_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else []
    inicio, fim = se_para_datas(se, ano)
    # Preserva o gerado_em de uma entrada já existente (1ª publicação da SE), em vez de
    # reescrever com date.today() a cada run — senão o manifest muda todo dia (churn).
    prev = next((m for m in manifest if m["se"]==se and m["ano"]==ano), None)
    gerado_em = prev.get("gerado_em") if prev else date.today().isoformat()
    manifest = [m for m in manifest if not (m["se"]==se and m["ano"]==ano)]
    manifest.append({"se":se,"ano":ano,"data_inicio":inicio.isoformat(),"data_fim":fim.isoformat(),"arquivo":nome_pdf,"gerado_em":gerado_em})
    manifest.sort(key=lambda x:(x["ano"],x["se"]),reverse=True)
    mpath.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(f"  manifest.json: {len(manifest)} boletins")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel-data", default="channel_data.json")
    p.add_argument("--boletim-data",  default="boletim_data.json")
    p.add_argument("--output-dir",    default="boletins")
    p.add_argument("--se",  type=int, default=None)
    p.add_argument("--ano", type=int, default=None)
    args = p.parse_args()
    with open(args.channel_data) as f: cd = json.load(f)
    with open(args.boletim_data)  as f: bd = json.load(f)
    channels = cd.get("channels", cd)
    _, se_c, ano_c = se_atual_calc()
    se  = args.se  or int((cd.get("metadata") or {}).get("se_atual")  or se_c)
    ano = args.ano or int((cd.get("metadata") or {}).get("ano_atual") or ano_c)
    se = min(se, se_c)
    print(f"  SE {se}/{ano}")
    items = enriquecer(bd, channels, se)
    html = gerar_html(items, se, ano)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = nome_arquivo(se, ano)
    hp = out / f"{base}.html"
    hp.write_text(html, encoding="utf-8")
    print(f"  HTML: {hp}")
    atualizar_manifest(out, se, ano, f"{base}.pdf")
    print(hp)

if __name__ == "__main__":
    main()
