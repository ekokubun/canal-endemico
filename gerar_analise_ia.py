#!/usr/bin/env python3
"""
gerar_analise_ia.py — Rascunho de análise epidemiológica narrativa por IA (Claude).

Gera um RASCUNHO interpretativo da situação da SE (tendências, destaques, recomendações)
a partir do canal endêmico. NÃO publica nada: escreve um arquivo de rascunho para REVISÃO
HUMANA. A publicação é um passo deliberado e separado (promover o texto aprovado para
analises_ia/SE{n}_{ano}.md no repo, que o gerar_boletim_pdf.py então renderiza).

Dado de entrada é AGREGADO por agravo × SE (sem PII). A chave vem da env ANTHROPIC_API_KEY
(injetada via `docker run --env-file`). Modelo padrão: claude-opus-4-8 (1×/semana no cron).

Fallback NÃO-bloqueante: sem chave, sem o SDK, ou erro de API → imprime AVISO e sai 0.
Nunca derruba o cron nem o boletim.

Uso:
    python3 gerar_analise_ia.py --output /drafts/analise_SE.md
"""

import argparse
import json
import sys
from datetime import date

# Reaproveita a lógica de enriquecimento já validada do boletim (status/zona, obs YTD,
# var% vs esperado) — zero duplicação. Importar é seguro: main() lá está sob __main__.
from gerar_boletim_pdf import enriquecer, se_atual_calc, se_para_datas, ZONA_NOME

MODELO_PADRAO = "claude-opus-4-8"

SYSTEM = """Você é epidemiologista da Sala de Situação / CIEVS da Fundação Municipal de \
Saúde de Rio Claro/SP. Escreve a análise interpretativa de um boletim sindrômico semanal \
baseado em canais endêmicos (modelo Gamma-Poisson) sobre atendimentos por hipótese \
diagnóstica nas UPAs.

Regras invioláveis:
- Baseie-se EXCLUSIVAMENTE nos dados fornecidos. NUNCA invente números, agravos ou \
tendências que não estejam nos dados. Se algo não está nos dados, não afirme.
- Os números determinísticos (contagens, percentuais) já estão na tabela do boletim; seu \
papel é INTERPRETAR, não recontar todos os valores.
- Tom técnico, objetivo e sóbrio, próprio de vigilância epidemiológica. Sem alarmismo e \
sem minimização.
- Estrutura: 3 a 5 parágrafos curtos. (1) panorama geral da semana; (2) destaques em zona \
de alerta/epidêmico/emergência, se houver, com a leitura da tendência; (3) agravos \
controlados/estáveis; (4) recomendações de rotina proporcionais ao quadro.
- As zonas, da melhor para a pior: Sucesso, Seguranca, Alerta, Epidemico, Emergencia.
- Português do Brasil. Não use markdown de cabeçalho (#); texto corrido em parágrafos.
- Lembre que é uma estimativa indireta a partir de atendimentos, não contagem direta de casos."""


def montar_resumo(items, channels, se, ano):
    """Resumo AGREGADO (sem PII) por agravo para alimentar a IA."""
    ano_s = str(ano)
    linhas = []
    for it in items:
        ch = channels.get(it["name"]) or {}
        zonas = ch.get("classifications", {}).get(ano_s, [])[:it.get("seN", se)]
        ult4 = [ZONA_NOME.get(z, z) for z in zonas[-4:]]
        linhas.append({
            "agravo": it["nome_limpo"],
            "zona_atual": ZONA_NOME.get(it["status"], it["status"]),
            "atendimentos_ano": it.get("obs_ytd"),
            "variacao_pct_vs_esperado": (round(it["var_pct"], 1) if it.get("var_pct") is not None else None),
            "zonas_ultimas_4_SEs": ult4,
        })
    # Ordena: pior zona primeiro (items já vem ordenado assim do enriquecer)
    return linhas


def gerar(channel_data_path, boletim_data_path, output_path, se_arg, ano_arg, model):
    with open(channel_data_path, encoding="utf-8") as f:
        cd = json.load(f)
    with open(boletim_data_path, encoding="utf-8") as f:
        bd = json.load(f)
    channels = cd.get("channels", cd)

    _, se_completa, ano_c = se_atual_calc()
    se = se_arg or int((cd.get("metadata") or {}).get("se_atual") or se_completa)
    ano = ano_arg or int((cd.get("metadata") or {}).get("ano_atual") or ano_c)
    se = min(se, se_completa)  # boletim reporta a última SE COMPLETA
    inicio, fim = se_para_datas(se, ano)
    se_label = f"SE {se:02d}/{ano} ({inicio.strftime('%d/%m/%Y')} a {fim.strftime('%d/%m/%Y')})"

    items = enriquecer(bd, channels, se)
    resumo = montar_resumo(items, channels, se, ano)

    n_crit = sum(1 for r in resumo if r["zona_atual"] in ("Epidemico", "Emergencia"))
    n_alerta = sum(1 for r in resumo if r["zona_atual"] == "Alerta")
    user_text = (
        f"Semana epidemiológica do boletim: {se_label}.\n"
        f"Resumo: {n_crit} agravo(s) em zona Epidemico/Emergencia, {n_alerta} em Alerta, "
        f"{len(resumo)} agravos monitorados.\n\n"
        f"Dados agregados por agravo (JSON):\n{json.dumps(resumo, ensure_ascii=False, indent=1)}\n\n"
        "Escreva a análise interpretativa da semana seguindo as regras."
    )

    try:
        from anthropic import Anthropic
    except Exception as e:
        print(f"AVISO: SDK anthropic indisponível ({e}) — pulando análise IA")
        return 0

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("AVISO: ANTHROPIC_API_KEY ausente — pulando análise IA")
        return 0

    try:
        client = Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_text}],
        )
        texto = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    except Exception as e:
        print(f"AVISO: chamada à Claude API falhou ({e}) — boletim segue sem análise IA")
        return 0

    if not texto:
        print("AVISO: resposta da IA vazia — nada gerado")
        return 0

    cabecalho = (
        f"<!-- RASCUNHO gerado por IA ({model}) para {se_label} em {date.today().isoformat()}. "
        "REVISAR antes de publicar. Para publicar: copie o texto aprovado para "
        f"analises_ia/SE{se:02d}_{ano}.md no repositório. -->\n\n"
        f"# Análise Epidemiológica — {se_label} (RASCUNHO IA, revisar)\n\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cabecalho + texto + "\n")
    print(f"rascunho de análise IA: {output_path} ({len(texto)} chars)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Rascunho de análise epidemiológica por IA (Claude)")
    ap.add_argument("--channel-data", default="channel_data.json")
    ap.add_argument("--boletim-data", default="boletim_data.json")
    ap.add_argument("--output", default="analise_ia_rascunho.md")
    ap.add_argument("--se", type=int, default=None)
    ap.add_argument("--ano", type=int, default=None)
    ap.add_argument("--model", default=MODELO_PADRAO)
    args = ap.parse_args()
    sys.exit(gerar(args.channel_data, args.boletim_data, args.output,
                   args.se, args.ano, args.model))


if __name__ == "__main__":
    main()
