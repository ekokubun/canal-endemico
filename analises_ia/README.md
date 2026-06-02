# Análises IA aprovadas

Análises narrativas da semana **revisadas e aprovadas** pela Vigilância. O boletim PDF
(`gerar_boletim_pdf.py`) renderiza a seção "Análise Epidemiológica da Semana" **se e
somente se** existir aqui um arquivo para a SE do boletim:

```
analises_ia/SE{nn}_{ano}.md      ex.: analises_ia/SE21_2026.md
```

## Fluxo

1. **Rascunho (automático, segunda):** o cron na VPS roda `gerar_analise_ia.py` (Claude)
   e grava um rascunho **privado** em `/home/epikinesis/analises_ia_rascunhos/` na VPS —
   **não** vai pro git. Conteúdo gerado por IA, **não revisado**.
2. **Revisão (humana):** ler/editar o rascunho.
3. **Publicação (deliberada):** copiar o texto **aprovado** para `analises_ia/SE{nn}_{ano}.md`
   neste repositório e commitar. Na próxima geração do boletim, a seção aparece no PDF.

O arquivo aprovado é **texto corrido** (parágrafos separados por linha em branco). Linhas
de cabeçalho markdown (`#`) e comentários HTML são ignorados na renderização.
