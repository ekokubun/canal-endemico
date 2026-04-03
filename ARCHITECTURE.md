# ARCHITECTURE.md — Canal Endêmico / Vigilância Sindrômica UPAs Rio Claro

**Versão:** 2.1  
**Data:** 2026-04-03  
**Repositório:** https://github.com/ekokubun/canal-endemico  
**Dashboard:** https://ekokubun.github.io/canal-endemico  
**Título público:** Vigilância Sindrômica nas UPAs — Canais Sindrômicos Bayesianos

---

## Visão Geral

Sistema de vigilância sindrômica automatizado para as UPAs de Rio Claro/SP.
Extrai hipóteses diagnósticas do sistema IDS Saúde (Maestro), computa canais
sindrômicos Gamma-Poisson e publica dashboard interativo via GitHub Pages.

> **Importante:** Os dados representam **hipóteses diagnósticas** registradas
> no atendimento das UPAs — não diagnósticos confirmados. Um paciente pode
> gerar múltiplos registros por atendimento (um por CID apresentado).

```
IDS Saúde (Maestro API)
    ↓  Google Apps Script — dailyAutomation() [3h BRT, diário]
    │      ├─ [1/3] dailyExtraction() → Maestro187_Exportados/ (Drive)
    │      ├─ [2/3] consolidarMaestro187Auto() → canal_endemico_input.csv (Drive)
    │      └─ [3/3] dispatchGitHubActions_() → GitHub Actions
    ↓
GitHub Actions — pipeline.py
    ↓
GitHub Pages — index.html (self-contained, ~2.3MB)
```

**Tempo total do ciclo:** ~2–3h → dashboard atualizado todos os dias antes das 7h BRT.

---

## Pipelines

### Pipeline 1 — Extração e Consolidação Diária

**Quem roda:** Google Apps Script (automático)  
**Quando:** Todo dia às 3h BRT (trigger time-based)  
**Arquivo GAS:** `Código.gs` + `1_GAS_consolidar_maestro187.gs`  
**Função principal:** `dailyAutomation()`

#### [1/3] Extração — `dailyExtraction()`

```
loginIDS() → JWT auth em rioclaro-saude2.ids.inf.br
    ↓
exportCSV(Relatório 187, token, data=ontem)
    ↓
parseFlatCSV() → pseudoId_() → decodeAge_() → epiWeek_()
    ↓
Salva dados do dia em: Maestro187_Exportados/{yyyy_mm_dd_yyyy_mm_dd_UPA_187.csv}
Local: Google Drive / Maestro187_Exportados/
```

- Extrai apenas **1 dia por execução** (ontem)
- Verifica duplicatas: não reextrai data já processada
- Pseudonimiza campo `Usuário` com SHA-256 + salt (LGPD)

#### [2/3] Consolidação — `consolidarMaestro187Auto()`

```
Lê todos os CSVs de Maestro187_Exportados/ (58 arquivos, ~1.78M linhas brutas)
    ↓
Agrega: ano_epi × semana_epi × faixa_etaria × cid_descricao → quantidade
    ↓
Remove duplicatas (chave: data+unidade+usuário+CID)
    ↓
Gera: canal_endemico_input.csv (~380k linhas, ~20MB)
Local: Google Drive / UPA_Dados/canal_endemico_input.csv
    ↓
Compartilha publicamente (link de leitura para GitHub Actions)
    ↓
dispatchGitHubActions_() → evento csv-updated
```

- Usa cursor de retomada (GAS tem limite de 6 min/execução)
- Cria trigger a cada 10 min para `consolidarMaestro187()` até concluir
- Ao concluir: chama `dispatchGitHubActions_()` automaticamente

#### [3/3] Dispatch — `dispatchGitHubActions_()`

```
POST https://api.github.com/repos/ekokubun/canal-endemico/dispatches
    event_type: csv-updated
    ↓
Inicia Pipeline 2
```

---

### Pipeline 2 — Geração do Dashboard

**Quem roda:** GitHub Actions (automático)  
**Quando:** Disparado por `csv-updated` (Pipeline 1) OU cron `0 6 * * *` (fallback 3h BRT)  
**Arquivo:** `.github/workflows/update-dashboard.yml`

```
Baixa canal_endemico_input.csv do Google Drive (via GDRIVE_FILE_ID secret)
    ↓
python3 pipeline.py dados_ids.csv --output index.html
    │
    ├─ Step 0: Detecção de formato → pseudonimização → dedup → agrega
    ├─ Step 1: compute_channels.py → 58 canais Gamma-Poisson → channel_data.json
    │           BASE_HIST_YEARS = [2023, 2024, 2025] (fixo, sem leave-one-out)
    ├─ Step 2: Agrega por faixa etária (20 agravos × 6 faixas)
    ├─ Step 3: Canais Gamma-Poisson por faixa etária
    ├─ Step 4: Boletim enriquecido (14 agravos prioritários, tendência, zonas)
    └─ Step 5: Injeta DATA + AGE_CHANNELS + BOLETIM_DATA → index.html
    ↓
git commit + push → index.html atualizado no repositório
    ↓
GitHub Pages publica automaticamente (~60s)
```

**Lógica de recálculo dos canais:**

```yaml
if mês == janeiro OU channel_data.json ausente:
    compute_channels.py --base-hist-years 2023,2024,2025
    # Recalibra limiares com histórico completo do ano anterior
else:
    compute_channels.py --base-hist-years 2023,2024,2025 --skip-channel-estimation
    # Mantém limiares existentes, atualiza apenas os dados observados
```

---

### Pipeline 3 — Boletim Epidemiológico Semanal

**Quem roda:** GitHub Actions (automático)  
**Quando:** Toda segunda-feira às 8h BRT (`cron: '0 11 * * 1'`)  
**Arquivo:** `.github/workflows/boletim-semanal.yml`

```
Calcula SE anterior completa (domingo–sábado da semana passada)
    ↓
python3 pipeline.py dados_ids.csv --boletim --se-num X --se-label "SE X/AAAA..."
    ↓
Gera boletim enriquecido no index.html com:
    - Classificação por zona (Sucesso/Segurança/Alerta/Epidêmico/Emergência)
    - Tendência das últimas 4 SEs por agravo
    - Comparação com ano anterior (mesma SE)
    - Recomendações operacionais para a FMS
    ↓
git commit + push
```

> **Status (2026-04-03):** Workflow criado e ativo. Argumento `--boletim`
> ainda não implementado no `pipeline.py` — a ser implementado.

---

### Pipeline 4 — Recálculo Anual dos Canais

**Quem roda:** GitHub Actions (automático, embutido no Pipeline 2)  
**Quando:** Janeiro de cada ano (detectado pelo mês no workflow)  

```
if date +%m == '01' OU channel_data.json ausente:
    compute_channels.py --base-hist-years ANO-3,ANO-2,ANO-1
    # Ex: em 2027 → --base-hist-years 2024,2025,2026
```

**Atualização manual anual (em janeiro):**
```python
# Em compute_channels.py
BASE_HIST_YEARS = [2024, 2025, 2026]  # remover 2023, adicionar 2026
```

---

## Arquivos e Localização

### Google Drive (ekokubun@gmail.com)

```
Google Drive
├── Maestro187_Exportados/              ← CSVs brutos mensais (HIST_FOLDER_ID)
│   ├── 2026_04_02_2026_04_01_UPA_187.csv
│   ├── 2026_03_31_2026_03_01_UPA_187.csv
│   ├── ...
│   └── 2021_01_31_2021_01_01_UPA_187.csv
│       (~58+ arquivos, jan/2021 → hoje)
│
└── UPA_Dados/                          ← saída da consolidação (DATA_FOLDER_ID)
    └── canal_endemico_input.csv        ← CSV agregado para o pipeline (GDRIVE_FILE_ID)
```

**Nomenclatura dos arquivos brutos:**
```
{último_dia}_{primeiro_dia}_UPA_187.csv
Exemplo: 2026_03_31_2026_03_01_UPA_187.csv
```

### Google Apps Script

**Projeto:** "Atendimentos UPA atalizacao diaria"  
**URL:** `script.google.com/home/projects/1vq-8DGm1dTuyUhDVII6AllgmTw1EE57vX_9iZT_Lbg51i4h6EXB0NqYM`

| Arquivo GAS | Função principal |
|---|---|
| `Código.gs` | `loginIDS()`, `exportCSV()`, `dailyExtraction()`, `dailyAutomation()` |
| `1_GAS_consolidar_maestro187.gs` | `consolidarMaestro187Auto()`, `consolidarMaestro187()` |
| `Extração historica.gs` | `iniciarExtracaoHistorica()`, `autoExtracaoHistorica_()` |
| `Extração maestro 187 Histo...gs` | Scripts de extração histórica auxiliares |

**Script Properties obrigatórias:**

| Propriedade | Descrição |
|---|---|
| `IDS_USERNAME` | Usuário do sistema IDS Saúde |
| `IDS_PASSWORD` | Senha do sistema IDS Saúde |
| `PSEUDO_SALT` | Salt SHA-256 para pseudonimização LGPD (padrão: `upa-rc-2026`) |
| `DATA_FOLDER_ID` | ID da pasta `UPA_Dados` no Drive |
| `HIST_FOLDER_ID` | ID da pasta `Maestro187_Exportados` no Drive |
| `GITHUB_PAT` | Personal Access Token GitHub (scope: `workflow`) |
| `GITHUB_OWNER` | `ekokubun` |
| `GITHUB_REPO` | `canal-endemico` |

**Triggers ativos:**

| Função | Tipo | Horário |
|---|---|---|
| `dailyAutomation` | Time-based | 3h BRT (diário) |
| `consolidarMaestro187` | Time-based | a cada 10 min (temporário, criado pelo Auto) |

### GitHub

**Repositório:** `https://github.com/ekokubun/canal-endemico`

| Arquivo | Onde | O que é |
|---|---|---|
| `pipeline.py` | raiz | Orquestrador Steps 0–5 |
| `compute_channels.py` | raiz | Modelo Gamma-Poisson hierárquico |
| `index.html` | raiz | Dashboard self-contained (~2.3MB) |
| `update-dashboard.yml` | `.github/workflows/` | Pipeline diário + fallback cron |
| `boletim-semanal.yml` | `.github/workflows/` | Boletim toda segunda-feira |

**Segredos do repositório (Settings → Secrets → Actions):**

| Secret | Descrição |
|---|---|
| `GDRIVE_FILE_ID` | ID do `canal_endemico_input.csv` no Drive |

---

## Modelo Estatístico

**Arquivo:** `compute_channels.py`

| Parâmetro | Valor |
|---|---|
| Modelo | Gamma-Poisson hierárquico bayesiano |
| Estimação | MoM → MLE (grid-search L-BFGS-B) → Monte Carlo (500k amostras) |
| Anos de treino (base fixa) | 2023, 2024, 2025 |
| Anos excluídos | 2021, 2022 (período de implantação — dados inconsistentes) |
| Ano monitorado | 2026 |
| Recalibração | Anual (janeiro), sem leave-one-out |
| Quantis | P10, P25, P50, P75, P90 |
| Zonas | Sucesso (≤P25) · Segurança (P25–P50) · Alerta (P50–P75) · Epidêmico (P75–P90) · Emergência (>P90) |
| Canais computados | 58 (21 capítulos CID + 10 CIDs individuais + 12 compostos + 15 SINAN) |

---

## Dashboard (GitHub Pages)

**URL:** `https://ekokubun.github.io/canal-endemico`  
**Arquivo:** `index.html` — self-contained, sem servidor

**Dados embutidos no HTML:**

| Variável JS | Conteúdo |
|---|---|
| `DATA` | 58 canais Gamma-Poisson + classificações + metadados |
| `AGE_GROUP_DATA` | 20 agravos × 6 faixas etárias |
| `AGE_CHANNELS` | Canais por faixa etária (formato compacto) |
| `BOLETIM_DATA` | 14 agravos prioritários com tendência e zonas |

**Metadados embutidos (`DATA.metadata`):**

| Campo | Descrição |
|---|---|
| `generated` | Timestamp ISO da geração |
| `se_atual` | Última SE com dados observados |
| `ano_atual` | Ano epidemiológico monitorado |
| `base_hist_years` | Anos usados na calibração dos canais |
| `source` | Arquivo CSV de entrada |

**Painéis do dashboard:**

1. Canal sindrômico principal (58 agravos)
2. Heatmap de classificação semanal por capítulo CID
3. Canais por faixa etária (20 agravos × 6 faixas)
4. Boletim epidemiológico semanal com análise orientada à FMS

---

## Fluxo Completo

```
══ 3h BRT — diário ══════════════════════════════════════════════════

GAS dailyAutomation()
  │
  ├─[1/3] dailyExtraction()
  │         loginIDS() → exportCSV(Relatório 187, data=ontem)
  │         → parse → pseudonimiza → dedup
  │         → Maestro187_Exportados/{data}_UPA_187.csv (Drive)
  │
  ├─[2/3] consolidarMaestro187Auto()
  │         Lê: Maestro187_Exportados/ (~58 CSVs, 1.78M linhas)
  │         Agrega: ano_epi×semana_epi×faixa×CID → quantidade
  │         → UPA_Dados/canal_endemico_input.csv (~380k linhas, ~20MB)
  │         Compartilha publicamente
  │         [trigger a cada 10 min até concluir — ~2h]
  │
  └─[3/3] dispatchGitHubActions_(event: csv-updated)

══ ~5–6h BRT — após conclusão do consolidar ══════════════════════

GitHub Actions update-dashboard.yml
  │
  ├─ Baixa canal_endemico_input.csv do Drive
  ├─ pipeline.py dados_ids.csv --output index.html
  │    Step 0: detecção → pseudonimiza → dedup → agrega
  │    Step 1: compute_channels.py → 58 canais Gamma-Poisson
  │    Step 2: agrega por faixa etária
  │    Step 3: canais por faixa
  │    Step 4: boletim enriquecido
  │    Step 5: injeta tudo → index.html
  └─ git commit + push
       ↓
  GitHub Pages (~60s)
  → https://ekokubun.github.io/canal-endemico ✅ ATUALIZADO

══ Segunda-feira 8h BRT — semanal ══════════════════════════════

GitHub Actions boletim-semanal.yml
  ├─ Calcula SE anterior completa
  ├─ pipeline.py --boletim --se-num X --se-label "SE X/AAAA..."
  └─ Boletim enriquecido com análise orientada à FMS
```

---

## Pseudonimização (LGPD Art. 13 §4°)

```javascript
// GAS — Código.gs
function pseudoId_(userId) {
  var raw = getProp('PSEUDO_SALT') + ':' + userId;
  var bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, raw);
  return bytes.map(function(b){ return ('0'+((b+256)%256).toString(16)).slice(-2); })
              .join('').substring(0, 12);
}
```

```python
# Python — pipeline.py Step 0
import hashlib
def _pseudo_id_s0(uid):
    key = 'upa-rc-2026' + ':' + str(uid).strip()
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]
```

- Resultado **idêntico** em GAS e Python — mesmo paciente → mesmo pseudo_id
- Dados brutos nunca sobem ao GitHub
- `canal_endemico_input.csv` compartilhado publicamente apenas para leitura

---

## Limites e Restrições

| Sistema | Limite | Impacto |
|---|---|---|
| GAS — tempo de execução | 6 min por execução | Cursor de retomada em todas as funções longas |
| GAS — cota diária | 90 min/dia | Consolidação distribuída em ~2h (trigger 10 min) |
| GAS — `everyMinutes()` | Apenas 1, 5, 10, 15 ou 30 | Usar 10 min como padrão |
| Drive — `setContent()` | ~50 MB | Agregar antes de salvar — nunca salvar bruto consolidado |
| GitHub Actions | 6h por execução | Pipeline atual ~20 min — dentro do limite |
| index.html | ~2.3 MB | Self-contained — sem servidor necessário |

---

## Manutenção

### Verificações periódicas (GAS)

```javascript
verificarConsolidacao()     // Ver estado do canal_endemico_input.csv
consolidarMaestro187Auto()  // Forçar regeneração do CSV (se desatualizado)
```

### Atualização manual do dashboard (GitHub)

```
GitHub → Actions → "Atualizar Dashboard Epidemiológico" → Run workflow
```

### Atualizar ano de monitoramento (virada de ano)

**Em `compute_channels.py`:**
```python
BASE_HIST_YEARS = [2024, 2025, 2026]  # remover 2023, adicionar 2026
EXCLUDED_YEARS  = [2021, 2022]        # manter excluídos
```

**No workflow `update-dashboard.yml`:**
```yaml
--base-hist-years 2024,2025,2026
```

### Adicionar novo relatório Maestro

Usar o skill `Extração_Maestro`:
1. Claude abre o Chrome e navega no sistema IDS
2. Captura o payload via DevTools (Network → POST maestrorelatorio)
3. Gera função GAS adicional plugada no `Código.gs` existente
4. Salva CSVs em `Maestro{ID}_Exportados/` no Drive

---

## Histórico de Versões

| Versão | Data | Mudança |
|---|---|---|
| 1.0 | 2025 | Pipeline inicial — CSV agregado GAS → GitHub Actions → Pages |
| 1.5 | 2025-Q4 | Canais por faixa etária e boletim |
| 2.0 | 2026-04-02 | Step 0 (formato bruto), extração histórica 2021–2026, consolidação com agregação |
| 2.1 | 2026-04-03 | `dailyAutomation` corrigido: [2/3] chama `consolidarMaestro187Auto()` diretamente; canais com base histórica fixa (sem leave-one-out); título público atualizado para "Vigilância Sindrômica nas UPAs"; `boletim-semanal.yml` criado; cabeçalho do boletim dinâmico; mapa de calor CID×SE adicionado |
