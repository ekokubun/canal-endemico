# Fase 4 — Canal Endêmico no Postgres (`fms_prod`)

O método Gamma-Poisson roda na VPS e grava o resultado no Postgres (schema
`rio_claro`); o **Metabase lê de `rio_claro.vw_canal_completo`**. Dado agregado,
sem PII. Fonte do piloto: UPA / relatório 187 (`fonte = 'upa_187'`).

```
canal_endemico_input.csv  (agregado, sem PII)
        │
        ▼
carga_postgres.py --recompute    ← compute_channels.run_pipeline + pipeline.step2/3
        │  (Gamma-Poisson, MLE + Monte Carlo)
        ▼
Postgres fms_prod / schema rio_claro
   dim_agravo · fato_observacao_se · canal_endemico · classificacao_se · execucao
        │
        ▼  vw_canal_completo
     Metabase
```

## Tabelas (ver `db/schema.sql`)

| Objeto | Papel |
|---|---|
| `dim_agravo` | catálogo (capítulo CID / CID / SINAN / síndrome / total) |
| `fato_observacao_se` | **fonte de verdade**: casos observados por agravo × faixa × ano × SE |
| `canal_endemico` | limiares do método (p10–p90, shape/rate) |
| `classificacao_se` | zona + exceedance computados pelo método |
| `execucao` | auditoria: 1 linha por carga |
| `vw_canal_completo` | **view do Metabase**: observado + faixas + zona (recalculada) |

`faixa_etaria = 'Todas'` é o agregado; as 6 faixas etárias vêm do step3.

## Implantação na VPS

```bash
# 1. Código (GitHub = fonte de verdade do código)
git clone https://github.com/ekokubun/canal-endemico.git
cd canal-endemico            # ou: git pull

# 2. Dependências
pip install -r requirements-vps.txt

# 3. Schema (idempotente — pode reaplicar)
psql -d fms_prod -f db/schema.sql

# 4. Credenciais (senha é secret na VPS — NÃO versionar)
export PGHOST=localhost PGDATABASE=fms_prod PGUSER=epikinesis
export PGPASSWORD='***'      # ler de um secret/arquivo protegido

# 5. Obter o CSV agregado (hoje no Drive via GDRIVE_FILE_ID) e carregar
python3 carga_postgres.py --recompute \
    --input canal_endemico_input.csv \
    --pop 210000 --schema rio_claro --fonte upa_187
```

### Modo incremental (rápido — congela limiares)
Reaproveita `channel_state.json` / `age_state.json`, recalcula só os observados:
```bash
python3 carga_postgres.py --recompute --input canal_endemico_input.csv \
    --pop 210000 --skip-channel-estimation
```
A recalibração completa dos canais é anual (janeiro) — rodar sem
`--skip-channel-estimation` (igual ao Pipeline 4 do `ARCHITECTURE.md`).

### Verificação sem banco (local ou VPS)
```bash
python3 carga_postgres.py --from-json --dry-run   # contagens + amostra, não conecta
```

## Cron na VPS (Docker)

Na VPS o Postgres roda no container `fms_postgres` (rede `fms-rc_fms-net`, 5432
**não** publicado no host) e o host **não tem psql/pip**. A carga roda num
container Python efêmero na rede da stack — encapsulado em `cron_carga.sh`
(baixa CSV do Drive → `git pull` → carga incremental → Postgres).

> ⚠️ **Sempre incremental (`--skip-channel-estimation`).** O `canal_endemico_input.csv`
> do Drive é parcial (ano corrente + anterior); a história profunda vive no
> `channel_state.json` commitado. Um recompute **completo** na VPS sobrescreveria
> o estado e perderia a história. A recalibração completa (janeiro) só com um CSV
> de história cheia — feita fora da VPS e trazida via `git pull`.

**Pré-requisitos (uma vez):**
```bash
sudo usermod -aG docker epikinesis          # docker sem sudo — RELOGAR depois
git config --global credential.helper store && git fetch  # salva o PAT (1x)
chmod +x cron_carga.sh
# Obs.: o cron usa `git fetch` + `git checkout origin/main -- <arquivos>` (não `git pull`),
# para não conflitar com o channel_data.json que a carga regenera (root-owned).
```

**Crontab (rode `crontab -e` como epikinesis):**
```cron
# Carga incremental diária — AJUSTE A HORA ao fuso da VPS (veja `date`/`timedatectl`).
# Ex.: VPS em UTC e alvo 06:30 BRT (UTC-3) → 30 9 * * *
30 9 * * *  /home/epikinesis/canal-endemico/cron_carga.sh >> /home/epikinesis/canal_carga.log 2>&1
```

Teste manual antes de agendar: `./cron_carga.sh` (deve terminar com
`✓ Postgres atualizado`). Logs em `~/canal_carga.log`.

## Metabase
1. Admin → Databases → adicionar/usar `fms_prod`.
2. Apontar os dashboards para **`rio_claro.vw_canal_completo`**.
3. Para o gráfico de canal: eixo X = `se`, filtros `agravo` / `faixa_etaria='Todas'`
   / `ano=2026`; séries `casos`, `p25`, `p50`, `p75`, `p90`; cor por `zona`.

> Se o Metabase conectar com usuário de leitura dedicado, conceder `USAGE`/`SELECT`
> no schema `rio_claro` (bloco comentado no fim de `db/schema.sql`).

## Notas
- A carga é **atômica por fonte**: dentro de uma transação, apaga as linhas de
  `fonte` e reinsere — o Metabase nunca vê estado parcial.
- **Fora de escopo (próximas etapas):** APS como 2ª fonte (`fonte='aps'`), captura
  direta substituindo o Drive, pseudonimização HMAC do 187 fino.
