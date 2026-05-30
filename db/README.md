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

## Cron sugerido (VPS)

```cron
# Carga incremental diária às 06:30 BRT (após o consolidar do dia)
30 6 * * *  cd /opt/canal-endemico && git pull -q && \
            PGPASSWORD="$(cat /etc/epikinesis/pg.pass)" \
            python3 carga_postgres.py --recompute --input canal_endemico_input.csv \
            --pop 210000 --skip-channel-estimation >> /var/log/canal_carga.log 2>&1

# Recalibração completa em 1º de janeiro às 04:00
0 4 1 1 *   cd /opt/canal-endemico && \
            PGPASSWORD="$(cat /etc/epikinesis/pg.pass)" \
            python3 carga_postgres.py --recompute --input canal_endemico_input.csv \
            --pop 210000 --base-hist-years 2024,2025,2026 >> /var/log/canal_carga.log 2>&1
```

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
