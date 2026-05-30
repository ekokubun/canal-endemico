#!/usr/bin/env bash
# ============================================================================
# Carga incremental diária do canal endêmico (UPA 187) no Postgres fms_prod.
# Roda na VPS via cron. Baixa o CSV do Drive, atualiza o repo (código + estado
# congelado), e roda a carga INCREMENTAL num container Python na rede da stack.
#
# IMPORTANTE: usa SEMPRE --skip-channel-estimation. O CSV do Drive é parcial
# (só ano corrente + anterior); a história profunda vive no channel_state.json
# commitado. Um recompute completo aqui SOBRESCREVERIA o estado e perderia a
# história — por isso nunca rodamos completo na VPS.
#
# Pré-requisitos (uma vez — ver db/README.md):
#   - epikinesis no grupo docker          (docker sem sudo)
#   - git config credential.helper store  (PAT salvo p/ git pull)
#   - git update-index --skip-worktree channel_data.json
# ============================================================================
set -uo pipefail

REPO_DIR="/home/epikinesis/canal-endemico"
GDRIVE_ID="11sS3Ue4cUC-bGfBSVlDLdefbMNGv77Bf"
NET="fms-rc_fms-net"
PG_CONTAINER="fms_postgres"
PYIMG="python:3.12"
CSV="canal_endemico_input.csv"

cd "$REPO_DIR" || { echo "repo não encontrado em $REPO_DIR"; exit 1; }
echo "================ carga $(date -Is) ================"

# 1. CSV fresco do Drive
docker run --rm -v "$REPO_DIR":/app -w /app "$PYIMG" \
  bash -c "pip install -q gdown && gdown $GDRIVE_ID -O $CSV" || {
    echo "ERRO: download do CSV falhou"; exit 1; }

# Sanidade: CSV deve ter alguns MB (se vier <500KB, baixou página de erro)
SZ=$(stat -c%s "$CSV" 2>/dev/null || echo 0)
if [ "$SZ" -lt 500000 ]; then
  echo "ERRO: $CSV suspeito ($SZ bytes) — abortando"; exit 1
fi

# 2. Código + estado congelado atualizados (channel_data.json é skip-worktree).
#    Falha de pull não aborta a carga — segue com o que há local.
git pull --quiet || echo "AVISO: git pull falhou — usando código/estado local"

# 3. Credenciais do Postgres (da env do próprio container)
PGUSER_V=$(docker exec "$PG_CONTAINER" printenv POSTGRES_USER); PGUSER_V=${PGUSER_V:-postgres}
PGPW=$(docker exec "$PG_CONTAINER" printenv POSTGRES_PASSWORD)

# 4. Carga incremental → Postgres
docker run --rm --network "$NET" -v "$REPO_DIR":/app -w /app \
  -e PYTHONUNBUFFERED=1 -e PGHOST="$PG_CONTAINER" -e PGDATABASE=fms_prod \
  -e PGUSER="$PGUSER_V" -e PGPASSWORD="$PGPW" \
  "$PYIMG" bash -c "pip install -q -r requirements-vps.txt && \
    python carga_postgres.py --recompute --input $CSV --pop 210000 --skip-channel-estimation" || {
    echo "ERRO: carga falhou"; exit 1; }

echo "================ fim $(date -Is) ================"
