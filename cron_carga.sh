#!/usr/bin/env bash
# ============================================================================
# Carga incremental diária do canal endêmico (UPA 187) no Postgres fms_prod.
# Roda na VPS via cron. Baixa o CSV do Drive, atualiza o repo (código + estado
# congelado), roda a carga INCREMENTAL, regenera o dashboard estático, gera o
# boletim PDF e publica-o de volta no GitHub.
#
# Tudo roda numa imagem própria (canal-carga:latest) com as deps + Chromium
# pré-instalados, com --user $(id -u):$(id -g): sem `pip install` a cada run e
# os arquivos gerados ficam epikinesis-owned (não root).
#
# IMPORTANTE: usa SEMPRE --skip-channel-estimation. O CSV do Drive é parcial
# (só ano corrente + anterior); a história profunda vive no channel_state.json
# commitado. Um recompute completo aqui SOBRESCREVERIA o estado e perderia a
# história — por isso nunca rodamos completo na VPS.
#
# Pré-requisitos (uma vez — ver db/README.md):
#   - epikinesis no grupo docker          (docker sem sudo)
#   - git config credential.helper store  (PAT salvo p/ git fetch + push)
# ============================================================================
set -uo pipefail

REPO_DIR="/home/epikinesis/canal-endemico"
DASH_DIR="/home/epikinesis/canal-dashboard"   # estáticos servidos pelo nginx (fora do git)
GDRIVE_ID="11sS3Ue4cUC-bGfBSVlDLdefbMNGv77Bf"
NET="fms-rc_fms-net"
PG_CONTAINER="fms_postgres"
IMG="canal-carga:latest"
CSV="canal_endemico_input.csv"
UG="$(id -u):$(id -g)"                         # roda os containers como epikinesis

cd "$REPO_DIR" || { echo "repo não encontrado em $REPO_DIR"; exit 1; }
echo "================ carga $(date -Is) ================"

# 0. Garantir a imagem (constrói na 1ª vez ou se tiver sido removida).
#    Rebuild manual quando mudar requirements-vps.txt/Dockerfile (ver README).
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "imagem $IMG ausente — construindo..."
  docker build -t "$IMG" "$REPO_DIR" || { echo "ERRO: build da imagem falhou"; exit 1; }
fi

# 1. CSV fresco do Drive (gdown já vem na imagem)
docker run --rm --user "$UG" -e HOME=/tmp -v "$REPO_DIR":/app -w /app "$IMG" \
  gdown "$GDRIVE_ID" -O "$CSV" || { echo "ERRO: download do CSV falhou"; exit 1; }

# Sanidade: CSV deve ter alguns MB (se vier <500KB, baixou página de erro)
SZ=$(stat -c%s "$CSV" 2>/dev/null || echo 0)
if [ "$SZ" -lt 500000 ]; then
  echo "ERRO: $CSV suspeito ($SZ bytes) — abortando"; exit 1
fi

# 2. Atualizar código + estado do GitHub SEM merge completo. Faz checkout só dos
#    arquivos versionados (exceto channel_data.json, que a carga regenera) — evita
#    o conflito que travava o `git pull` e dispensa stash/sudo. O HEAD fica "atrás"
#    de propósito; `git status` ruidoso é normal. (Mantido mesmo após a imagem
#    --user resolver o root-owned: o conflito é de CONTEÚDO do channel_data.json
#    — VPS regenera vs CI commita — e persiste independente de ownership.)
git fetch origin -q || echo "AVISO: git fetch falhou — usando local"
git checkout origin/main -- '*.py' Dockerfile db/ boletins/ index.html requirements-vps.txt \
  cron_carga.sh channel_state.json age_state.json age_channels.json \
  age_group_data.json boletim_data.json 2>/dev/null || echo "AVISO: checkout parcial falhou"

# 3. Credenciais do Postgres (da env do próprio container)
PGUSER_V=$(docker exec "$PG_CONTAINER" printenv POSTGRES_USER); PGUSER_V=${PGUSER_V:-postgres}
PGPW=$(docker exec "$PG_CONTAINER" printenv POSTGRES_PASSWORD)

# 4. Carga incremental → Postgres (gera channel_data.json fresco em $REPO_DIR)
docker run --rm --user "$UG" --network "$NET" -v "$REPO_DIR":/app -w /app \
  -e HOME=/tmp -e PGHOST="$PG_CONTAINER" -e PGDATABASE=fms_prod \
  -e PGUSER="$PGUSER_V" -e PGPASSWORD="$PGPW" \
  "$IMG" python carga_postgres.py --recompute --input "$CSV" --pop 210000 \
    --skip-channel-estimation || { echo "ERRO: carga falhou"; exit 1; }

# 5. Dashboard estático: pipeline --no-recompute lê o channel_data.json fresco e
#    gera os 4 JSONs + index.html em $DASH_DIR (não toca no channel_state.json).
mkdir -p "$DASH_DIR"
docker run --rm --user "$UG" -e HOME=/tmp -v "$REPO_DIR":/app -v "$DASH_DIR":/dashboard -w /app \
  "$IMG" python pipeline.py "$CSV" --pop 210000 --output /dashboard/index.html \
    --template /app/index.html --no-recompute || {
    echo "AVISO: geração do dashboard falhou (Postgres já atualizado)"; }

# 5a. Boletim PDF — gerado NA VPS (antes era só no GitHub Actions). gerar_boletim_pdf.py
#     escreve o HTML em boletins/ + atualiza o manifest; o Chromium da imagem converte.
docker run --rm --user "$UG" -e HOME=/tmp -v "$REPO_DIR":/app -w /app "$IMG" bash -c '
  python gerar_boletim_pdf.py --channel-data channel_data.json \
    --boletim-data boletim_data.json --output-dir boletins/ || exit 1
  HTML=$(ls -t boletins/*.html 2>/dev/null | head -1); [ -z "$HTML" ] && exit 0
  PDF="${HTML%.html}.pdf"
  chromium --headless=new --no-sandbox --disable-gpu --disable-dev-shm-usage \
    --user-data-dir=/tmp/chrome --no-pdf-header-footer \
    --print-to-pdf="$PDF" "file://$(realpath "$HTML")" 2>/dev/null
  rm -f "$HTML"
  [ -f "$PDF" ] && echo "boletim PDF: $PDF" || echo "AVISO: PDF não gerado"
' || echo "AVISO: geração do boletim falhou"

# 5b. Publicar o boletim de volta no GitHub (mantém o GitHub Pages alimentado),
#     via clone EFÊMERO — não toca no worktree principal (que fica de propósito
#     com HEAD congelado). O CI não gera mais boletim; a VPS é a fonte dele.
PUSH_URL="$(git -C "$REPO_DIR" remote get-url origin)"   # contém o PAT salvo
TMP="$(mktemp -d)"
if git clone --depth 1 -q "$PUSH_URL" "$TMP" 2>/dev/null; then
  cp "$REPO_DIR"/boletins/*.pdf "$TMP/boletins/" 2>/dev/null
  cp "$REPO_DIR/boletins/manifest.json" "$TMP/boletins/" 2>/dev/null
  git -C "$TMP" add boletins/ 2>/dev/null
  if ! git -C "$TMP" diff --cached --quiet 2>/dev/null; then
    git -C "$TMP" -c user.name=vps-canal -c user.email=vps@epikinesis \
      commit -q -m "Boletim PDF gerado na VPS [$(date +%d/%m/%Y)]" \
      && git -C "$TMP" push -q origin HEAD:main \
      && echo "boletim publicado no GitHub" || echo "AVISO: push do boletim falhou"
  else
    echo "boletim sem mudanças — nada a publicar"
  fi
else
  echo "AVISO: clone efêmero p/ push do boletim falhou"
fi
rm -rf "$TMP"

# 5c. Acervo de boletins (PDFs + manifest) → dashboard servido em /canal/boletins/.
#     rm antes do cp p/ refletir remoções/renomeações e não aninhar (boletins/boletins).
rm -rf "$DASH_DIR/boletins"
cp -r "$REPO_DIR/boletins" "$DASH_DIR/" 2>/dev/null || echo "AVISO: cópia dos boletins falhou"
chmod -R a+rX "$DASH_DIR" 2>/dev/null   # nginx (usuário não-root) precisa ler os arquivos

echo "================ fim $(date -Is) ================"
