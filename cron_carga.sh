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
git checkout origin/main -- '*.py' Dockerfile db/ boletins/ analises_ia/ index.html requirements-vps.txt \
  cron_carga.sh channel_state.json age_state.json age_channels.json \
  age_group_data.json boletim_data.json 'boletim_SE*.docx' 2>/dev/null || echo "AVISO: checkout parcial falhou"

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
  # Zera metadados voláteis (CreationDate/ModDate/ID) → PDF determinístico entre runs
  # → o guard git diff só publica quando o conteúdo da SE muda (não todo dia).
  [ -f "$PDF" ] && python normaliza_pdf.py "$PDF" 2>/dev/null
  [ -f "$PDF" ] && echo "boletim PDF: $PDF" || echo "AVISO: PDF não gerado"
' || echo "AVISO: geração do boletim falhou"

# 5d. Boletim DOCX semanal — gerado NA VPS só às SEGUNDAS (antes era o GitHub Actions
#     boletim-semanal.yml). Usa o channel_data.json fresco do passo 4. Sai como
#     boletim_SE{n}.docx na raiz do repo. matplotlib+python-docx vêm na imagem.
GEN_DOCX=0
if [ "$(TZ=America/Sao_Paulo date +%u)" = "1" ]; then GEN_DOCX=1; fi
if [ "$GEN_DOCX" = "1" ]; then
  echo "segunda-feira — gerando boletim DOCX semanal"
  # SE anterior completa (último sábado; SE1=29/12/2025), calculada no container.
  SE_INFO=$(docker run --rm --user "$UG" -e HOME=/tmp -e TZ=America/Sao_Paulo "$IMG" \
    python3 -c 'import datetime;t=datetime.date.today();sa=t-datetime.timedelta(days=2);su=sa-datetime.timedelta(days=6);s1=datetime.date(2025,12,29);n=(su-s1).days//7+1;y=su.year if su>=s1 else su.year-1;m=["","janeiro","fevereiro","março","abril","maio","junho","julho","agosto","setembro","outubro","novembro","dezembro"];print(f"{n}|{y}|SE {n}/{y} — {su.day} de {m[su.month]} a {sa.day} de {m[sa.month]} de {sa.year}")' 2>/dev/null)
  IFS='|' read -r SE_NUM SE_YEAR SE_LABEL <<EOF
$SE_INFO
EOF
  if [ -n "$SE_NUM" ]; then
    echo "Boletim: $SE_LABEL"
    docker run --rm --user "$UG" -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl -e TZ=America/Sao_Paulo \
      -v "$REPO_DIR":/app -w /app "$IMG" \
      python pipeline.py --from-json channel_data.json --output index.html \
        --se-num "$SE_NUM" --se-year "$SE_YEAR" --se-label "$SE_LABEL" \
      && echo "boletim DOCX: boletim_SE${SE_NUM}.docx" || echo "AVISO: geração do DOCX falhou"
  else
    echo "AVISO: cálculo da SE para o DOCX falhou — pulando"
  fi
fi

# 5e. Análise por IA (Claude) — RASCUNHO PRIVADO, só às segundas e só se a chave existir.
#     Escreve em diretório privado da VPS (NÃO é commitado/empurrado). Revisão humana;
#     publicar = promover o texto aprovado p/ analises_ia/SE{n}_{ano}.md no repo (à mão).
#     Chave via --env-file (não aparece em docker inspect). Fallback: nunca derruba o cron.
SECRET="/home/epikinesis/.secrets/anthropic.env"
DRAFTS="/home/epikinesis/analises_ia_rascunhos"
if [ "$GEN_DOCX" = "1" ] && [ -f "$SECRET" ]; then
  echo "segunda-feira — gerando rascunho de análise por IA"
  mkdir -p "$DRAFTS"
  docker run --rm --user "$UG" -e HOME=/tmp --env-file "$SECRET" \
    -v "$REPO_DIR":/app -v "$DRAFTS":/drafts -w /app "$IMG" \
    python gerar_analise_ia.py --output "/drafts/analise_$(date +%Y%m%d).md" \
    || echo "AVISO: análise IA falhou (boletim segue sem ela)"
elif [ "$GEN_DOCX" = "1" ]; then
  echo "análise IA pulada (sem $SECRET)"
fi

# 5b. Publicar o boletim de volta no GitHub (mantém o GitHub Pages alimentado),
#     via clone EFÊMERO — não toca no worktree principal (que fica de propósito
#     com HEAD congelado). O CI não gera mais boletim; a VPS é a fonte dele.
PUSH_URL="$(git -C "$REPO_DIR" remote get-url origin)"   # contém o PAT salvo
TMP="$(mktemp -d)"
if git clone --depth 1 -q "$PUSH_URL" "$TMP" 2>/dev/null; then
  cp "$REPO_DIR"/boletins/*.pdf "$TMP/boletins/" 2>/dev/null
  cp "$REPO_DIR/boletins/manifest.json" "$TMP/boletins/" 2>/dev/null
  # DOCX semanal (passo 5d, só às segundas) — vai no MESMO commit/push do PDF.
  [ "$GEN_DOCX" = "1" ] && cp "$REPO_DIR"/boletim_SE*.docx "$TMP/" 2>/dev/null
  git -C "$TMP" add boletins/ boletim_SE*.docx 2>/dev/null
  if ! git -C "$TMP" diff --cached --quiet 2>/dev/null; then
    git -C "$TMP" -c user.name=vps-canal -c user.email=vps@epikinesis \
      commit -q -m "Boletins gerados na VPS [$(date +%d/%m/%Y)]" \
      && git -C "$TMP" push -q origin HEAD:main \
      && echo "boletins publicados no GitHub" || echo "AVISO: push do boletim falhou"
  else
    echo "boletins sem mudanças — nada a publicar"
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
