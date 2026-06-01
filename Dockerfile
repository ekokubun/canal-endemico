# Imagem da carga do canal endêmico na VPS (Fase 4).
# Empacota TODAS as deps + Chromium (p/ o boletim PDF) de uma vez, para o cron
# parar de rodar `pip install` a cada execução e poder rodar com
# `--user $(id -u):$(id -g)` (arquivos gerados ficam epikinesis-owned, não root).
#
# Build (na VPS):  docker build -t canal-carga:latest .
# Rebuild quando mudar requirements-vps.txt ou este Dockerfile.
FROM python:3.12-slim

# chromium (Debian slim provê /usr/bin/chromium) p/ HTML->PDF do boletim;
# fonts-liberation p/ renderizar Arial/Segoe (a fonte usada no boletim).
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-vps.txt /tmp/requirements-vps.txt
RUN pip install --no-cache-dir -r /tmp/requirements-vps.txt

# HOME=/tmp: o cron roda com --user 1000:1000, que não existe no /etc/passwd do
# container; sem um HOME gravável, gdown (~/.cache) e o chromium (user-data) falham.
ENV CHROME_BIN=/usr/bin/chromium \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONUNBUFFERED=1

WORKDIR /app
