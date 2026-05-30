-- ============================================================================
-- Canal Endêmico / Vigilância Sindrômica — schema Postgres (fms_prod)
-- Fase 4: o método Gamma-Poisson roda na VPS e grava aqui; Metabase lê daqui.
--
-- Cliente.....: Rio Claro/SP (FMS)            schema: rio_claro
-- Fonte piloto: UPA — relatório 187           coluna fonte = 'upa_187'
-- Granularidade: ano × semana epidemiológica × agravo × faixa etária
-- Dado AGREGADO, sem PII.
--
-- Idempotente: pode ser reaplicado (CREATE ... IF NOT EXISTS).
-- Aplicar:  psql -d fms_prod -f db/schema.sql
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS rio_claro AUTHORIZATION epikinesis;
SET search_path TO rio_claro;

-- ── Catálogo de agravos ─────────────────────────────────────────────────────
-- Um agravo é um capítulo CID, um CID individual, um agravo SINAN, uma síndrome
-- composta, ou o total ("Todos os atendimentos"). `nome` é a chave natural usada
-- pelo loader (corresponde à chave dos JSON / dicts em memória).
CREATE TABLE IF NOT EXISTS dim_agravo (
    agravo_id   SERIAL PRIMARY KEY,
    nome        TEXT NOT NULL UNIQUE,
    tipo        TEXT CHECK (tipo IN ('todos','capitulo','cid','sinan','sindrome')),
    cid_codigo  TEXT,          -- ex.: 'A90' (quando tipo='cid')
    capitulo    TEXT,          -- ex.: 'X - Aparelho respiratório'
    prioridade  TEXT           -- ALTA | MODERADA | BAIXA (do boletim), opcional
);

-- ── Fonte de verdade: observações semanais (o que de fato foi observado) ─────
CREATE TABLE IF NOT EXISTS fato_observacao_se (
    agravo_id     INTEGER NOT NULL REFERENCES dim_agravo(agravo_id),
    faixa_etaria  TEXT    NOT NULL DEFAULT 'Todas',  -- 'Todas' = agregado total
    fonte         TEXT    NOT NULL DEFAULT 'upa_187',
    ano           SMALLINT NOT NULL,
    se            SMALLINT NOT NULL CHECK (se BETWEEN 1 AND 53),
    casos         INTEGER  NOT NULL DEFAULT 0,
    PRIMARY KEY (agravo_id, faixa_etaria, fonte, ano, se)
);

-- ── Limiares do canal endêmico (saída do método Gamma-Poisson) ───────────────
-- Para o agregado, os limiares variam por ano monitorado (exposição/base).
-- Para faixas etárias, o método usa um único conjunto de limiares replicado
-- por ano (shape/rate ficam NULL nesse caso).
CREATE TABLE IF NOT EXISTS canal_endemico (
    agravo_id     INTEGER NOT NULL REFERENCES dim_agravo(agravo_id),
    faixa_etaria  TEXT    NOT NULL DEFAULT 'Todas',
    fonte         TEXT    NOT NULL DEFAULT 'upa_187',
    ano           SMALLINT NOT NULL,
    se            SMALLINT NOT NULL CHECK (se BETWEEN 1 AND 53),
    p10  INTEGER, p25 INTEGER, p50 INTEGER, p75 INTEGER, p90 INTEGER,
    shape DOUBLE PRECISION,   -- parâmetro Gamma (NULL para faixas etárias)
    rate  DOUBLE PRECISION,
    PRIMARY KEY (agravo_id, faixa_etaria, fonte, ano, se)
);

-- ── Classificação de zona + exceedance (saída do método) ─────────────────────
-- Mantida como verdade do método (auditoria). A view recalcula a zona a partir
-- dos limiares (determinístico) para garantir disponibilidade mesmo quando a
-- classificação do método não foi gravada (ex.: carga --from-json de faixas).
CREATE TABLE IF NOT EXISTS classificacao_se (
    agravo_id     INTEGER NOT NULL REFERENCES dim_agravo(agravo_id),
    faixa_etaria  TEXT    NOT NULL DEFAULT 'Todas',
    fonte         TEXT    NOT NULL DEFAULT 'upa_187',
    ano           SMALLINT NOT NULL,
    se            SMALLINT NOT NULL CHECK (se BETWEEN 1 AND 53),
    zona          TEXT CHECK (zona IN ('sucesso','seguranca','alerta','epidemico','emergencia')),
    exceedance    DOUBLE PRECISION,
    PRIMARY KEY (agravo_id, faixa_etaria, fonte, ano, se)
);

-- ── Auditoria: uma linha por execução da carga ───────────────────────────────
CREATE TABLE IF NOT EXISTS execucao (
    execucao_id      SERIAL PRIMARY KEY,
    gerado_em        TIMESTAMPTZ NOT NULL DEFAULT now(),
    fonte            TEXT NOT NULL DEFAULT 'upa_187',
    modelo           TEXT,
    mc_samples       INTEGER,
    base_hist_years  INTEGER[],
    se_atual         SMALLINT,
    fonte_csv        TEXT,           -- arquivo/origem do CSV recomputado
    n_agravos        INTEGER,
    modo             TEXT            -- 'recompute' | 'from-json'
);

-- ── Índices de apoio para o Metabase ─────────────────────────────────────────
CREATE INDEX IF NOT EXISTS ix_obs_ano_se   ON fato_observacao_se (ano, se);
CREATE INDEX IF NOT EXISTS ix_obs_faixa     ON fato_observacao_se (faixa_etaria);

-- ── View principal para o Metabase: canal completo, pronto para plotar ───────
-- Observado vs faixas p10–p90 + zona (recalculada) + zona do método + exceedance.
CREATE OR REPLACE VIEW vw_canal_completo AS
SELECT
    a.nome        AS agravo,
    a.tipo,
    a.cid_codigo,
    a.capitulo,
    a.prioridade,
    o.fonte,
    o.faixa_etaria,
    o.ano,
    o.se,
    o.casos,
    c.p10, c.p25, c.p50, c.p75, c.p90,
    c.shape, c.rate,
    cl.exceedance,
    cl.zona       AS zona_metodo,
    CASE
        WHEN c.p25 IS NULL    THEN NULL
        WHEN o.casos <= c.p25 THEN 'sucesso'
        WHEN o.casos <= c.p50 THEN 'seguranca'
        WHEN o.casos <= c.p75 THEN 'alerta'
        WHEN o.casos <= c.p90 THEN 'epidemico'
        ELSE 'emergencia'
    END           AS zona
FROM fato_observacao_se o
JOIN dim_agravo a        USING (agravo_id)
LEFT JOIN canal_endemico c   USING (agravo_id, faixa_etaria, fonte, ano, se)
LEFT JOIN classificacao_se cl USING (agravo_id, faixa_etaria, fonte, ano, se);

-- ── Permissões de leitura para o Metabase ────────────────────────────────────
-- O owner (epikinesis) já tem acesso total. Se o Metabase conecta com um usuário
-- de leitura dedicado, descomente e ajuste o nome do role:
--
--   GRANT USAGE ON SCHEMA rio_claro TO metabase_ro;
--   GRANT SELECT ON ALL TABLES IN SCHEMA rio_claro TO metabase_ro;
--   ALTER DEFAULT PRIVILEGES IN SCHEMA rio_claro GRANT SELECT ON TABLES TO metabase_ro;
