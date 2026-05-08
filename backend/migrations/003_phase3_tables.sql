-- News scraping tables (schema was reserved in Phase 1)
CREATE TABLE IF NOT EXISTS news_items (
    id         BIGSERIAL PRIMARY KEY,
    source     TEXT NOT NULL,
    url        TEXT UNIQUE NOT NULL,
    title      TEXT,
    body       TEXT,
    fetched_ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS news_sentiment (
    id           BIGSERIAL PRIMARY KEY,
    news_item_id BIGINT REFERENCES news_items(id),
    symbol       TEXT NOT NULL,
    model        TEXT NOT NULL,
    relevance    FLOAT NOT NULL,
    score        FLOAT NOT NULL,
    summary      TEXT,
    ts           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_ts
    ON news_sentiment(symbol, ts DESC);

-- Versioned agent system prompts
CREATE TABLE IF NOT EXISTS agent_prompts (
    id            BIGSERIAL PRIMARY KEY,
    agent_name    TEXT NOT NULL,
    version       INT  NOT NULL DEFAULT 1,
    system_prompt TEXT NOT NULL,
    perf_score    FLOAT,
    active        BOOLEAN NOT NULL DEFAULT true,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(agent_name, version)
);

-- ML inference audit log
CREATE TABLE IF NOT EXISTS ml_signals (
    id               BIGSERIAL PRIMARY KEY,
    symbol           TEXT NOT NULL,
    ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
    regime_pred      TEXT NOT NULL,
    direction_pred   TEXT NOT NULL,
    short_direction  TEXT NOT NULL,
    confidence       FLOAT NOT NULL,
    size_adjustment  TEXT NOT NULL,
    model_version    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ml_signals_symbol_ts
    ON ml_signals(symbol, ts DESC);

-- ML training run audit log
CREATE TABLE IF NOT EXISTS training_runs (
    id           BIGSERIAL PRIMARY KEY,
    model_type   TEXT NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    sample_count INT,
    metrics      JSONB
);

-- Add id column to rl_tuples if missing
ALTER TABLE rl_tuples ADD COLUMN IF NOT EXISTS id BIGSERIAL;

-- Idempotent column additions for pre-existing news_sentiment table
-- (Phase 1 created news_sentiment with fewer columns — add missing ones)
ALTER TABLE news_sentiment ADD COLUMN IF NOT EXISTS symbol       TEXT    NOT NULL DEFAULT '';
ALTER TABLE news_sentiment ADD COLUMN IF NOT EXISTS model        TEXT    NOT NULL DEFAULT 'unknown';
ALTER TABLE news_sentiment ADD COLUMN IF NOT EXISTS relevance    FLOAT   NOT NULL DEFAULT 0.0;
ALTER TABLE news_sentiment ADD COLUMN IF NOT EXISTS score        FLOAT   NOT NULL DEFAULT 0.0;
ALTER TABLE news_sentiment ADD COLUMN IF NOT EXISTS summary      TEXT;
CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_ts ON news_sentiment(symbol, ts DESC);
