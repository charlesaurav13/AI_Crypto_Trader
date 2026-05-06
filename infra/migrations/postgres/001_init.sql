-- Agent decision audit log (Phase 2 populates, schema ready now)
CREATE TABLE IF NOT EXISTS decisions (
    id             BIGSERIAL PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    agent_name     TEXT NOT NULL,
    input_state    JSONB,
    output         JSONB,
    reasoning      TEXT,
    confidence     DOUBLE PRECISION,
    ts             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_decisions_cid ON decisions(correlation_id);

-- Trade audit log (paper + live)
CREATE TABLE IF NOT EXISTS trades (
    id             BIGSERIAL PRIMARY KEY,
    correlation_id TEXT NOT NULL UNIQUE,
    symbol         TEXT NOT NULL,
    side           TEXT NOT NULL,
    qty            DOUBLE PRECISION NOT NULL,
    entry_price    DOUBLE PRECISION NOT NULL,
    exit_price     DOUBLE PRECISION,
    exit_reason    TEXT,
    leverage       INTEGER,
    sl             DOUBLE PRECISION,
    tp             DOUBLE PRECISION,
    entry_state    JSONB,
    realized_pnl   DOUBLE PRECISION,
    funding_paid   DOUBLE PRECISION DEFAULT 0,
    fees           DOUBLE PRECISION DEFAULT 0,
    opened_ts      TIMESTAMPTZ NOT NULL,
    closed_ts      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_open   ON trades(closed_ts) WHERE closed_ts IS NULL;

-- Position snapshots (written on every update)
CREATE TABLE IF NOT EXISTS positions (
    id             BIGSERIAL PRIMARY KEY,
    symbol         TEXT NOT NULL,
    side           TEXT NOT NULL,
    qty            DOUBLE PRECISION,
    entry_price    DOUBLE PRECISION,
    mark_price     DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    isolated_margin DOUBLE PRECISION,
    liq_price      DOUBLE PRECISION,
    ts             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol, ts DESC);

-- Circuit breaker event log
CREATE TABLE IF NOT EXISTS circuit_events (
    id             BIGSERIAL PRIMARY KEY,
    breaker_name   TEXT NOT NULL,
    value          DOUBLE PRECISION,
    threshold      DOUBLE PRECISION,
    tripped_ts     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    manual_reset_ts TIMESTAMPTZ
);

-- RL training tuples: (state, action, reward, next_state) — written from day 1
CREATE TABLE IF NOT EXISTS rl_tuples (
    id         BIGSERIAL PRIMARY KEY,
    state      JSONB NOT NULL,
    action     JSONB NOT NULL,
    reward     DOUBLE PRECISION,
    next_state JSONB,
    ts         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- News pipeline (Phase 2 Sentiment Agent — schema reserved)
CREATE TABLE IF NOT EXISTS news_items (
    id        BIGSERIAL PRIMARY KEY,
    source    TEXT,
    url       TEXT,
    title     TEXT,
    body      TEXT,
    fetched_ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS news_sentiment (
    id           BIGSERIAL PRIMARY KEY,
    news_item_id BIGINT REFERENCES news_items(id),
    model        TEXT,
    score        DOUBLE PRECISION,
    label        TEXT,
    ts           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
