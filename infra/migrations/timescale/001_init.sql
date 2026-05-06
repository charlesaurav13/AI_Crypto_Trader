CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV 1-minute candles (primary)
CREATE TABLE IF NOT EXISTS klines_1m (
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('klines_1m', 'ts', if_not_exists => TRUE);
ALTER TABLE klines_1m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('klines_1m', INTERVAL '7 days', if_not_exists => TRUE);

-- Continuous aggregates: 5m, 1h
CREATE MATERIALIZED VIEW IF NOT EXISTS klines_5m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('5 minutes', ts) AS ts,
    first(open, ts)  AS open,
    max(high)        AS high,
    min(low)         AS low,
    last(close, ts)  AS close,
    sum(volume)      AS volume
FROM klines_1m
GROUP BY symbol, time_bucket('5 minutes', ts)
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS klines_1h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', ts) AS ts,
    first(open, ts)  AS open,
    max(high)        AS high,
    min(low)         AS low,
    last(close, ts)  AS close,
    sum(volume)      AS volume
FROM klines_1m
GROUP BY symbol, time_bucket('1 hour', ts)
WITH NO DATA;

-- Mark price (for liquidation math and RL state)
CREATE TABLE IF NOT EXISTS mark_price (
    symbol     TEXT NOT NULL,
    ts         TIMESTAMPTZ NOT NULL,
    mark_price DOUBLE PRECISION,
    index_price DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('mark_price', 'ts', if_not_exists => TRUE);

-- Funding rates (every 8h)
CREATE TABLE IF NOT EXISTS funding_rate (
    symbol       TEXT NOT NULL,
    funding_time TIMESTAMPTZ NOT NULL,
    rate         DOUBLE PRECISION,
    PRIMARY KEY (symbol, funding_time)
);
SELECT create_hypertable('funding_rate', 'funding_time', if_not_exists => TRUE);

-- Open interest
CREATE TABLE IF NOT EXISTS open_interest (
    symbol        TEXT NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    open_interest DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('open_interest', 'ts', if_not_exists => TRUE);

-- Force liquidations from Binance
CREATE TABLE IF NOT EXISTS liquidations (
    id     BIGSERIAL,
    symbol TEXT NOT NULL,
    ts     TIMESTAMPTZ NOT NULL,
    side   TEXT NOT NULL,
    price  DOUBLE PRECISION,
    qty    DOUBLE PRECISION
);
SELECT create_hypertable('liquidations', 'ts', if_not_exists => TRUE);

-- Top-of-book (best bid/ask)
CREATE TABLE IF NOT EXISTS book_ticker (
    symbol   TEXT NOT NULL,
    ts       TIMESTAMPTZ NOT NULL,
    best_bid DOUBLE PRECISION,
    best_ask DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('book_ticker', 'ts', if_not_exists => TRUE);
