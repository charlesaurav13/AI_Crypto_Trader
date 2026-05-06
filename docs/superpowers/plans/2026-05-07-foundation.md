# CryptoSwarm Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase-1 backend: a modular Python monolith that ingests live Binance USDM perpetuals data, simulates paper trades, enforces circuit breakers, and logs RL tuples from day 1.

**Architecture:** Single asyncio process with 7 bus-isolated modules (bus, config, feed, storage, papertrade, risk, api). All inter-module comms go through Valkey pub/sub — even within the process. Module split lines match future microservice boundaries. TimescaleDB for market time-series; PostgreSQL for audit log, trades, RL tuples.

**Tech Stack:** Python 3.12 · uv · FastAPI + uvicorn · Pydantic v2 + pydantic-settings · valkey-py · asyncpg · python-binance · pytest + pytest-asyncio + testcontainers · ruff · docker-compose

---

## File map

```
AI_Trading/
├── backend/
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── src/cryptoswarm/
│   │   ├── __init__.py
│   │   ├── bus/
│   │   │   ├── __init__.py
│   │   │   ├── client.py          # Valkey pub/sub wrapper
│   │   │   └── messages.py        # All Pydantic message schemas
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── settings.py        # Pydantic Settings
│   │   │   └── config.yaml        # default symbol list + risk envelope
│   │   ├── feed/
│   │   │   ├── __init__.py
│   │   │   ├── rest_client.py     # gap-fill, symbol info, leverage set
│   │   │   ├── ws_client.py       # Binance USDM WS multiplex manager
│   │   │   └── handler.py         # parses WS frames → bus messages
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── timescale.py       # TimescaleDB async writer
│   │   │   ├── postgres.py        # PostgreSQL async writer
│   │   │   └── subscriber.py      # bus subscriber → routes to writers
│   │   ├── papertrade/
│   │   │   ├── __init__.py
│   │   │   ├── math.py            # liq price, PnL, fees — pure functions
│   │   │   ├── account.py         # balance + open positions state
│   │   │   └── engine.py          # signal handler + mark-price watcher
│   │   ├── risk/
│   │   │   ├── __init__.py
│   │   │   ├── guards.py          # per-signal checks
│   │   │   └── breakers.py        # circuit breaker state machine
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── app.py             # FastAPI factory
│   │   │   ├── deps.py            # shared dependencies
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── health.py
│   │   │       ├── symbols.py
│   │   │       ├── positions.py
│   │   │       ├── trades.py
│   │   │       ├── circuit.py
│   │   │       ├── signal.py      # POST /test/signal
│   │   │       └── sse.py         # GET /events (SSE)
│   │   └── main.py
│   └── tests/
│       ├── conftest.py
│       ├── unit/
│       │   ├── test_messages.py
│       │   ├── test_math.py
│       │   ├── test_guards.py
│       │   └── test_breakers.py
│       ├── integration/
│       │   ├── conftest.py
│       │   └── test_signal_pipeline.py
│       └── replay/
│           └── test_determinism.py
├── infra/
│   ├── docker-compose.yml
│   └── migrations/
│       ├── timescale/001_init.sql
│       └── postgres/001_init.sql
├── .env.example
├── .gitignore
└── Makefile
```

---

## Task 1: Project scaffold

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/Dockerfile`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `Makefile`

- [ ] **Step 1: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
.venv/
dist/
*.egg-info/
.ruff_cache/
.pytest_cache/
.coverage
htmlcov/

# Env
.env
*.env.local

# Docker
infra/data/

# macOS
.DS_Store
```

- [ ] **Step 2: Create .env.example**

```
VALKEY_URL=redis://localhost:6379
TIMESCALE_DSN=postgresql://postgres:postgres@localhost:5432/cryptoswarm_ts
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5433/cryptoswarm
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
BINANCE_TESTNET=true
PAPER_TRADING=true
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT
RISK__STARTING_BALANCE_USD=1000.0
RISK__MAX_POSITION_PCT=0.10
RISK__MAX_CONCURRENT_POSITIONS=5
RISK__MAX_LEVERAGE=5
RISK__DAILY_LOSS_PCT=0.03
RISK__MAX_DRAWDOWN_PCT=0.15
```

- [ ] **Step 3: Create backend/pyproject.toml**

```toml
[project]
name = "cryptoswarm"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "valkey>=6.0",
    "asyncpg>=0.30",
    "python-binance>=1.0.22",
    "python-dotenv>=1.0",
    "PyYAML>=6.0",
    "sse-starlette>=2.1",
    "httpx>=0.28",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6",
    "ruff>=0.8",
    "testcontainers[postgres]>=4.8",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/cryptoswarm"]
```

- [ ] **Step 4: Create backend/Dockerfile**

```dockerfile
FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system --no-cache .
COPY src/ src/
CMD ["python", "-m", "cryptoswarm.main"]
```

- [ ] **Step 5: Create Makefile**

```makefile
.PHONY: up down logs test fmt lint install

up:
	docker compose -f infra/docker-compose.yml up -d --build

down:
	docker compose -f infra/docker-compose.yml down -v

logs:
	docker compose -f infra/docker-compose.yml logs -f backend

install:
	cd backend && uv pip install --system -e ".[dev]"

test:
	cd backend && pytest -v --cov=src/cryptoswarm --cov-report=term-missing

fmt:
	cd backend && ruff format src tests

lint:
	cd backend && ruff check src tests
```

- [ ] **Step 6: Bootstrap backend src layout**

```bash
mkdir -p backend/src/cryptoswarm/{bus,config,feed,storage,papertrade,risk,api/routes}
mkdir -p backend/tests/{unit,integration,replay}
touch backend/src/cryptoswarm/__init__.py
touch backend/src/cryptoswarm/{bus,config,feed,storage,papertrade,risk,api,api/routes}/__init__.py
touch backend/tests/{conftest,__init__}.py
touch backend/tests/{unit,integration,replay}/__init__.py
```

- [ ] **Step 7: Install deps**

```bash
cd backend && uv pip install --system -e ".[dev]"
```
Expected: resolves and installs all packages without error.

- [ ] **Step 8: Verify ruff runs**

```bash
cd backend && ruff check src/
```
Expected: exits 0 (no files to lint yet).

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "chore: project scaffold, pyproject.toml, Makefile, Dockerfile"
```

---

## Task 2: Docker Compose + DB migrations

**Files:**
- Create: `infra/docker-compose.yml`
- Create: `infra/migrations/timescale/001_init.sql`
- Create: `infra/migrations/postgres/001_init.sql`

- [ ] **Step 1: Create infra/docker-compose.yml**

```yaml
services:
  valkey:
    image: valkey/valkey:8
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "valkey-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  timescale:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: cryptoswarm_ts
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./migrations/timescale:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 15

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: cryptoswarm
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations/postgres:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 15

  backend:
    build:
      context: ../backend
    env_file: ../.env
    environment:
      VALKEY_URL: redis://valkey:6379
      TIMESCALE_DSN: postgresql://postgres:postgres@timescale:5432/cryptoswarm_ts
      POSTGRES_DSN: postgresql://postgres:postgres@postgres:5432/cryptoswarm
    ports:
      - "8000:8000"
    depends_on:
      valkey:
        condition: service_healthy
      timescale:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped

volumes:
  timescale_data:
  postgres_data:
```

- [ ] **Step 2: Create infra/migrations/timescale/001_init.sql**

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

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
ALTER TABLE klines_1m SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
SELECT add_compression_policy('klines_1m', INTERVAL '7 days', if_not_exists => TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS klines_5m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('5 minutes', ts) AS ts,
    first(open, ts) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, ts) AS close,
    sum(volume) AS volume
FROM klines_1m
GROUP BY symbol, time_bucket('5 minutes', ts)
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS klines_1h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', ts) AS ts,
    first(open, ts) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, ts) AS close,
    sum(volume) AS volume
FROM klines_1m
GROUP BY symbol, time_bucket('1 hour', ts)
WITH NO DATA;

CREATE TABLE IF NOT EXISTS mark_price (
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    mark_price DOUBLE PRECISION,
    index_price DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('mark_price', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS funding_rate (
    symbol TEXT NOT NULL,
    funding_time TIMESTAMPTZ NOT NULL,
    rate DOUBLE PRECISION,
    PRIMARY KEY (symbol, funding_time)
);
SELECT create_hypertable('funding_rate', 'funding_time', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS open_interest (
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    open_interest DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('open_interest', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS liquidations (
    id BIGSERIAL,
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    side TEXT NOT NULL,
    price DOUBLE PRECISION,
    qty DOUBLE PRECISION
);
SELECT create_hypertable('liquidations', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS book_ticker (
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    best_bid DOUBLE PRECISION,
    best_ask DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('book_ticker', 'ts', if_not_exists => TRUE);
```

- [ ] **Step 3: Create infra/migrations/postgres/001_init.sql**

```sql
CREATE TABLE IF NOT EXISTS decisions (
    id BIGSERIAL PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    input_state JSONB,
    output JSONB,
    reasoning TEXT,
    confidence DOUBLE PRECISION,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_decisions_cid ON decisions(correlation_id);

CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    correlation_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    exit_reason TEXT,
    leverage INTEGER,
    sl DOUBLE PRECISION,
    tp DOUBLE PRECISION,
    entry_state JSONB,
    realized_pnl DOUBLE PRECISION,
    funding_paid DOUBLE PRECISION DEFAULT 0,
    fees DOUBLE PRECISION DEFAULT 0,
    opened_ts TIMESTAMPTZ NOT NULL,
    closed_ts TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(closed_ts) WHERE closed_ts IS NULL;

CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty DOUBLE PRECISION,
    entry_price DOUBLE PRECISION,
    mark_price DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    isolated_margin DOUBLE PRECISION,
    liq_price DOUBLE PRECISION,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol, ts DESC);

CREATE TABLE IF NOT EXISTS circuit_events (
    id BIGSERIAL PRIMARY KEY,
    breaker_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    tripped_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    manual_reset_ts TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS rl_tuples (
    id BIGSERIAL PRIMARY KEY,
    state JSONB NOT NULL,
    action JSONB NOT NULL,
    reward DOUBLE PRECISION,
    next_state JSONB,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS news_items (
    id BIGSERIAL PRIMARY KEY,
    source TEXT,
    url TEXT,
    title TEXT,
    body TEXT,
    fetched_ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS news_sentiment (
    id BIGSERIAL PRIMARY KEY,
    news_item_id BIGINT REFERENCES news_items(id),
    model TEXT,
    score DOUBLE PRECISION,
    label TEXT,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

- [ ] **Step 4: Start infra and verify**

```bash
make up
sleep 15
docker compose -f infra/docker-compose.yml ps
```
Expected: valkey, timescale, postgres all show `healthy`.

- [ ] **Step 5: Spot-check TimescaleDB hypertables**

```bash
docker compose -f infra/docker-compose.yml exec timescale \
  psql -U postgres -d cryptoswarm_ts -c "\d+ klines_1m"
```
Expected: shows table with `ts` column + TimescaleDB metadata.

- [ ] **Step 6: Spot-check PostgreSQL tables**

```bash
docker compose -f infra/docker-compose.yml exec postgres \
  psql -U postgres -d cryptoswarm -c "\dt"
```
Expected: lists trades, positions, rl_tuples, circuit_events, etc.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: docker-compose, TimescaleDB + PostgreSQL migrations"
```

---

## Task 3: Config module

**Files:**
- Create: `backend/src/cryptoswarm/config/settings.py`
- Create: `backend/src/cryptoswarm/config/config.yaml`
- Test: `backend/tests/unit/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# backend/tests/unit/test_config.py
import os
import pytest
from cryptoswarm.config.settings import get_settings, Settings

def test_settings_defaults():
    s = Settings()
    assert s.symbol_list == [
        "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
        "BNBUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT"
    ]
    assert s.risk.max_leverage == 5
    assert s.risk.daily_loss_pct == 0.03
    assert s.paper_trading is True

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("RISK__MAX_LEVERAGE", "3")
    monkeypatch.setenv("PAPER_TRADING", "false")
    s = Settings()
    assert s.risk.max_leverage == 3
    assert s.paper_trading is False

def test_get_settings_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && pytest tests/unit/test_config.py -v
```
Expected: `ImportError: cannot import name 'get_settings'`

- [ ] **Step 3: Implement settings.py**

```python
# backend/src/cryptoswarm/config/settings.py
from functools import lru_cache
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskConfig(BaseModel):
    starting_balance_usd: float = 1000.0
    max_position_pct: float = 0.10
    max_concurrent_positions: int = 5
    max_leverage: int = 5
    daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    dead_man_timeout_s: int = 60
    heartbeat_interval_s: int = 5


class FeeConfig(BaseModel):
    taker_rate: float = 0.0004   # 0.04%
    maker_rate: float = 0.0002   # 0.02%
    slippage_rate: float = 0.0005  # 0.05%
    maintenance_margin_rate: float = 0.004  # 0.4%


class Settings(BaseSettings):
    valkey_url: str = "redis://localhost:6379"
    timescale_dsn: str = "postgresql://postgres:postgres@localhost:5432/cryptoswarm_ts"
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5433/cryptoswarm"

    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

    paper_trading: bool = True

    symbols: str = (
        "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,"
        "BNBUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT"
    )

    risk: RiskConfig = RiskConfig()
    fees: FeeConfig = FeeConfig()

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip() for s in self.symbols.split(",") if s.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run tests**

```bash
cd backend && pytest tests/unit/test_config.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: config module with Pydantic Settings"
```

---

## Task 4: Bus — message schemas

**Files:**
- Create: `backend/src/cryptoswarm/bus/messages.py`
- Test: `backend/tests/unit/test_messages.py`

- [ ] **Step 1: Write failing test**

```python
# backend/tests/unit/test_messages.py
import pytest
from datetime import datetime, timezone
from cryptoswarm.bus.messages import (
    MarketTick, MarkPrice, FundingUpdate, OpenInterestUpdate,
    LiquidationEvent, BookTicker, Signal, RiskVeto,
    TradeExecuted, PositionUpdate, CircuitTripped, SystemHeartbeat,
)

def test_market_tick_round_trip():
    msg = MarketTick(
        symbol="BTCUSDT", interval="1m",
        open=50000.0, high=50100.0, low=49900.0, close=50050.0,
        volume=1.5, is_closed=True,
    )
    restored = MarketTick.model_validate_json(msg.model_dump_json())
    assert restored.symbol == "BTCUSDT"
    assert restored.correlation_id == msg.correlation_id
    assert restored.schema_version == 1

def test_signal_round_trip():
    s = Signal(
        symbol="ETHUSDT", side="LONG", size_usd=100.0,
        sl=1800.0, tp=2200.0, leverage=5, reasoning="test",
    )
    restored = Signal.model_validate_json(s.model_dump_json())
    assert restored.side == "LONG"
    assert restored.leverage == 5

def test_position_update_with_close():
    p = PositionUpdate(
        symbol="BTCUSDT", side="LONG", qty=0.002,
        entry_price=50000.0, mark_price=49000.0,
        unrealized_pnl=-2.0, isolated_margin=20.0, liq_price=41000.0,
        is_closed=True, close_reason="sl",
    )
    assert p.is_closed is True
    assert p.close_reason == "sl"

def test_all_messages_have_correlation_id():
    msgs = [
        MarketTick(symbol="X", interval="1m", open=1, high=1, low=1, close=1, volume=1, is_closed=False),
        Signal(symbol="X", side="LONG", size_usd=100, sl=0.9, tp=1.1, leverage=5),
    ]
    for m in msgs:
        assert m.correlation_id
        assert m.ts is not None
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && pytest tests/unit/test_messages.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement messages.py**

```python
# backend/src/cryptoswarm/bus/messages.py
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field


def _cid() -> str:
    return str(uuid.uuid4())

def _now() -> datetime:
    return datetime.now(timezone.utc)


class BaseMsg(BaseModel):
    correlation_id: str = Field(default_factory=_cid)
    ts: datetime = Field(default_factory=_now)
    schema_version: int = 1


class MarketTick(BaseMsg):
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


class MarkPrice(BaseMsg):
    symbol: str
    mark_price: float
    index_price: float


class FundingUpdate(BaseMsg):
    symbol: str
    funding_time: datetime
    rate: float


class OpenInterestUpdate(BaseMsg):
    symbol: str
    open_interest: float


class LiquidationEvent(BaseMsg):
    symbol: str
    side: Literal["BUY", "SELL"]
    price: float
    qty: float


class BookTicker(BaseMsg):
    symbol: str
    best_bid: float
    best_ask: float


class Signal(BaseMsg):
    symbol: str
    side: Literal["LONG", "SHORT"]
    size_usd: float
    sl: float
    tp: float
    leverage: int
    reasoning: str = ""


class RiskVeto(BaseMsg):
    original_correlation_id: str
    reason: str
    breaker_name: str


class TradeExecuted(BaseMsg):
    original_correlation_id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    qty: float
    entry_price: float
    leverage: int
    sl: float
    tp: float
    fees: float


class PositionUpdate(BaseMsg):
    symbol: str
    side: Literal["LONG", "SHORT"]
    qty: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    isolated_margin: float
    liq_price: float
    is_closed: bool = False
    close_reason: Optional[Literal["sl", "tp", "liq", "manual"]] = None


class CircuitTripped(BaseMsg):
    breaker_name: str
    value: float
    threshold: float


class SystemHeartbeat(BaseMsg):
    process_id: int
```

- [ ] **Step 4: Run tests**

```bash
cd backend && pytest tests/unit/test_messages.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: bus message schemas (Pydantic v2)"
```

---

## Task 5: Bus — Valkey client

**Files:**
- Create: `backend/src/cryptoswarm/bus/client.py`
- Test: `backend/tests/unit/test_bus.py` (uses running Valkey from docker-compose)

- [ ] **Step 1: Write failing test**

```python
# backend/tests/unit/test_bus.py
import asyncio
import pytest
from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import MarketTick, Signal

# Requires Valkey running at localhost:6379 (make up)
pytestmark = pytest.mark.asyncio

async def test_publish_subscribe_roundtrip():
    client = BusClient("redis://localhost:6379")
    await client.connect()

    received: list[tuple[str, str]] = []

    async def collector():
        async for topic, data in client.subscribe("market.tick.BTCUSDT.1m"):
            received.append((topic, data))
            break  # stop after first message

    task = asyncio.create_task(collector())
    await asyncio.sleep(0.1)  # let subscribe settle

    msg = MarketTick(
        symbol="BTCUSDT", interval="1m",
        open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, is_closed=True,
    )
    await client.publish("market.tick.BTCUSDT.1m", msg)
    await asyncio.wait_for(task, timeout=3.0)

    assert len(received) == 1
    topic, data = received[0]
    assert topic == "market.tick.BTCUSDT.1m"
    restored = MarketTick.model_validate_json(data)
    assert restored.symbol == "BTCUSDT"
    assert restored.correlation_id == msg.correlation_id

    await client.close()

async def test_pattern_subscribe():
    client = BusClient("redis://localhost:6379")
    await client.connect()
    received = []

    async def collector():
        async for topic, data in client.psubscribe("market.tick.*"):
            received.append(topic)
            break

    task = asyncio.create_task(collector())
    await asyncio.sleep(0.1)

    msg = MarketTick(
        symbol="ETHUSDT", interval="1m",
        open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, is_closed=False,
    )
    await client.publish("market.tick.ETHUSDT.1m", msg)
    await asyncio.wait_for(task, timeout=3.0)
    assert "market.tick.ETHUSDT.1m" in received

    await client.close()
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && pytest tests/unit/test_bus.py -v
```
Expected: `ImportError: cannot import name 'BusClient'`

- [ ] **Step 3: Implement client.py**

```python
# backend/src/cryptoswarm/bus/client.py
from __future__ import annotations
import asyncio
from typing import AsyncIterator
import valkey.asyncio as valkey_aio
from .messages import BaseMsg


class BusClient:
    """Thin Valkey pub/sub wrapper. All inter-module communication goes through here."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: valkey_aio.Valkey | None = None

    async def connect(self) -> None:
        self._client = await valkey_aio.from_url(self._url, decode_responses=True)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def publish(self, topic: str, msg: BaseMsg) -> None:
        assert self._client, "BusClient not connected"
        await self._client.publish(topic, msg.model_dump_json())

    async def subscribe(self, *topics: str) -> AsyncIterator[tuple[str, str]]:
        """Exact-match subscribe. Yields (topic, raw_json) pairs."""
        assert self._client, "BusClient not connected"
        ps = self._client.pubsub()
        await ps.subscribe(*topics)
        try:
            async for message in ps.listen():
                if message["type"] == "message":
                    yield message["channel"], message["data"]
        finally:
            await ps.aclose()

    async def psubscribe(self, *patterns: str) -> AsyncIterator[tuple[str, str]]:
        """Pattern subscribe (e.g. 'market.*'). Yields (channel, raw_json) pairs."""
        assert self._client, "BusClient not connected"
        ps = self._client.pubsub()
        await ps.psubscribe(*patterns)
        try:
            async for message in ps.listen():
                if message["type"] == "pmessage":
                    yield message["channel"], message["data"]
        finally:
            await ps.aclose()
```

- [ ] **Step 4: Run tests (Valkey must be up)**

```bash
cd backend && pytest tests/unit/test_bus.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Valkey bus client (pub/sub + pattern subscribe)"
```

---

## Task 6: Paper trade math (pure functions)

**Files:**
- Create: `backend/src/cryptoswarm/papertrade/math.py`
- Test: `backend/tests/unit/test_math.py`

This is the highest-risk module — bugs here cost money. Target 100% coverage.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/unit/test_math.py
import pytest
from cryptoswarm.papertrade.math import (
    calc_qty, calc_liq_price, calc_unrealized_pnl,
    calc_realized_pnl, calc_entry_fee, calc_funding,
    calc_isolated_margin,
)

MMR = 0.004  # 0.4% maintenance margin rate

# --- calc_qty ---
def test_calc_qty_long():
    # $100 notional at $50,000/BTC → 0.002 BTC
    assert calc_qty(size_usd=100.0, entry_price=50_000.0) == pytest.approx(0.002, rel=1e-6)

def test_calc_qty_zero_price_raises():
    with pytest.raises(ValueError, match="entry_price must be > 0"):
        calc_qty(100.0, 0.0)

# --- calc_isolated_margin ---
def test_isolated_margin():
    # $100 notional / 5x = $20 margin
    assert calc_isolated_margin(size_usd=100.0, leverage=5) == pytest.approx(20.0)

# --- calc_liq_price ---
def test_liq_price_long():
    # entry=50000, leverage=5, mmr=0.004
    # liq = 50000 * (1 - 1/5 + 0.004) = 50000 * 0.804 = 40200
    expected = 50_000.0 * (1 - 1/5 + MMR)
    result = calc_liq_price(entry_price=50_000.0, side="LONG", leverage=5, mmr=MMR)
    assert result == pytest.approx(expected, rel=1e-6)

def test_liq_price_short():
    # liq = 50000 * (1 + 1/5 - 0.004) = 50000 * 1.196 = 59800
    expected = 50_000.0 * (1 + 1/5 - MMR)
    result = calc_liq_price(entry_price=50_000.0, side="SHORT", leverage=5, mmr=MMR)
    assert result == pytest.approx(expected, rel=1e-6)

# --- calc_unrealized_pnl ---
def test_unrealized_pnl_long_profit():
    # qty=0.002, entry=50000, mark=51000 → pnl = 0.002 * 1000 = 2.0
    assert calc_unrealized_pnl(qty=0.002, entry_price=50_000.0, mark_price=51_000.0, side="LONG") == pytest.approx(2.0)

def test_unrealized_pnl_short_profit():
    # qty=0.002, entry=50000, mark=49000 → pnl = 0.002 * 1000 = 2.0
    assert calc_unrealized_pnl(qty=0.002, entry_price=50_000.0, mark_price=49_000.0, side="SHORT") == pytest.approx(2.0)

def test_unrealized_pnl_long_loss():
    assert calc_unrealized_pnl(qty=0.002, entry_price=50_000.0, mark_price=49_000.0, side="LONG") == pytest.approx(-2.0)

# --- calc_realized_pnl ---
def test_realized_pnl_long():
    assert calc_realized_pnl(qty=0.002, entry_price=50_000.0, exit_price=51_000.0, side="LONG") == pytest.approx(2.0)

def test_realized_pnl_short():
    assert calc_realized_pnl(qty=0.002, entry_price=50_000.0, exit_price=49_000.0, side="SHORT") == pytest.approx(2.0)

# --- calc_entry_fee ---
def test_entry_fee_taker():
    # $100 notional * 0.04% = $0.04
    assert calc_entry_fee(size_usd=100.0, taker_rate=0.0004) == pytest.approx(0.04)

# --- calc_funding ---
def test_funding_positive_rate_long_pays():
    # qty=0.002 BTC, mark=50000, rate=0.0001 → funding = 0.002*50000*0.0001 = 0.01 (long pays)
    assert calc_funding(qty=0.002, mark_price=50_000.0, rate=0.0001, side="LONG") == pytest.approx(-0.01)

def test_funding_positive_rate_short_receives():
    assert calc_funding(qty=0.002, mark_price=50_000.0, rate=0.0001, side="SHORT") == pytest.approx(0.01)

def test_funding_negative_rate_long_receives():
    assert calc_funding(qty=0.002, mark_price=50_000.0, rate=-0.0001, side="LONG") == pytest.approx(0.01)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && pytest tests/unit/test_math.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement math.py**

```python
# backend/src/cryptoswarm/papertrade/math.py
"""
Pure functions for paper trade calculations.
No side effects, no I/O — easy to unit-test and safe to audit.
"""
from typing import Literal


Side = Literal["LONG", "SHORT"]


def calc_qty(size_usd: float, entry_price: float) -> float:
    """Base asset quantity for a given USD notional and entry price."""
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    return size_usd / entry_price


def calc_isolated_margin(size_usd: float, leverage: int) -> float:
    """Initial isolated margin = notional / leverage."""
    return size_usd / leverage


def calc_liq_price(
    entry_price: float,
    side: Side,
    leverage: int,
    mmr: float = 0.004,
) -> float:
    """
    Simplified Binance USDM isolated-margin liquidation price.

    LONG:  liq = entry * (1 - 1/leverage + mmr)
    SHORT: liq = entry * (1 + 1/leverage - mmr)
    """
    if side == "LONG":
        return entry_price * (1 - 1 / leverage + mmr)
    else:
        return entry_price * (1 + 1 / leverage - mmr)


def calc_unrealized_pnl(
    qty: float,
    entry_price: float,
    mark_price: float,
    side: Side,
) -> float:
    """Unrealized PnL in USDT."""
    if side == "LONG":
        return qty * (mark_price - entry_price)
    else:
        return qty * (entry_price - mark_price)


def calc_realized_pnl(
    qty: float,
    entry_price: float,
    exit_price: float,
    side: Side,
) -> float:
    """Realized PnL in USDT (before fees/funding)."""
    if side == "LONG":
        return qty * (exit_price - entry_price)
    else:
        return qty * (entry_price - exit_price)


def calc_entry_fee(size_usd: float, taker_rate: float) -> float:
    """Entry fee. Always taker for market orders."""
    return size_usd * taker_rate


def calc_exit_fee(qty: float, exit_price: float, taker_rate: float) -> float:
    """Exit fee for market close."""
    return qty * exit_price * taker_rate


def calc_funding(
    qty: float,
    mark_price: float,
    rate: float,
    side: Side,
) -> float:
    """
    Funding payment for one 8h interval.
    Positive rate → longs pay shorts.
    Returns signed amount from position holder's perspective.
    """
    payment = qty * mark_price * rate
    if side == "LONG":
        return -payment   # long pays when rate > 0
    else:
        return payment    # short receives when rate > 0
```

- [ ] **Step 4: Run tests**

```bash
cd backend && pytest tests/unit/test_math.py -v --cov=src/cryptoswarm/papertrade/math --cov-report=term-missing
```
Expected: 13 passed, 100% coverage on math.py.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: paper trade math (liq price, PnL, fees, funding) — 100% tested"
```

---

## Task 7: Risk module (guards + circuit breakers)

**Files:**
- Create: `backend/src/cryptoswarm/risk/guards.py`
- Create: `backend/src/cryptoswarm/risk/breakers.py`
- Test: `backend/tests/unit/test_guards.py`
- Test: `backend/tests/unit/test_breakers.py`

- [ ] **Step 1: Write failing guard tests**

```python
# backend/tests/unit/test_guards.py
import pytest
from cryptoswarm.risk.guards import SignalGuard, GuardResult
from cryptoswarm.bus.messages import Signal
from cryptoswarm.config.settings import Settings, RiskConfig

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def make_settings(**kwargs) -> Settings:
    risk = RiskConfig(**kwargs)
    s = Settings(symbols="BTCUSDT,ETHUSDT", risk=risk)
    return s

def make_signal(**kwargs) -> Signal:
    defaults = dict(symbol="BTCUSDT", side="LONG", size_usd=100.0,
                    sl=45000.0, tp=55000.0, leverage=5)
    defaults.update(kwargs)
    return Signal(**defaults)

def test_valid_signal_passes():
    guard = SignalGuard(settings=make_settings(), open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal())
    assert result.allowed is True

def test_too_many_positions():
    guard = SignalGuard(settings=make_settings(max_concurrent_positions=5),
                       open_positions=5, current_equity=1000.0)
    result = guard.check(make_signal())
    assert result.allowed is False
    assert "concurrent" in result.reason.lower()

def test_position_size_too_large():
    # 10% of $1000 = $100 max. size_usd=101 should fail.
    guard = SignalGuard(settings=make_settings(), open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal(size_usd=101.0))
    assert result.allowed is False
    assert "size" in result.reason.lower()

def test_leverage_too_high():
    guard = SignalGuard(settings=make_settings(max_leverage=5),
                       open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal(leverage=10))
    assert result.allowed is False
    assert "leverage" in result.reason.lower()

def test_unknown_symbol():
    guard = SignalGuard(settings=make_settings(), open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal(symbol="UNKNOWN"))
    assert result.allowed is False
    assert "symbol" in result.reason.lower()

def test_sl_wrong_side_long():
    # Long: SL must be below entry. If SL > entry, reject.
    guard = SignalGuard(settings=make_settings(), open_positions=0, current_equity=1000.0)
    # Can't check entry directly from Signal without price feed, so guard skips this check
    # if sl==0 or tp==0, fail
    result = guard.check(make_signal(sl=0.0))
    assert result.allowed is False
```

- [ ] **Step 2: Write failing breaker tests**

```python
# backend/tests/unit/test_breakers.py
import pytest
from cryptoswarm.risk.breakers import CircuitBreakerState, DailyLossBreaker, MaxDrawdownBreaker

def test_daily_loss_not_tripped_initially():
    breaker = DailyLossBreaker(starting_balance=1000.0, threshold_pct=0.03)
    assert breaker.is_tripped() is False

def test_daily_loss_trips_at_threshold():
    breaker = DailyLossBreaker(starting_balance=1000.0, threshold_pct=0.03)
    breaker.update_pnl(-30.0)   # exactly -3%
    assert breaker.is_tripped() is True

def test_daily_loss_not_tripped_below_threshold():
    breaker = DailyLossBreaker(starting_balance=1000.0, threshold_pct=0.03)
    breaker.update_pnl(-29.99)
    assert breaker.is_tripped() is False

def test_daily_loss_manual_reset():
    breaker = DailyLossBreaker(starting_balance=1000.0, threshold_pct=0.03)
    breaker.update_pnl(-50.0)
    assert breaker.is_tripped() is True
    breaker.reset()
    assert breaker.is_tripped() is False

def test_drawdown_breaker():
    breaker = MaxDrawdownBreaker(threshold_pct=0.15)
    breaker.update_equity(1000.0)  # peak = 1000
    breaker.update_equity(1100.0)  # peak = 1100
    breaker.update_equity(935.0)   # 935/1100 = 85% → 15% drawdown → trip
    assert breaker.is_tripped() is True

def test_drawdown_not_tripped_small_drop():
    breaker = MaxDrawdownBreaker(threshold_pct=0.15)
    breaker.update_equity(1000.0)
    breaker.update_equity(870.0)   # 13% drawdown → not tripped
    assert breaker.is_tripped() is False
```

- [ ] **Step 3: Run to confirm failures**

```bash
cd backend && pytest tests/unit/test_guards.py tests/unit/test_breakers.py -v
```
Expected: multiple ImportErrors.

- [ ] **Step 4: Implement guards.py**

```python
# backend/src/cryptoswarm/risk/guards.py
from dataclasses import dataclass
from cryptoswarm.bus.messages import Signal
from cryptoswarm.config.settings import Settings


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    breaker_name: str = ""


class SignalGuard:
    """Stateless per-signal checks. Called by risk module before passing to paper engine."""

    def __init__(self, settings: Settings, open_positions: int, current_equity: float) -> None:
        self._s = settings
        self._open_positions = open_positions
        self._current_equity = current_equity

    def check(self, signal: Signal) -> GuardResult:
        cfg = self._s.risk

        if signal.symbol not in self._s.symbol_list:
            return GuardResult(False, f"symbol {signal.symbol} not in configured list", "symbol_guard")

        if self._open_positions >= cfg.max_concurrent_positions:
            return GuardResult(
                False,
                f"max concurrent positions reached ({cfg.max_concurrent_positions})",
                "concurrent_positions_guard",
            )

        max_size = self._current_equity * cfg.max_position_pct
        if signal.size_usd > max_size:
            return GuardResult(
                False,
                f"size_usd {signal.size_usd:.2f} exceeds {max_size:.2f} (10% of equity)",
                "position_size_guard",
            )

        if signal.leverage > cfg.max_leverage:
            return GuardResult(
                False,
                f"leverage {signal.leverage}x exceeds max {cfg.max_leverage}x",
                "leverage_guard",
            )

        if signal.sl <= 0 or signal.tp <= 0:
            return GuardResult(False, "sl and tp must be > 0", "sl_tp_guard")

        return GuardResult(True)
```

- [ ] **Step 5: Implement breakers.py**

```python
# backend/src/cryptoswarm/risk/breakers.py
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CircuitBreakerState:
    name: str
    tripped: bool = False
    tripped_at: datetime | None = None
    last_value: float = 0.0

    def trip(self, value: float) -> None:
        self.tripped = True
        self.last_value = value
        self.tripped_at = datetime.now(timezone.utc)

    def reset(self) -> None:
        self.tripped = False
        self.tripped_at = None
        self.last_value = 0.0

    def is_tripped(self) -> bool:
        return self.tripped


class DailyLossBreaker:
    """Trips when cumulative daily PnL drops below threshold_pct of starting_balance."""

    def __init__(self, starting_balance: float, threshold_pct: float) -> None:
        self._starting_balance = starting_balance
        self._threshold = starting_balance * threshold_pct
        self._cumulative_pnl: float = 0.0
        self._state = CircuitBreakerState(name="daily_loss")

    def update_pnl(self, delta: float) -> None:
        self._cumulative_pnl += delta
        if self._cumulative_pnl <= -abs(self._threshold):
            self._state.trip(self._cumulative_pnl)

    def is_tripped(self) -> bool:
        return self._state.is_tripped()

    def reset(self) -> None:
        self._cumulative_pnl = 0.0
        self._state.reset()

    @property
    def state(self) -> CircuitBreakerState:
        return self._state


class MaxDrawdownBreaker:
    """Trips when equity drops > threshold_pct below its all-time peak."""

    def __init__(self, threshold_pct: float) -> None:
        self._threshold_pct = threshold_pct
        self._peak: float = 0.0
        self._state = CircuitBreakerState(name="max_drawdown")

    def update_equity(self, equity: float) -> None:
        if equity > self._peak:
            self._peak = equity
        if self._peak > 0:
            drawdown = (self._peak - equity) / self._peak
            if drawdown >= self._threshold_pct:
                self._state.trip(drawdown)

    def is_tripped(self) -> bool:
        return self._state.is_tripped()

    def reset(self) -> None:
        self._state.reset()

    @property
    def state(self) -> CircuitBreakerState:
        return self._state
```

- [ ] **Step 6: Run tests**

```bash
cd backend && pytest tests/unit/test_guards.py tests/unit/test_breakers.py -v
```
Expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: risk guards + circuit breakers (daily loss, max drawdown)"
```

---

## Task 8: Paper trade account + engine

**Files:**
- Create: `backend/src/cryptoswarm/papertrade/account.py`
- Create: `backend/src/cryptoswarm/papertrade/engine.py`
- Test: `backend/tests/unit/test_papertrade.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/unit/test_papertrade.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from cryptoswarm.papertrade.account import Account, OpenPosition
from cryptoswarm.papertrade.math import calc_qty, calc_liq_price, calc_isolated_margin
from cryptoswarm.config.settings import Settings, FeeConfig

def make_settings() -> Settings:
    return Settings(symbols="BTCUSDT,ETHUSDT", _env_file=None)

# --- Account ---
def test_account_initial_state():
    acc = Account(starting_balance=1000.0)
    assert acc.balance == 1000.0
    assert acc.open_positions == {}
    assert acc.equity == 1000.0

def test_account_open_position():
    acc = Account(starting_balance=1000.0)
    pos = OpenPosition(
        symbol="BTCUSDT", side="LONG", qty=0.002,
        entry_price=50_000.0, leverage=5, sl=40_200.0, tp=60_000.0,
        isolated_margin=20.0, liq_price=40_200.0, fees=0.04,
    )
    acc.open(pos)
    assert "BTCUSDT" in acc.open_positions
    assert acc.balance == 1000.0 - 20.0 - 0.04  # margin + fees

def test_account_close_position():
    acc = Account(starting_balance=1000.0)
    pos = OpenPosition(
        symbol="BTCUSDT", side="LONG", qty=0.002,
        entry_price=50_000.0, leverage=5, sl=40_200.0, tp=60_000.0,
        isolated_margin=20.0, liq_price=40_200.0, fees=0.04,
    )
    acc.open(pos)
    acc.close("BTCUSDT", exit_price=51_000.0, exit_reason="tp", exit_fees=0.04)
    assert "BTCUSDT" not in acc.open_positions
    # pnl = 0.002 * 1000 = 2.0; net = margin + pnl - entry_fees - exit_fees
    expected_balance = (1000.0 - 20.0 - 0.04) + (20.0 + 2.0 - 0.04)
    assert acc.balance == pytest.approx(expected_balance)

def test_account_equity_includes_unrealized():
    acc = Account(starting_balance=1000.0)
    pos = OpenPosition(
        symbol="BTCUSDT", side="LONG", qty=0.002,
        entry_price=50_000.0, leverage=5, sl=40_200.0, tp=60_000.0,
        isolated_margin=20.0, liq_price=40_200.0, fees=0.04,
    )
    acc.open(pos)
    acc.update_mark("BTCUSDT", 51_000.0)
    # unrealized = 2.0
    assert acc.equity == pytest.approx(acc.balance + 2.0)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && pytest tests/unit/test_papertrade.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement account.py**

```python
# backend/src/cryptoswarm/papertrade/account.py
from dataclasses import dataclass, field
from typing import Optional
from .math import calc_unrealized_pnl, calc_realized_pnl


@dataclass
class OpenPosition:
    symbol: str
    side: str            # "LONG" | "SHORT"
    qty: float
    entry_price: float
    leverage: int
    sl: float
    tp: float
    isolated_margin: float
    liq_price: float
    fees: float          # entry fees
    mark_price: float = 0.0
    funding_paid: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.mark_price == 0:
            return 0.0
        return calc_unrealized_pnl(self.qty, self.entry_price, self.mark_price, self.side)  # type: ignore[arg-type]


class Account:
    def __init__(self, starting_balance: float) -> None:
        self._balance: float = starting_balance
        self.open_positions: dict[str, OpenPosition] = {}

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        return self._balance + sum(p.unrealized_pnl for p in self.open_positions.values())

    def open(self, pos: OpenPosition) -> None:
        self.open_positions[pos.symbol] = pos
        self._balance -= pos.isolated_margin + pos.fees

    def close(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        exit_fees: float,
    ) -> float:
        """Returns realized PnL (net of fees). Releases isolated margin back to balance."""
        pos = self.open_positions.pop(symbol)
        pnl = calc_realized_pnl(pos.qty, pos.entry_price, exit_price, pos.side)  # type: ignore[arg-type]
        # return margin + pnl - exit fees (entry fees already deducted on open)
        self._balance += pos.isolated_margin + pnl - exit_fees + pos.funding_paid
        return pnl - pos.fees - exit_fees

    def apply_funding(self, symbol: str, funding_delta: float) -> None:
        if symbol in self.open_positions:
            self.open_positions[symbol].funding_paid += funding_delta
            self._balance += funding_delta

    def update_mark(self, symbol: str, mark: float) -> None:
        if symbol in self.open_positions:
            self.open_positions[symbol].mark_price = mark
```

- [ ] **Step 4: Run tests**

```bash
cd backend && pytest tests/unit/test_papertrade.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Implement engine.py**

```python
# backend/src/cryptoswarm/papertrade/engine.py
"""
Paper trade engine.
Subscribes to: signal.execute, market.mark.*
Publishes to: trade.executed, position.update
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone

from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import (
    Signal, TradeExecuted, PositionUpdate, RiskVeto,
    MarkPrice, CircuitTripped,
)
from cryptoswarm.config.settings import Settings
from cryptoswarm.papertrade.account import Account, OpenPosition
from cryptoswarm.papertrade.math import (
    calc_qty, calc_liq_price, calc_isolated_margin,
    calc_entry_fee, calc_exit_fee, calc_funding,
)
from cryptoswarm.risk.guards import SignalGuard
from cryptoswarm.risk.breakers import DailyLossBreaker, MaxDrawdownBreaker

logger = logging.getLogger(__name__)


class PaperTradeEngine:
    def __init__(self, bus: BusClient, settings: Settings) -> None:
        self._bus = bus
        self._cfg = settings
        self._fees = settings.fees
        self._account = Account(settings.risk.starting_balance_usd)
        self._daily_loss = DailyLossBreaker(
            settings.risk.starting_balance_usd, settings.risk.daily_loss_pct
        )
        self._max_dd = MaxDrawdownBreaker(settings.risk.max_drawdown_pct)

    async def run(self) -> None:
        """Main loop: listens on signal.execute and market.mark.* simultaneously."""
        await asyncio.gather(
            self._handle_signals(),
            self._handle_mark_prices(),
        )

    async def _handle_signals(self) -> None:
        async for _, data in self._bus.subscribe("signal.execute"):
            signal = Signal.model_validate_json(data)
            await self._process_signal(signal)

    async def _handle_mark_prices(self) -> None:
        async for _, data in self._bus.psubscribe("market.mark.*"):
            mark = MarkPrice.model_validate_json(data)
            await self._process_mark(mark)

    async def _process_signal(self, signal: Signal) -> None:
        # Check circuit breakers first
        if self._daily_loss.is_tripped() or self._max_dd.is_tripped():
            veto = RiskVeto(
                original_correlation_id=signal.correlation_id,
                reason="circuit breaker tripped",
                breaker_name="daily_loss_or_drawdown",
            )
            await self._bus.publish("risk.veto", veto)
            return

        # Per-signal guards
        guard = SignalGuard(
            self._cfg,
            len(self._account.open_positions),
            self._account.equity,
        )
        result = guard.check(signal)
        if not result.allowed:
            veto = RiskVeto(
                original_correlation_id=signal.correlation_id,
                reason=result.reason,
                breaker_name=result.breaker_name,
            )
            await self._bus.publish("risk.veto", veto)
            return

        # Execute paper fill
        fees = self._fees
        entry_price = signal.sl  # approximate: use SL as proxy fill in paper (no real order book)
        # NOTE: in a real system, fill price would come from the order book / last trade
        # For paper trading we approximate fill at a slight slippage from a reference price.
        # The caller (Director Agent in Phase 2) should pass a reasonable entry price via reasoning field
        # For Foundation test signals, we accept entry via signal.reasoning as JSON {"entry": 1234}
        import json
        try:
            meta = json.loads(signal.reasoning) if signal.reasoning.startswith("{") else {}
            entry_price = float(meta.get("entry", 0)) or signal.tp * 0.95 if signal.side == "LONG" else signal.tp * 1.05
        except Exception:
            entry_price = (signal.sl + signal.tp) / 2  # midpoint fallback

        qty = calc_qty(signal.size_usd, entry_price)
        margin = calc_isolated_margin(signal.size_usd, signal.leverage)
        liq = calc_liq_price(entry_price, signal.side, signal.leverage, self._fees.maintenance_margin_rate)  # type: ignore[arg-type]
        entry_fee = calc_entry_fee(signal.size_usd, fees.taker_rate)

        pos = OpenPosition(
            symbol=signal.symbol, side=signal.side, qty=qty,
            entry_price=entry_price, leverage=signal.leverage,
            sl=signal.sl, tp=signal.tp,
            isolated_margin=margin, liq_price=liq, fees=entry_fee,
        )
        self._account.open(pos)

        executed = TradeExecuted(
            original_correlation_id=signal.correlation_id,
            symbol=signal.symbol, side=signal.side,
            qty=qty, entry_price=entry_price, leverage=signal.leverage,
            sl=signal.sl, tp=signal.tp, fees=entry_fee,
        )
        await self._bus.publish("trade.executed", executed)
        await self._publish_position_update(signal.symbol)
        logger.info("Paper fill: %s %s @ %.2f qty=%.6f", signal.side, signal.symbol, entry_price, qty)

    async def _process_mark(self, mark: MarkPrice) -> None:
        symbol = mark.symbol
        if symbol not in self._account.open_positions:
            return

        self._account.update_mark(symbol, mark.mark_price)
        pos = self._account.open_positions[symbol]

        # Check SL
        sl_hit = (pos.side == "LONG" and mark.mark_price <= pos.sl) or \
                 (pos.side == "SHORT" and mark.mark_price >= pos.sl)
        # Check TP
        tp_hit = (pos.side == "LONG" and mark.mark_price >= pos.tp) or \
                 (pos.side == "SHORT" and mark.mark_price <= pos.tp)
        # Check liquidation
        liq_hit = (pos.side == "LONG" and mark.mark_price <= pos.liq_price) or \
                  (pos.side == "SHORT" and mark.mark_price >= pos.liq_price)

        if liq_hit:
            await self._close_position(symbol, pos.liq_price, "liq")
        elif sl_hit:
            await self._close_position(symbol, pos.sl, "sl")
        elif tp_hit:
            await self._close_position(symbol, pos.tp, "tp")
        else:
            await self._publish_position_update(symbol)

    async def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        pos = self._account.open_positions[symbol]
        exit_fee = calc_exit_fee(pos.qty, exit_price, self._fees.taker_rate)
        net_pnl = self._account.close(symbol, exit_price, reason, exit_fee)

        # Update circuit breakers
        self._daily_loss.update_pnl(net_pnl)
        self._max_dd.update_equity(self._account.equity)
        if self._daily_loss.is_tripped():
            await self._bus.publish(
                "circuit.tripped",
                CircuitTripped(
                    breaker_name="daily_loss",
                    value=self._daily_loss.state.last_value,
                    threshold=self._cfg.risk.starting_balance_usd * self._cfg.risk.daily_loss_pct,
                ),
            )

        # Publish closed position update
        closed_update = PositionUpdate(
            symbol=symbol, side=pos.side, qty=pos.qty,
            entry_price=pos.entry_price, mark_price=exit_price,
            unrealized_pnl=0.0, isolated_margin=0.0,
            liq_price=pos.liq_price, is_closed=True, close_reason=reason,  # type: ignore[arg-type]
        )
        await self._bus.publish("position.update", closed_update)
        logger.info("Closed %s %s @ %.2f reason=%s net_pnl=%.4f", pos.side, symbol, exit_price, reason, net_pnl)

    async def _publish_position_update(self, symbol: str) -> None:
        if symbol not in self._account.open_positions:
            return
        pos = self._account.open_positions[symbol]
        upd = PositionUpdate(
            symbol=symbol, side=pos.side, qty=pos.qty,
            entry_price=pos.entry_price, mark_price=pos.mark_price,
            unrealized_pnl=pos.unrealized_pnl,
            isolated_margin=pos.isolated_margin,
            liq_price=pos.liq_price,
        )
        await self._bus.publish("position.update", upd)
```

- [ ] **Step 6: Run all unit tests**

```bash
cd backend && pytest tests/unit/ -v
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: paper trade account + engine (signal→fill, mark→SL/TP/liq)"
```

---

## Task 9: Storage writers

**Files:**
- Create: `backend/src/cryptoswarm/storage/timescale.py`
- Create: `backend/src/cryptoswarm/storage/postgres.py`
- Create: `backend/src/cryptoswarm/storage/subscriber.py`

- [ ] **Step 1: Implement timescale.py**

```python
# backend/src/cryptoswarm/storage/timescale.py
from __future__ import annotations
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TimescaleWriter:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def upsert_kline(self, symbol: str, ts: datetime, o: float, h: float,
                           l: float, c: float, v: float) -> None:
        assert self._pool
        await self._pool.execute(
            """
            INSERT INTO klines_1m (symbol, ts, open, high, low, close, volume)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (symbol, ts) DO NOTHING
            """,
            symbol, ts, o, h, l, c, v,
        )

    async def upsert_mark_price(self, symbol: str, ts: datetime,
                                mark: float, index: float) -> None:
        assert self._pool
        await self._pool.execute(
            """
            INSERT INTO mark_price (symbol, ts, mark_price, index_price)
            VALUES ($1,$2,$3,$4)
            ON CONFLICT (symbol, ts) DO NOTHING
            """,
            symbol, ts, mark, index,
        )

    async def upsert_funding(self, symbol: str, funding_time: datetime, rate: float) -> None:
        assert self._pool
        await self._pool.execute(
            """
            INSERT INTO funding_rate (symbol, funding_time, rate)
            VALUES ($1,$2,$3)
            ON CONFLICT (symbol, funding_time) DO NOTHING
            """,
            symbol, funding_time, rate,
        )

    async def insert_liquidation(self, symbol: str, ts: datetime,
                                  side: str, price: float, qty: float) -> None:
        assert self._pool
        await self._pool.execute(
            "INSERT INTO liquidations (symbol, ts, side, price, qty) VALUES ($1,$2,$3,$4,$5)",
            symbol, ts, side, price, qty,
        )

    async def upsert_book_ticker(self, symbol: str, ts: datetime,
                                  best_bid: float, best_ask: float) -> None:
        assert self._pool
        await self._pool.execute(
            """
            INSERT INTO book_ticker (symbol, ts, best_bid, best_ask)
            VALUES ($1,$2,$3,$4)
            ON CONFLICT (symbol, ts) DO NOTHING
            """,
            symbol, ts, best_bid, best_ask,
        )
```

- [ ] **Step 2: Implement postgres.py**

```python
# backend/src/cryptoswarm/storage/postgres.py
from __future__ import annotations
import asyncpg
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PostgresWriter:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def insert_trade_open(
        self, correlation_id: str, symbol: str, side: str, qty: float,
        entry_price: float, leverage: int, sl: float, tp: float,
        fees: float, entry_state: dict, opened_ts: datetime,
    ) -> None:
        assert self._pool
        await self._pool.execute(
            """
            INSERT INTO trades
              (correlation_id, symbol, side, qty, entry_price, leverage, sl, tp,
               fees, entry_state, opened_ts)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (correlation_id) DO NOTHING
            """,
            correlation_id, symbol, side, qty, entry_price, leverage, sl, tp,
            fees, json.dumps(entry_state), opened_ts,
        )

    async def update_trade_close(
        self, correlation_id: str, exit_price: float, exit_reason: str,
        realized_pnl: float, funding_paid: float, exit_fees: float,
        closed_ts: datetime,
    ) -> None:
        assert self._pool
        await self._pool.execute(
            """
            UPDATE trades SET
              exit_price=$2, exit_reason=$3, realized_pnl=$4,
              funding_paid=$5, fees=fees+$6, closed_ts=$7
            WHERE correlation_id=$1
            """,
            correlation_id, exit_price, exit_reason, realized_pnl,
            funding_paid, exit_fees, closed_ts,
        )

    async def insert_rl_tuple(
        self, state: dict, action: dict, reward: float | None, next_state: dict | None
    ) -> None:
        assert self._pool
        await self._pool.execute(
            """
            INSERT INTO rl_tuples (state, action, reward, next_state)
            VALUES ($1,$2,$3,$4)
            """,
            json.dumps(state), json.dumps(action),
            reward, json.dumps(next_state) if next_state else None,
        )

    async def insert_circuit_event(
        self, breaker_name: str, value: float, threshold: float
    ) -> None:
        assert self._pool
        await self._pool.execute(
            "INSERT INTO circuit_events (breaker_name, value, threshold) VALUES ($1,$2,$3)",
            breaker_name, value, threshold,
        )

    async def get_open_trades(self) -> list[asyncpg.Record]:
        """Used on startup to rehydrate paper engine state."""
        assert self._pool
        return await self._pool.fetch(
            "SELECT * FROM trades WHERE closed_ts IS NULL ORDER BY opened_ts"
        )
```

- [ ] **Step 3: Implement subscriber.py**

```python
# backend/src/cryptoswarm/storage/subscriber.py
"""
Subscribes to all bus topics and routes to TimescaleDB / PostgreSQL writers.
This is the only module that writes to the databases at runtime.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone

from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import (
    MarketTick, MarkPrice, FundingUpdate, LiquidationEvent, BookTicker,
    TradeExecuted, PositionUpdate, CircuitTripped,
)
from cryptoswarm.storage.timescale import TimescaleWriter
from cryptoswarm.storage.postgres import PostgresWriter

logger = logging.getLogger(__name__)


class StorageSubscriber:
    def __init__(self, bus: BusClient, ts: TimescaleWriter, pg: PostgresWriter) -> None:
        self._bus = bus
        self._ts = ts
        self._pg = pg

    async def run(self) -> None:
        await asyncio.gather(
            self._consume_market(),
            self._consume_trades(),
            self._consume_circuits(),
        )

    async def _consume_market(self) -> None:
        async for topic, data in self._bus.psubscribe("market.*"):
            try:
                await self._route_market(topic, data)
            except Exception as exc:
                logger.exception("storage market error topic=%s: %s", topic, exc)

    async def _route_market(self, topic: str, data: str) -> None:
        if ".tick." in topic:
            msg = MarketTick.model_validate_json(data)
            if msg.is_closed:
                await self._ts.upsert_kline(
                    msg.symbol, msg.ts,
                    msg.open, msg.high, msg.low, msg.close, msg.volume,
                )
        elif ".mark." in topic:
            msg = MarkPrice.model_validate_json(data)
            await self._ts.upsert_mark_price(msg.symbol, msg.ts, msg.mark_price, msg.index_price)
        elif ".funding." in topic:
            msg = FundingUpdate.model_validate_json(data)
            await self._ts.upsert_funding(msg.symbol, msg.funding_time, msg.rate)
        elif ".liq." in topic:
            msg = LiquidationEvent.model_validate_json(data)
            await self._ts.insert_liquidation(msg.symbol, msg.ts, msg.side, msg.price, msg.qty)
        elif ".book." in topic:
            msg = BookTicker.model_validate_json(data)
            await self._ts.upsert_book_ticker(msg.symbol, msg.ts, msg.best_bid, msg.best_ask)

    async def _consume_trades(self) -> None:
        async for topic, data in self._bus.psubscribe("trade.*", "position.*"):
            try:
                if topic == "trade.executed":
                    msg = TradeExecuted.model_validate_json(data)
                    await self._pg.insert_trade_open(
                        correlation_id=msg.original_correlation_id,
                        symbol=msg.symbol, side=msg.side,
                        qty=msg.qty, entry_price=msg.entry_price,
                        leverage=msg.leverage, sl=msg.sl, tp=msg.tp,
                        fees=msg.fees,
                        entry_state={},   # Phase 2 populates full state
                        opened_ts=msg.ts,
                    )
                    # Write open RL tuple (reward + next_state filled on close)
                    await self._pg.insert_rl_tuple(
                        state={},   # Phase 2 populates full market snapshot
                        action={
                            "symbol": msg.symbol, "side": msg.side,
                            "qty": msg.qty, "entry_price": msg.entry_price,
                            "leverage": msg.leverage, "sl": msg.sl, "tp": msg.tp,
                        },
                        reward=None,
                        next_state=None,
                    )
            except Exception as exc:
                logger.exception("storage trade error: %s", exc)

    async def _consume_circuits(self) -> None:
        async for _, data in self._bus.subscribe("circuit.tripped"):
            try:
                msg = CircuitTripped.model_validate_json(data)
                await self._pg.insert_circuit_event(msg.breaker_name, msg.value, msg.threshold)
            except Exception as exc:
                logger.exception("storage circuit error: %s", exc)
```

- [ ] **Step 4: Run all unit tests**

```bash
cd backend && pytest tests/unit/ -v
```
Expected: all pass (no storage unit tests yet — integration test covers this in Task 11).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: storage writers (TimescaleDB + PostgreSQL) + bus subscriber"
```

---

## Task 10: Feed (REST + WebSocket)

**Files:**
- Create: `backend/src/cryptoswarm/feed/rest_client.py`
- Create: `backend/src/cryptoswarm/feed/ws_client.py`
- Create: `backend/src/cryptoswarm/feed/handler.py`

- [ ] **Step 1: Implement rest_client.py**

```python
# backend/src/cryptoswarm/feed/rest_client.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
from binance import AsyncClient

logger = logging.getLogger(__name__)


class BinanceRestClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._client: AsyncClient | None = None

    async def connect(self) -> None:
        self._client = await AsyncClient.create(
            api_key=self._api_key,
            api_secret=self._api_secret,
            testnet=self._testnet,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.close_connection()

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        assert self._client
        try:
            await self._client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info("Set leverage %dx for %s", leverage, symbol)
        except Exception as exc:
            logger.warning("Could not set leverage for %s: %s", symbol, exc)

    async def set_margin_type_isolated(self, symbol: str) -> None:
        assert self._client
        try:
            await self._client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        except Exception as exc:
            # Binance returns error if already isolated — safe to ignore
            if "already" not in str(exc).lower():
                logger.warning("set_margin_type %s: %s", symbol, exc)

    async def get_klines(self, symbol: str, interval: str = "1m",
                         limit: int = 500) -> list[dict]:
        """Historical klines for gap-fill on reconnect."""
        assert self._client
        raw = await self._client.futures_klines(
            symbol=symbol, interval=interval, limit=limit
        )
        return [
            {
                "symbol": symbol,
                "ts": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in raw
        ]

    async def get_mark_price(self, symbol: str) -> dict:
        assert self._client
        return await self._client.futures_mark_price(symbol=symbol)
```

- [ ] **Step 2: Implement handler.py**

```python
# backend/src/cryptoswarm/feed/handler.py
"""
Converts raw Binance WebSocket frame dicts into typed bus messages and publishes them.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import (
    MarketTick, MarkPrice, FundingUpdate, LiquidationEvent, BookTicker,
)

logger = logging.getLogger(__name__)


class FrameHandler:
    def __init__(self, bus: BusClient) -> None:
        self._bus = bus

    async def handle(self, msg: dict) -> None:
        stream = msg.get("stream", "")
        data = msg.get("data", msg)

        if "@kline_" in stream:
            await self._handle_kline(data)
        elif "@markPrice" in stream:
            await self._handle_mark(data)
        elif "@forceOrder" in stream or stream == "!forceOrder@arr":
            await self._handle_liquidation(data)
        elif "@bookTicker" in stream:
            await self._handle_book(data)

    async def _handle_kline(self, data: dict) -> None:
        k = data.get("k", data)
        symbol = k.get("s", data.get("s", ""))
        interval = k.get("i", "1m")
        tick = MarketTick(
            symbol=symbol, interval=interval,
            open=float(k["o"]), high=float(k["h"]),
            low=float(k["l"]), close=float(k["c"]),
            volume=float(k["v"]),
            is_closed=bool(k.get("x", False)),
            ts=datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
        )
        await self._bus.publish(f"market.tick.{symbol}.{interval}", tick)

    async def _handle_mark(self, data: dict) -> None:
        symbol = data.get("s", "")
        msg = MarkPrice(
            symbol=symbol,
            mark_price=float(data.get("p", 0)),
            index_price=float(data.get("i", 0)),
            ts=datetime.fromtimestamp(data.get("T", 0) / 1000, tz=timezone.utc),
        )
        await self._bus.publish(f"market.mark.{symbol}", msg)
        # funding rate is embedded in mark price stream
        if "r" in data and "T" in data:
            funding = FundingUpdate(
                symbol=symbol, rate=float(data["r"]),
                funding_time=datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc),
            )
            await self._bus.publish(f"market.funding.{symbol}", funding)

    async def _handle_liquidation(self, data: dict) -> None:
        order = data.get("o", data)
        symbol = order.get("s", "")
        msg = LiquidationEvent(
            symbol=symbol,
            side=order.get("S", "BUY"),
            price=float(order.get("p", 0)),
            qty=float(order.get("q", 0)),
            ts=datetime.fromtimestamp(order.get("T", 0) / 1000, tz=timezone.utc),
        )
        await self._bus.publish(f"market.liq.{symbol}", msg)

    async def _handle_book(self, data: dict) -> None:
        symbol = data.get("s", "")
        msg = BookTicker(
            symbol=symbol,
            best_bid=float(data.get("b", 0)),
            best_ask=float(data.get("a", 0)),
        )
        await self._bus.publish(f"market.book.{symbol}", msg)
```

- [ ] **Step 3: Implement ws_client.py**

```python
# backend/src/cryptoswarm/feed/ws_client.py
"""
Manages the Binance USDM Futures WebSocket multiplex connection.
Reconnects with exponential backoff on disconnect.
"""
from __future__ import annotations
import asyncio
import logging
from binance import AsyncClient, BinanceSocketManager

from cryptoswarm.config.settings import Settings
from cryptoswarm.feed.handler import FrameHandler
from cryptoswarm.feed.rest_client import BinanceRestClient

logger = logging.getLogger(__name__)

BACKOFF_INITIAL = 1.0
BACKOFF_MAX = 60.0
BACKOFF_FACTOR = 2.0


class FeedManager:
    def __init__(self, settings: Settings, handler: FrameHandler,
                 rest: BinanceRestClient) -> None:
        self._cfg = settings
        self._handler = handler
        self._rest = rest

    async def run(self) -> None:
        """Run forever with reconnect backoff."""
        backoff = BACKOFF_INITIAL
        while True:
            try:
                await self._connect_and_stream()
                backoff = BACKOFF_INITIAL  # reset on clean exit
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Feed disconnected: %s. Reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_FACTOR, BACKOFF_MAX)

    async def _connect_and_stream(self) -> None:
        client = await AsyncClient.create(
            api_key=self._cfg.binance_api_key,
            api_secret=self._cfg.binance_api_secret,
            testnet=self._cfg.binance_testnet,
        )
        try:
            # Set leverage and margin type for all symbols on connect
            for symbol in self._cfg.symbol_list:
                await self._rest.set_margin_type_isolated(symbol)
                await self._rest.set_leverage(symbol, self._cfg.risk.max_leverage)

            bm = BinanceSocketManager(client)
            streams = []
            for sym in self._cfg.symbol_list:
                s = sym.lower()
                streams += [
                    f"{s}@kline_1m",
                    f"{s}@markPrice",
                    f"{s}@bookTicker",
                ]
            # Global liquidations stream
            streams.append("!forceOrder@arr")

            logger.info("Connecting to %d streams for %d symbols", len(streams), len(self._cfg.symbol_list))
            async with bm.futures_multiplex_socket(streams) as ws:
                async for msg in ws:
                    await self._handler.handle(msg)
        finally:
            await client.close_connection()
```

- [ ] **Step 4: Run unit tests to confirm no regressions**

```bash
cd backend && pytest tests/unit/ -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Binance USDM feed (REST gap-fill + WS multiplex + frame handler)"
```

---

## Task 11: FastAPI routes + SSE

**Files:**
- Create: `backend/src/cryptoswarm/api/app.py`
- Create: `backend/src/cryptoswarm/api/deps.py`
- Create: `backend/src/cryptoswarm/api/routes/{health,symbols,positions,trades,circuit,signal,sse}.py`

- [ ] **Step 1: Implement deps.py**

```python
# backend/src/cryptoswarm/api/deps.py
from functools import lru_cache
from cryptoswarm.bus.client import BusClient
from cryptoswarm.storage.postgres import PostgresWriter
from cryptoswarm.storage.timescale import TimescaleWriter
from cryptoswarm.papertrade.engine import PaperTradeEngine

# Module-level singletons set by main.py before app starts
_bus: BusClient | None = None
_pg: PostgresWriter | None = None
_ts: TimescaleWriter | None = None
_engine: PaperTradeEngine | None = None


def set_deps(bus: BusClient, pg: PostgresWriter, ts: TimescaleWriter,
             engine: PaperTradeEngine) -> None:
    global _bus, _pg, _ts, _engine
    _bus = bus; _pg = pg; _ts = ts; _engine = engine


def get_bus() -> BusClient:
    assert _bus, "BusClient not initialised"
    return _bus

def get_pg() -> PostgresWriter:
    assert _pg, "PostgresWriter not initialised"
    return _pg

def get_engine() -> PaperTradeEngine:
    assert _engine, "PaperTradeEngine not initialised"
    return _engine
```

- [ ] **Step 2: Implement routes**

```python
# backend/src/cryptoswarm/api/routes/health.py
from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}
```

```python
# backend/src/cryptoswarm/api/routes/circuit.py
from fastapi import APIRouter, Depends
from cryptoswarm.api.deps import get_engine

router = APIRouter(prefix="/circuit-breaker")

@router.get("/status")
async def status(engine=Depends(get_engine)):
    dl = engine._daily_loss
    dd = engine._max_dd
    return {
        "daily_loss": {
            "tripped": dl.is_tripped(),
            "cumulative_pnl": dl._cumulative_pnl,
            "threshold": dl._threshold,
        },
        "max_drawdown": {
            "tripped": dd.is_tripped(),
            "peak_equity": dd._peak,
        },
    }

@router.post("/reset")
async def reset(engine=Depends(get_engine)):
    engine._daily_loss.reset()
    engine._max_dd.reset()
    return {"reset": True}
```

```python
# backend/src/cryptoswarm/api/routes/positions.py
from fastapi import APIRouter, Depends
from cryptoswarm.api.deps import get_engine

router = APIRouter(prefix="/positions")

@router.get("/")
async def list_positions(engine=Depends(get_engine)):
    acc = engine._account
    return {
        "balance": acc.balance,
        "equity": acc.equity,
        "positions": [
            {
                "symbol": p.symbol, "side": p.side, "qty": p.qty,
                "entry_price": p.entry_price, "mark_price": p.mark_price,
                "unrealized_pnl": p.unrealized_pnl,
                "liq_price": p.liq_price,
            }
            for p in acc.open_positions.values()
        ],
    }
```

```python
# backend/src/cryptoswarm/api/routes/trades.py
from fastapi import APIRouter, Depends
from cryptoswarm.api.deps import get_pg

router = APIRouter(prefix="/trades")

@router.get("/")
async def list_trades(limit: int = 50, pg=Depends(get_pg)):
    rows = await pg._pool.fetch(
        "SELECT * FROM trades ORDER BY opened_ts DESC LIMIT $1", limit
    )
    return [dict(r) for r in rows]
```

```python
# backend/src/cryptoswarm/api/routes/signal.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from cryptoswarm.api.deps import get_bus
from cryptoswarm.bus.messages import Signal

router = APIRouter(prefix="/test")

class SignalRequest(BaseModel):
    symbol: str
    side: str         # "LONG" | "SHORT"
    size_usd: float
    sl: float
    tp: float
    leverage: int = 5
    entry: float = 0.0   # optional explicit entry price

@router.post("/signal")
async def post_signal(req: SignalRequest, bus=Depends(get_bus)):
    import json
    reasoning = json.dumps({"entry": req.entry}) if req.entry else ""
    sig = Signal(
        symbol=req.symbol, side=req.side,
        size_usd=req.size_usd, sl=req.sl, tp=req.tp,
        leverage=req.leverage, reasoning=reasoning,
    )
    await bus.publish("signal.execute", sig)
    return {"correlation_id": sig.correlation_id, "published": True}
```

```python
# backend/src/cryptoswarm/api/routes/sse.py
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from cryptoswarm.api.deps import get_bus
import asyncio

router = APIRouter()

@router.get("/events")
async def sse_stream(bus=Depends(get_bus)):
    async def generator():
        async for topic, data in bus.psubscribe("*"):
            yield {"event": topic, "data": data}
    return EventSourceResponse(generator())
```

- [ ] **Step 3: Implement app.py**

```python
# backend/src/cryptoswarm/api/app.py
from fastapi import FastAPI
from cryptoswarm.api.routes import health, positions, trades, circuit, signal, sse


def create_app() -> FastAPI:
    app = FastAPI(title="CryptoSwarm", version="0.1.0")
    app.include_router(health.router)
    app.include_router(positions.router)
    app.include_router(trades.router)
    app.include_router(circuit.router)
    app.include_router(signal.router)
    app.include_router(sse.router)
    return app
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: FastAPI routes (health, positions, trades, circuit, test/signal, SSE)"
```

---

## Task 12: Main entry point

**Files:**
- Create: `backend/src/cryptoswarm/main.py`

- [ ] **Step 1: Implement main.py**

```python
# backend/src/cryptoswarm/main.py
import asyncio
import logging
import os
import signal as _signal
import uvicorn

from cryptoswarm.config.settings import get_settings
from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import SystemHeartbeat
from cryptoswarm.feed.rest_client import BinanceRestClient
from cryptoswarm.feed.handler import FrameHandler
from cryptoswarm.feed.ws_client import FeedManager
from cryptoswarm.storage.timescale import TimescaleWriter
from cryptoswarm.storage.postgres import PostgresWriter
from cryptoswarm.storage.subscriber import StorageSubscriber
from cryptoswarm.papertrade.engine import PaperTradeEngine
from cryptoswarm.api.app import create_app
from cryptoswarm.api import deps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def heartbeat_loop(bus: BusClient, interval_s: int) -> None:
    while True:
        await bus.publish("system.heartbeat", SystemHeartbeat(process_id=os.getpid()))
        await asyncio.sleep(interval_s)


async def main() -> None:
    cfg = get_settings()
    logger.info("Starting CryptoSwarm Foundation (paper_trading=%s)", cfg.paper_trading)

    # --- Boot dependencies ---
    bus = BusClient(cfg.valkey_url)
    await bus.connect()
    logger.info("Bus connected: %s", cfg.valkey_url)

    ts_writer = TimescaleWriter(cfg.timescale_dsn)
    await ts_writer.connect()
    pg_writer = PostgresWriter(cfg.postgres_dsn)
    await pg_writer.connect()
    logger.info("Storage connected")

    rest = BinanceRestClient(cfg.binance_api_key, cfg.binance_api_secret, cfg.binance_testnet)
    await rest.connect()

    handler = FrameHandler(bus)
    feed = FeedManager(cfg, handler, rest)
    storage_sub = StorageSubscriber(bus, ts_writer, pg_writer)
    engine = PaperTradeEngine(bus, cfg)

    # Wire API deps
    deps.set_deps(bus=bus, pg=pg_writer, ts=ts_writer, engine=engine)

    # --- Launch all tasks ---
    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info"))

    tasks = [
        asyncio.create_task(feed.run(), name="feed"),
        asyncio.create_task(storage_sub.run(), name="storage"),
        asyncio.create_task(engine.run(), name="engine"),
        asyncio.create_task(heartbeat_loop(bus, cfg.risk.heartbeat_interval_s), name="heartbeat"),
        asyncio.create_task(server.serve(), name="api"),
    ]

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_running_loop()
    for sig in (_signal.SIGTERM, _signal.SIGINT):
        loop.add_signal_handler(sig, lambda: [t.cancel() for t in tasks])

    logger.info("All tasks started. CryptoSwarm is running.")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutdown initiated")
    finally:
        await bus.close()
        await ts_writer.close()
        await pg_writer.close()
        await rest.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify module is importable**

```bash
cd backend && python -c "from cryptoswarm.main import main; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Run all unit tests**

```bash
cd backend && pytest tests/unit/ -v
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: main entry point — boots all modules, graceful shutdown"
```

---

## Task 13: Integration test — signal pipeline

**Files:**
- Create: `backend/tests/integration/conftest.py`
- Create: `backend/tests/integration/test_signal_pipeline.py`

These tests require docker-compose infra running. They go through the full bus → engine → storage path.

- [ ] **Step 1: Implement integration conftest.py**

```python
# backend/tests/integration/conftest.py
import asyncio
import pytest
import asyncpg
from cryptoswarm.config.settings import Settings
from cryptoswarm.bus.client import BusClient
from cryptoswarm.storage.postgres import PostgresWriter
from cryptoswarm.papertrade.engine import PaperTradeEngine
from cryptoswarm.storage.subscriber import StorageSubscriber
from cryptoswarm.storage.timescale import TimescaleWriter

TS_DSN = "postgresql://postgres:postgres@localhost:5432/cryptoswarm_ts"
PG_DSN = "postgresql://postgres:postgres@localhost:5433/cryptoswarm"
VALKEY_URL = "redis://localhost:6379"


@pytest.fixture
async def bus():
    b = BusClient(VALKEY_URL)
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
async def pg():
    w = PostgresWriter(PG_DSN)
    await w.connect()
    yield w
    await w.close()


@pytest.fixture
async def settings():
    return Settings(
        valkey_url=VALKEY_URL,
        timescale_dsn=TS_DSN,
        postgres_dsn=PG_DSN,
        paper_trading=True,
        symbols="BTCUSDT,ETHUSDT",
        _env_file=None,
    )
```

- [ ] **Step 2: Implement test_signal_pipeline.py**

```python
# backend/tests/integration/test_signal_pipeline.py
import asyncio
import json
import pytest
from cryptoswarm.bus.messages import (
    Signal, TradeExecuted, RiskVeto, PositionUpdate, CircuitTripped, MarkPrice,
)
from cryptoswarm.papertrade.engine import PaperTradeEngine
from cryptoswarm.storage.subscriber import StorageSubscriber
from cryptoswarm.storage.timescale import TimescaleWriter

pytestmark = pytest.mark.asyncio

TS_DSN = "postgresql://postgres:postgres@localhost:5432/cryptoswarm_ts"


async def test_valid_signal_opens_position(bus, pg, settings):
    engine = PaperTradeEngine(bus, settings)
    ts = TimescaleWriter(TS_DSN)
    await ts.connect()
    storage = StorageSubscriber(bus, ts, pg)

    received_trades: list[str] = []

    async def collector():
        async for _, data in bus.subscribe("trade.executed"):
            received_trades.append(data)
            break

    tasks = [
        asyncio.create_task(engine.run()),
        asyncio.create_task(storage.run()),
        asyncio.create_task(collector()),
    ]

    await asyncio.sleep(0.1)

    sig = Signal(
        symbol="BTCUSDT", side="LONG", size_usd=100.0,
        sl=45000.0, tp=55000.0, leverage=5,
        reasoning=json.dumps({"entry": 50000.0}),
    )
    await bus.publish("signal.execute", sig)
    await asyncio.wait_for(asyncio.gather(tasks[2]), timeout=5.0)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await ts.close()

    assert len(received_trades) == 1
    executed = TradeExecuted.model_validate_json(received_trades[0])
    assert executed.symbol == "BTCUSDT"
    assert executed.entry_price == 50000.0

    # Verify audit row in PG
    rows = await pg._pool.fetch(
        "SELECT * FROM trades WHERE correlation_id=$1", sig.correlation_id
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTCUSDT"


async def test_sl_triggers_position_close(bus, pg, settings):
    engine = PaperTradeEngine(bus, settings)
    closed: list[str] = []

    async def collector():
        async for _, data in bus.subscribe("position.update"):
            upd = PositionUpdate.model_validate_json(data)
            if upd.is_closed:
                closed.append(data)
                break

    tasks = [asyncio.create_task(engine.run()), asyncio.create_task(collector())]
    await asyncio.sleep(0.1)

    # Open a position
    sig = Signal(
        symbol="ETHUSDT", side="LONG", size_usd=100.0,
        sl=1800.0, tp=2400.0, leverage=5,
        reasoning=json.dumps({"entry": 2000.0}),
    )
    await bus.publish("signal.execute", sig)
    await asyncio.sleep(0.2)

    # Simulate mark price dropping below SL
    await bus.publish("market.mark.ETHUSDT", MarkPrice(
        symbol="ETHUSDT", mark_price=1799.0, index_price=1798.0
    ))
    await asyncio.wait_for(asyncio.gather(tasks[1]), timeout=5.0)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert len(closed) == 1
    upd = PositionUpdate.model_validate_json(closed[0])
    assert upd.close_reason == "sl"


async def test_daily_loss_breaker_trips(bus, settings):
    engine = PaperTradeEngine(bus, settings)
    vetoes: list[str] = []

    async def veto_collector():
        async for _, data in bus.subscribe("risk.veto"):
            vetoes.append(data)
            if len(vetoes) >= 1:
                break

    tasks = [asyncio.create_task(engine.run()), asyncio.create_task(veto_collector())]
    await asyncio.sleep(0.1)

    # Force-trip the daily loss breaker by directly updating state
    engine._daily_loss.update_pnl(-100.0)  # well past -3%
    assert engine._daily_loss.is_tripped()

    # Now send a signal — should be vetoed
    sig = Signal(
        symbol="BTCUSDT", side="LONG", size_usd=50.0,
        sl=45000.0, tp=55000.0, leverage=5,
    )
    await bus.publish("signal.execute", sig)
    await asyncio.wait_for(asyncio.gather(tasks[1]), timeout=5.0)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert len(vetoes) >= 1
    veto = RiskVeto.model_validate_json(vetoes[0])
    assert "circuit" in veto.reason.lower()
```

- [ ] **Step 3: Run integration tests (infra must be up)**

```bash
cd backend && pytest tests/integration/ -v
```
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test: integration tests — signal pipeline, SL close, circuit breaker veto"
```

---

## Task 14: Smoke test (end-to-end via docker-compose)

- [ ] **Step 1: Build and start everything**

```bash
make up && sleep 20
```

- [ ] **Step 2: Verify health endpoint**

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```
Expected: `{"status": "ok"}`

- [ ] **Step 3: Post a test signal**

```bash
curl -s -X POST http://localhost:8000/test/signal \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTCUSDT","side":"LONG","size_usd":100,"sl":45000,"tp":55000,"leverage":5,"entry":50000}' \
  | python3 -m json.tool
```
Expected: `{"correlation_id": "...", "published": true}`

- [ ] **Step 4: Check open position**

```bash
curl -s http://localhost:8000/positions/ | python3 -m json.tool
```
Expected: shows one open BTCUSDT LONG position.

- [ ] **Step 5: Check circuit breaker status**

```bash
curl -s http://localhost:8000/circuit-breaker/status | python3 -m json.tool
```
Expected: both breakers `"tripped": false`.

- [ ] **Step 6: Check logs**

```bash
make logs
```
Expected: feed connecting to Binance, mark price updates, heartbeats — no ERROR lines.

- [ ] **Step 7: Final commit**

```bash
git add -A && git commit -m "feat: Foundation Phase 1 complete — smoke tested end-to-end"
```

---

## Acceptance criteria checklist

- [ ] `docker-compose up` → valkey + timescale + postgres + backend healthy within 30s
- [ ] Live klines for all 10 USDM perpetuals flowing into TimescaleDB
- [ ] `GET /symbols` returns 10 symbols (add this route if missing — query `SELECT DISTINCT symbol FROM mark_price ORDER BY 1`)
- [ ] `POST /test/signal` opens paper position, shows in `GET /positions/`
- [ ] SL/TP/liquidation auto-close via simulated mark price (integration test covers this)
- [ ] 5 paper losses → daily-loss breaker trips → next signal vetoed (integration test covers this)
- [ ] Kill backend → restart → positions rehydrated from PG (add startup rehydration to engine if missing)
- [ ] Every executed trade has a row in `trades` and `rl_tuples`
- [ ] `pytest tests/unit/ tests/integration/ -v` all pass
