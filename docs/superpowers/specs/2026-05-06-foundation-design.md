# CryptoSwarm — Foundation Design (Phase 1)

**Status:** Approved through brainstorming · pending implementation plan
**Date:** 2026-05-06
**Owner:** Saurav Pandey
**Source blueprint:** `~/Downloads/cryptoswarm_blueprint.html`

---

## 1. Context

CryptoSwarm is a multi-agent AI trading system for **Binance USDM perpetual futures**, designed in 5 phases over 12 weeks per the source blueprint. This spec covers **Phase 1: Foundation** — the data, persistence, and paper-trading skeleton that everything else (agents, ML, dashboard, production hardening) sits on.

**Goal of Foundation:** ship a working, observable, testable backend that

1. ingests live Binance USDM market data for the configured symbols,
2. accepts hand-crafted trade signals via an HTTP endpoint,
3. simulates paper-trades through the same code path real agents will use later,
4. enforces circuit breakers from day 1,
5. logs every decision in a form Phase 3 RL can train on.

**Not in scope:** real agents (Phase 2) · ML models or RL training (Phase 3) · dashboard UI (Phase 4) · production deployment / live trading (Phase 5) · news scraping pipeline (Phase 2 Sentiment Agent).

---

## 2. Decisions locked in brainstorming

| Topic | Decision | Alternatives considered |
|---|---|---|
| Execution layer | Swarms framework + AutoHedge architectural pattern + Binance USDM | AutoHedge as-is on Solana; both adapters in v1 |
| Instrument | USDM perpetual futures only | Spot only; spot + futures |
| Symbols | Top 10 by volume, configurable. v1 default: `BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, BNBUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, SUIUSDT` | Top 3; explicit list |
| Trading envelope | Max 5x leverage · isolated margin · one-way position mode | 3x conservative; 10x aggressive; cross/hedge |
| Architecture style | Modular monolith with bus-first internal communication | Pragmatic monolith; microservices from day 1 |
| Cache & bus | Valkey 8 (BSD-3) | Redis (license concerns post-2024) |
| LLM (agents) | Claude (Anthropic) — Phase 2 | OpenAI |
| LLM (sentiment / scraping) | Qwen via Ollama — Phase 2 | Claude (cost), OpenAI |
| Web scraping | ScrapeGraphAI — Phase 2 | Custom scrapers |
| Secrets | `.env` + python-dotenv for v1; Vault deferred to Phase 5 | Vault from day 1 (overkill) |
| Python tooling | `uv` | pip + venv; poetry |
| Binance client | `python-binance` | `ccxt` (cleaner async path for futures-only) |

## 3. Operating envelope (defaults — config-driven)

- Starting paper balance: **$1,000 USDT**
- Max position size: **10%** of balance per trade
- Max concurrent positions: **5**
- Daily-loss circuit breaker: **−3%** of starting balance
- Max-drawdown halt: **−15%** of equity peak (rolling)
- Funding accrual: every 8h on mark price
- Liquidation simulation: force-close when mark crosses simulated isolated-margin liq price
- Mode default: **paper trading on**, live trading off, no live toggle exists in Foundation

---

## 4. Repo layout

```
AI_Trading/
├── backend/                    # the single Python service
│   ├── pyproject.toml          # uv-managed
│   ├── src/cryptoswarm/
│   │   ├── bus/                # Valkey pub/sub + typed Pydantic messages
│   │   ├── config/             # env + YAML loader (symbols, risk envelope)
│   │   ├── feed/               # Binance USDM WS + REST clients
│   │   ├── storage/            # TimescaleDB + PG writers + migrations
│   │   ├── papertrade/         # simulated futures engine
│   │   ├── risk/               # circuit breakers + position guards
│   │   ├── api/                # FastAPI routes + SSE
│   │   └── main.py             # boots all modules as asyncio tasks
│   └── tests/
├── infra/
│   ├── docker-compose.yml      # valkey, timescale, postgres, backend
│   └── migrations/
│       ├── timescale/
│       └── postgres/
├── docs/superpowers/specs/     # this file lives here
├── .env.example
├── .gitignore
├── Makefile                    # up · down · logs · test · fmt · lint
└── README.md
```

`agents/`, `dashboard/`, `ml/` are **deferred** — added when Phases 2–4 land, not as empty stubs now.

---

## 5. Components — modular monolith, bus-isolated

All inter-module communication goes through the bus, even within a single process. Module boundaries match future microservice split lines.

| Module | Responsibility | Bus topics (pub / sub) |
|---|---|---|
| **bus** | Valkey pub/sub wrapper. `publish(topic, msg)`, `subscribe(topic)`. All messages are Pydantic v2 models with `correlation_id`, `ts`, `schema_version`. | (provides bus) |
| **config** | Loads `.env` + `config.yaml`. Validates via Pydantic Settings. Hot-reload deferred. | (provides config) |
| **feed** | Binance USDM Futures WebSocket: klines (1m), depth (top-of-book), mark price, funding, OI, force-orders. REST: gap-fill, account info, leverage set. | pubs `market.tick.*`, `market.mark.*`, `market.funding.*`, `market.oi.*`, `market.liq.*`, `market.book.*` |
| **storage** | TimescaleDB writer (hypertables) + PostgreSQL writer (audit + RL tuples). Idempotent inserts with `ON CONFLICT DO NOTHING`. | subs `market.*`, `trade.*`, `position.*`, `agent.decision.*`, `circuit.*`, `risk.*` |
| **papertrade** | Simulated USDM futures account. Tracks balance, isolated margin per position, mark-price liquidation, 8h funding accrual. Fees: 0.04% taker / 0.02% maker (Binance USDM defaults, configurable). Slippage: 0.05% market-order assumption. | subs `signal.execute`, `market.mark.*`; pubs `trade.executed`, `position.update` |
| **risk** | Circuit breakers + per-signal guards. Hardcoded — Director Agent (Phase 2) cannot override. | subs `signal.execute`, `position.update`; pubs `risk.veto`, `circuit.tripped` |
| **api** | FastAPI: `/health`, `/symbols`, `/positions`, `/trades`, `/config`, `/circuit-breaker/status`, `POST /test/signal`, plus SSE stream of bus events. | subs everything (read-only) |
| **main** | Boots Valkey/TS/PG connections, spawns asyncio tasks, graceful shutdown, publishes `system.heartbeat` every 5s for the dead-man switch. | pubs `system.heartbeat` |

## 6. Tech stack — Foundation only

Python 3.12 · `uv` · FastAPI + uvicorn · Pydantic v2 · `python-binance` · `valkey-py` · `asyncpg` · pytest + pytest-asyncio · ruff · docker-compose.

---

## 7. Data flow

### 7.1 Live market ingestion (always running)

```
Binance WS  →  feed module  →  bus(market.*)  ┬→  storage  →  TimescaleDB
                                              └→  api SSE   →  (future dashboard)
```

### 7.2 Paper-trade decision

In Foundation no agents exist; signals come from `POST /test/signal` so the integration is exercised end-to-end. Phase 2 Director Agent will publish to the same `signal.execute` topic — the path is identical.

```
test endpoint OR future Director Agent
       │
       ▼
   bus(signal.execute)
       │
       ▼
   risk module ──┬── [veto]  ──→ bus(risk.veto)      ──→ storage (PG audit) ──→ api SSE
                 │
                 └── [allow] ──→ papertrade
                                     │
                                     ├── bus(trade.executed)  ──→ storage (PG audit + RL tuple)
                                     └── bus(position.update) ──→ storage + risk + api SSE
```

The papertrade engine *also* runs:
- a **mark-price watcher** subscribed to `market.mark.{symbol}` — on every tick, re-evaluates open positions for SL hit / TP hit / liquidation, publishes `position.update` and (on close) `trade.executed`,
- a **funding timer** that accrues funding every 8h on mark price.

## 8. Bus message catalog (Pydantic v2)

| Topic | Payload — key fields |
|---|---|
| `market.tick.{symbol}.{interval}` | symbol, ts, o, h, l, c, v, interval |
| `market.mark.{symbol}` | symbol, ts, mark_price, index_price |
| `market.funding.{symbol}` | symbol, funding_time, rate |
| `market.oi.{symbol}` | symbol, ts, open_interest |
| `market.liq.{symbol}` | symbol, ts, side, price, qty |
| `market.book.{symbol}` | symbol, ts, best_bid, best_ask |
| `signal.execute` | correlation_id, symbol, side, size_usd, sl, tp, leverage, reasoning |
| `risk.veto` | correlation_id, reason, breaker_name |
| `trade.executed` | correlation_id, symbol, side, qty, entry, leverage, sl, tp, fees |
| `position.update` | symbol, side, qty, entry, mark, unrealized_pnl, isolated_margin, liq_price |
| `circuit.tripped` | breaker_name, value, threshold, ts |
| `system.heartbeat` | ts, process_id |
| `agent.decision.{agent}` | (Phase 2 — schema reserved) |

Every message carries `correlation_id`, `ts`, `schema_version`. Schema versions incremented on breaking change; storage handles old rows on replay.

---

## 9. Storage

### 9.1 TimescaleDB (hot, time-series, high-volume)

Hypertables, chunked by time, partitioned by symbol where row volume justifies it:

- `klines_1m(symbol, ts, o, h, l, c, v)` — primary OHLCV. Continuous aggregates roll **1m → 5m / 15m / 1h / 4h / 1d** in-DB.
- `mark_price(symbol, ts, mark, index)` — for liquidation math.
- `funding_rate(symbol, funding_time, rate)` — every 8h.
- `open_interest(symbol, ts, oi)`.
- `liquidations(symbol, ts, side, price, qty)` — from `@forceOrder`.
- `book_ticker(symbol, ts, best_bid, best_ask)` — top of book only. Full depth deferred.

Compression: enabled after 7 days. Retention: raw 1m kept 18 months; aggregates kept indefinitely.

### 9.2 PostgreSQL (cold, audit, source-of-truth)

- `decisions(id, correlation_id, agent_name, input_state JSONB, output JSONB, reasoning TEXT, confidence, ts)` — schema ready, populated by Phase 2.
- `trades(id, correlation_id, symbol, side, qty, entry_price, exit_price, exit_reason, entry_state JSONB, realized_pnl, funding_paid, fees, opened_ts, closed_ts)`.
- `positions(id, symbol, side, qty, entry, mark, unrealized_pnl, isolated_margin, liq_price, ts)` — written every update.
- `circuit_events(id, breaker_name, value, threshold, ts, manual_reset_ts)`.
- `rl_tuples(id, state JSONB, action JSONB, reward FLOAT, next_state JSONB, ts)` — **populated from day 1**.
- `news_items(id, source, url, title, body TEXT, fetched_ts)` — schema reserved for Phase 2 Sentiment Agent (ScrapeGraphAI).
- `news_sentiment(id, news_item_id FK, model TEXT, score FLOAT, label TEXT, ts)` — schema reserved for Phase 2.

### 9.3 Storage split rationale

Time-series data (millions of rows/day) belongs in TimescaleDB; audit/decision/JSONB-heavy data (thousands of rows/day, replayed) belongs in PG. Different access patterns, different optimizations. Both run as separate Docker services in compose.

### 9.4 Idempotency & replay

- All inserts keyed on `(symbol, ts)` or `correlation_id` with `ON CONFLICT DO NOTHING`. Crashes don't duplicate.
- Bus messages are fire-and-forget (Valkey doesn't persist); persistence is the storage module's job.
- Replay tool reads `decisions` + `trades` from PG and re-publishes to a sandbox bus channel — for debugging and Phase 3 RL training off historical data.

---

## 10. RL data capture (Phase-3-readiness from day 1)

Every executed paper trade writes one row to `rl_tuples` on close:

- **state** JSONB: market snapshot at decision time — mark price, last 60 1m klines for the symbol, latest funding rate, latest OI, current portfolio state (open positions, balance), current circuit-breaker state.
- **action** JSONB: the trade — symbol, side, size_usd, leverage, sl, tp.
- **reward** FLOAT: realized P&L net of funding and fees. Computed at trade close.
- **next_state** JSONB: market + portfolio snapshot at trade close.
- **ts** TIMESTAMPTZ.

Phase 3 RL training then becomes a SQL-out-then-train problem, not a re-instrument-the-system problem.

The reward function shape is intentionally Phase-3-deferred — Foundation just captures raw P&L, fees, funding. Risk-adjusted reward (differential Sharpe, drawdown penalty, etc.) is a Phase-3 design decision applied on top of the captured tuples.

---

## 11. Risk — circuit breakers and per-signal guards

### 11.1 Circuit breakers (hardcoded; Director Agent cannot override)

| Breaker | Threshold | Action |
|---|---|---|
| **Daily loss** | realized + unrealized P&L since UTC 00:00 ≤ −3% of starting balance | Reject all `signal.execute` for the rest of the UTC day. State persisted in Valkey. Manual reset requires explicit API flag. |
| **Max drawdown** | equity ≤ peak × 0.85 (15% DD from any high) | Full halt. Requires human review + explicit reset. |
| **Dead-man switch** | no `system.heartbeat` for 60s | Stop accepting signals + freeze positions. (Live mode would cancel all orders — N/A in paper.) |

### 11.2 Per-signal guards

Reject signal if any of:
- More than 5 concurrent open positions
- `size_usd` > 10% of **current** account equity (not starting balance)
- Leverage > 5x
- Symbol not in configured list
- SL or TP missing or on the wrong side of entry

Note: per-position sizing scales with current equity; the daily-loss circuit breaker scales off **starting** balance so the kill threshold doesn't drift during the day.

`GET /circuit-breaker/status` exposes live state of all breakers and guards.

---

## 12. Error handling — explicit policies

| Failure | Policy |
|---|---|
| Binance WS disconnect | Auto-reconnect with exponential backoff (1s → 60s cap). Heartbeat every 30s. On reconnect, REST gap-fill missed klines. |
| Binance REST 429 / `-1003` | Respect `Retry-After`. Backoff. All REST flows through one rate-limit-aware client. |
| Binance REST 5xx | Retry 3× with jitter, then bubble up via `bus.error`. |
| TimescaleDB / PG lost | Buffer writes in a Valkey list (cap 100k); drain on reconnect; alert when buffer breaches threshold. |
| Valkey lost | Bus is gone — process exits non-zero. docker-compose restarts. Log loudly. |
| Pydantic validation fail on bus | Drop message, emit `bus.error`, never crash subscriber. |
| Paper engine internal error | Catch, log full state, trip circuit, require human review. **No auto-recovery on positions.** |
| Process crash | docker-compose `restart: unless-stopped`. On boot, rehydrate from **PG** (positions + breaker state). |

### Crash-recovery contract

**PG is truth, bus is transient.** Any state that survives restart must be persisted to PG before being published. After a restart, the system rebuilds its world from PG alone.

---

## 13. Testing

### 13.1 Three layers

**Unit** (every commit, fast):
- Pydantic message round-trip + schema-version compatibility
- Risk math: position sizing, isolated-margin liquidation-price formula, daily-loss accumulation
- Paper engine: fill simulation, SL/TP detection, funding accrual every 8h
- Circuit breaker state transitions

**Integration** (docker-compose stack via testcontainers):
- Recorded WS replay → storage rows match expected
- `POST /test/signal` veto path → audit row exists, no position
- `POST /test/signal` allowed path → position opens → simulated mark crosses SL → position closes → realized P&L recorded → RL tuple written
- 5 forced losses → daily-loss breaker trips → next signal vetoed

**Replay** (deterministic regression):
- 2-week recorded Binance market data slice + deterministic synthetic signals
- Same input → same trades → same P&L → same DB state
- Catches time-zone, off-by-one, non-determinism bugs

### 13.2 Coverage targets

- **80%** on `risk/` and `papertrade/` (these modules cost real money later)
- **60%** elsewhere is fine for Foundation
- CI: GitHub Actions running `make test` (lint + unit + integration) on PRs

---

## 14. Acceptance criteria — "Foundation is done" when:

1. `docker-compose up` starts valkey + timescale + postgres + backend cleanly.
2. Within 30s, live klines for all 10 USDM perpetuals flow into TimescaleDB.
3. `GET /symbols` returns the 10 symbols with current mark prices.
4. `POST /test/signal` opens a paper position; SL / TP / liquidation close it automatically.
5. 5 forced paper losses totaling −3% trips the daily-loss breaker; next signal is vetoed.
6. Kill backend → restart → all open positions and breaker state recover from PG.
7. Every signal (allowed or vetoed) has a trade audit row; every executed trade has an RL tuple.
8. `pytest` passes including the 2-day replay test.

---

## 15. Out of scope (explicit non-goals for Foundation)

HashiCorp Vault · AWS deployment · Telegram / Discord alerts · Real agents (Director, Quant, Risk, Sentiment, Portfolio, Execution) · ML or RL models · Backtester engine · Dashboard UI · Prometheus / Grafana / Loki · Live trading mode · Full order-book depth ladder · Historical backfill beyond bootstrap · News scraping pipeline (ScrapeGraphAI + Ollama/Qwen) · Multi-region deployment.

---

## 16. Open questions / deferred to later phases

- **RL algorithm** (PPO vs DQN vs differential-Sharpe-direct) — Phase 3 design.
- **RL reward function shape** (raw P&L vs risk-adjusted vs CVaR-aware) — Phase 3 design.
- **LLM agent prompt structure** for the 6 Phase-2 agents — Phase 2 design.
- **Sentiment Agent details** — ScrapeGraphAI source list, Ollama/Qwen model choice (Qwen 2.5 7B vs 14B vs 32B), prompt template — Phase 2 design.
- **Backtester** (vectorbt vs custom replay through paper engine) — Phase 3 design.
- **Multi-region / low-latency Binance hosting** — Phase 5.
- **Funding-rate impact on RL reward** — Phase 3.
- **Hot-reload of `config.yaml`** — deferred (restart suffices for v1).

---

## 17. References

- Source blueprint: `~/Downloads/cryptoswarm_blueprint.html`
- AutoHedge: https://github.com/The-Swarm-Corporation/AutoHedge
- Swarms framework: https://github.com/kyegomez/swarms
- Binance USDM Futures API: https://developers.binance.com/docs/derivatives/usds-margined-futures
- Valkey: https://valkey.io/
- TimescaleDB: https://docs.timescale.com/
- ScrapeGraphAI (Phase 2): https://github.com/ScrapeGraphAI/Scrapegraph-ai
- Ollama (Phase 2): https://ollama.com/
- Qwen (Phase 2): https://qwenlm.github.io/
