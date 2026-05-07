# Phase 3: Self-Improving ML/RL Agents Design Spec

**Date:** 2026-05-08
**Status:** Approved

---

## 1. Goal

Make CryptoSwarm self-improving. After every paper trade closes, the system learns from the outcome — updating an RL policy, retraining ML models, and evolving agent prompts — so that each trading cycle is better informed than the last. The system targets aggressive but profitable trades: it hunts large upside while being ruthlessly penalized for drawdowns and losing streaks.

---

## 2. Architecture Overview

Two independent learning loops run in parallel, both feeding the DirectorAgent:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOOP C — ML / RL                             │
│                                                                      │
│  klines (TimescaleDB) ──→ FeatureEngine ──→ XGBoostModel            │
│  news_sentiment (PG)  ──→       │          LSTMModel  (batch 6h)   │
│                                 ↓                                   │
│  rl_tuples (PG) ──→ PPOPolicy (online update per trade close)       │
│                                 │                                   │
│                         MLAgent publishes MLSignal to bus           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ↓
          ┌────────────────────────────────────────┐
          │           DirectorAgent                │
          │  5 signals: Quant + Risk + Sentiment   │
          │           + Portfolio + ML             │
          │  system prompts loaded from DB         │
          └────────────┬───────────────────────────┘
                       │ Signal
                       ↓
          ┌────────────────────────────────────────┐
          │         PaperTradeEngine               │
          │  trade close → RewardComputer          │
          │  → rl_tuples updated (reward +         │
          │    next_state) → PPO online update     │
          └────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────┐
│                      LOOP B — PROMPT EVOLUTION                       │
│                                                                      │
│  Every 6h: PromptEvolutionEngine fetches last N closed trades       │
│  → scores each agent's contribution using composite reward          │
│  → Teacher LLM (strongest available) critiques worst agent          │
│  → generates improved system_prompt                                 │
│  → saves new version to agent_prompts table                         │
│  → agents pick up new prompt on next cycle                          │
│                                                                      │
│  ScrapeGraphAI (every 30 min):                                       │
│  CoinDesk + CoinTelegraph + Decrypt + The Block + CryptoSlate       │
│  + Reddit (r/CryptoMarkets, r/Bitcoin, r/ethtrader, r/altcoin)      │
│  + Bloomberg/Reuters crypto sections + Twitter/X (via nitter)       │
│  → Ollama/Qwen 2.5 7B scores relevance per symbol (0.0–1.0)        │
│  → stored in news_items + news_sentiment                            │
│  → published to bus: news.sentiment.{symbol}                        │
│  → SentimentAgent combines with Fear & Greed API                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Composite Reward Function

Used by both loops — Loop C for RL training, Loop B for prompt critique scoring.

```
reward = w1 * normalized_pnl
       + w2 * win_rate_contribution
       + w3 * rr_ratio
       - w4 * drawdown_penalty
       - w5 * time_in_loss_penalty

Where:
  normalized_pnl        = realized_pnl / position_size_usd          (range: ~-1 to +1)
  win_rate_contribution = +1.0 if trade profitable else -0.5
  rr_ratio              = abs(tp_pct / sl_pct)  capped at 5.0, normalized to [0,1]
  drawdown_penalty      = max(0, -realized_pnl / balance) * 3.0     (extra weight on losses)
  time_in_loss_penalty  = seconds_in_negative_pnl / total_duration  (0.0–1.0)

Default weights: w1=0.4, w2=0.2, w3=0.2, w4=0.15, w5=0.05
Weights stored in settings and tunable via .env: REWARD_W1..W5
```

Reward is computed by `RewardComputer` on every trade close event and written back into `rl_tuples.reward` and `rl_tuples.next_state`.

---

## 4. New Modules

### 4.1 `cryptoswarm/scraper/`

```
scraper/
  __init__.py
  runner.py          # ScraperRunner — schedules all scrapers, runs every 30min
  sources.py         # source definitions: URL, type, scrape config
  scorer.py          # OllamaScorer — calls Qwen 2.5 7B to score relevance per symbol
  writer.py          # NewsWriter — inserts to news_items + news_sentiment, publishes to bus
```

**ScraperRunner** uses ScrapeGraphAI's `SmartScraperGraph` (LLM-powered extraction) for article-based sources and `SearchGraph` for Reddit/social sources. For Twitter/X it uses nitter instances (public, no API key). If a source is unreachable it is skipped silently with a warning log — scraping is best-effort.

**OllamaScorer** sends each scraped article to a local Ollama Qwen 2.5 7B model with the prompt: `"Rate the relevance of this text to {symbol} crypto trading on a scale 0.0 to 1.0. Return JSON: {relevance: float, sentiment: float (-1 to 1), summary: string (max 50 words)}"`. Articles scoring < 0.3 relevance are stored but not published to bus.

**Bus topic published:** `news.sentiment.{symbol}` — payload is `NewsSentimentResult` message.

### 4.2 `cryptoswarm/ml/`

```
ml/
  __init__.py
  features.py        # FeatureEngine — builds feature vectors from klines + news_sentiment
  xgboost_model.py   # XGBoostModel — regime classification + 1h direction prediction
  lstm_model.py      # LSTMModel — 30-bar sequence price direction prediction
  ppo_policy.py      # PPOPolicy — Stable-Baselines3 PPO, online updates
  reward.py          # RewardComputer — composite reward, writes back to rl_tuples
  trainer.py         # MLTrainer — orchestrates batch retraining every 6h
  model_store.py     # ModelStore — save/load model artifacts to disk
```

**FeatureEngine** produces a fixed-size feature vector per symbol per cycle:

| Feature Group | Features |
|---|---|
| Price action | RSI-14, MACD signal, BB %B, EMA20/50 cross, ATR-14 normalized |
| Volume | OBV delta, volume ratio (current / 20-bar avg) |
| Trend | ADX-14, slope of EMA50 over last 5 bars |
| News | avg_sentiment_score (last 6h), news_count (last 6h), max_relevance |
| Market | Fear & Greed index (normalized), funding_rate, open_interest_delta |

**XGBoostModel** — two separate classifiers, both trained on the same feature vector:
- `regime_clf`: 4-class (trending_up / trending_down / ranging / volatile)
- `direction_clf`: binary (price up or down in next 1h)

Both trained on historical labeled data: labels derived from actual 1h forward returns in TimescaleDB. Batch retrain every 6h on last 30 days of klines. Minimum 500 samples required before first training run.

**LSTMModel** — single PyTorch LSTM (2 layers, hidden=64) taking last 30 1m bars as input sequence, predicting binary direction (up/down) for next 15 mins. Trained in same batch cycle. Provides a short-horizon signal XGBoost misses.

**PPOPolicy** — Stable-Baselines3 MlpPolicy PPO. State = feature vector (same as XGBoost). Action space = discrete 3 (hold / scale_up / scale_down) — modifies the Director's `size_pct` recommendation by ±20%. Reward = composite reward from `RewardComputer`. Online update called after every trade close with the `(state, action, reward, next_state)` tuple. Saved to disk after every update.

**MLTrainer** — runs in a background asyncio task. Every 6h: fetches klines + news_sentiment from DB, calls FeatureEngine, retrains XGBoost + LSTM, saves models, records metrics to `training_runs` table.

### 4.3 `cryptoswarm/learning/`

```
learning/
  __init__.py
  prompt_evolution.py   # PromptEvolutionEngine — teacher-student critique loop
  prompt_store.py       # PromptStore — reads/writes agent_prompts table
```

**PromptEvolutionEngine** runs every 6h (offset from MLTrainer by 1h):

1. Fetch last 50 closed trades from `rl_tuples` (with reward computed)
2. Join with `decisions` table to get which agent's signal drove the decision
3. Compute per-agent contribution score: average reward on trades where that agent's signal was the deciding factor
4. Identify lowest-scoring agent
5. Call Teacher LLM (the globally configured strongest model — `director_llm` override, defaulting to global `llm_provider`) with:
   - Current system prompt for that agent
   - 10 worst decisions that agent contributed to (with outcome, reward, reasoning)
   - Instruction: "Rewrite this agent's system prompt to avoid these patterns and improve profitability"
6. Save new prompt version to `agent_prompts` (with `active=true`, previous version set `active=false`)
7. Log which agent was evolved and what changed

**PromptStore** is used by each agent at startup and on every 10th cycle to reload their current active prompt from DB. Falls back to hardcoded default if DB has no entry for the agent.

### 4.4 `cryptoswarm/agents/ml_agent.py`

```python
class MLAgent:
    """5th signal agent. Uses XGBoost + LSTM + PPO to produce MLSignal."""
```

Subscribes to `agent.analyze.*` exactly like other agents. On each request:
1. Calls `FeatureEngine.build(symbol)` — fetches latest klines + news from DB
2. Gets `regime` and `direction` predictions from XGBoost
3. Gets `short_direction` prediction from LSTM
4. Gets `size_adjustment` from PPO policy (hold/scale_up/scale_down)
5. Publishes `MLSignal` to `agent.result.ml.{symbol}`
6. Saves prediction to `ml_signals` table

### 4.5 `cryptoswarm/agents/reward_computer.py`

```python
class RewardComputer:
    """Subscribes to position.update (is_closed=True). Computes composite reward.
    Updates rl_tuples. Triggers PPO online update."""
```

Subscribes to `position.update` (filtered to `is_closed=True`). On each closed position:
1. Fetch trade record from Postgres (entry state, P&L, fees, timing)
2. Compute composite reward using configurable weights
3. Update `rl_tuples` row: set `reward` + `next_state` (current market feature vector)
4. Call `ppo_policy.update(state, action, reward, next_state)` — incremental online update

---

## 5. Updated Existing Modules

### 5.1 `agents/sentiment.py`
- Subscribe to `news.sentiment.*` from the scraper bus topic in addition to Fear & Greed API
- Combine: `final_score = 0.5 * fng_score + 0.5 * avg_news_score` (when news available)
- Fall back to Fear & Greed only if no news data in last 2h

### 5.2 `agents/director.py`
- Accept `MLSignal` as 5th required signal (add to `_RESULT_CLASSES` + `_REQUIRED_AGENTS`)
- On startup + every 10 cycles: reload system prompt from `PromptStore`
- Include `ml_summary` in Director's LLM prompt
- Add `ml_summary` field to `DirectorDecision` message

### 5.3 `storage/postgres.py`
- Add `insert_news_item()`, `insert_news_sentiment()` methods
- Add `get_news_sentiment_for_symbol(symbol, hours)` for FeatureEngine
- Add `insert_ml_signal()`, `insert_training_run()` methods
- Add `get_agent_prompt(agent_name)` and `save_agent_prompt()` methods
- Add `update_rl_tuple_reward(id, reward, next_state)` method

### 5.4 `config/settings.py`
- Add scraper settings: `scraper_interval_s=1800`, `scraper_ollama_model="qwen2.5:7b"`, `scraper_min_relevance=0.3`
- Add ML settings: `ml_retrain_interval_s=21600`, `ml_min_samples=500`, `ml_model_dir="models/"`
- Add reward weights: `reward_w1=0.4`, `reward_w2=0.2`, `reward_w3=0.2`, `reward_w4=0.15`, `reward_w5=0.05`
- Add prompt evolution: `prompt_evolution_interval_s=25200` (7h), `prompt_evolution_lookback=50`

### 5.5 `main.py`
- Wire `ScraperRunner`, `MLAgent`, `RewardComputer`, `PromptEvolutionEngine`, `MLTrainer`
- All run as asyncio tasks alongside existing agents

---

## 6. New Database Tables

```sql
-- news_items: raw scraped articles
CREATE TABLE IF NOT EXISTS news_items (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    url         TEXT UNIQUE NOT NULL,
    title       TEXT,
    body        TEXT,
    fetched_ts  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- news_sentiment: per-symbol relevance + sentiment scores
CREATE TABLE IF NOT EXISTS news_sentiment (
    id           BIGSERIAL PRIMARY KEY,
    news_item_id BIGINT REFERENCES news_items(id),
    symbol       TEXT NOT NULL,
    model        TEXT NOT NULL,
    relevance    FLOAT NOT NULL,
    score        FLOAT NOT NULL,    -- -1.0 to 1.0
    summary      TEXT,
    ts           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_ts ON news_sentiment(symbol, ts DESC);

-- agent_prompts: versioned evolved system prompts
CREATE TABLE IF NOT EXISTS agent_prompts (
    id           BIGSERIAL PRIMARY KEY,
    agent_name   TEXT NOT NULL,
    version      INT NOT NULL DEFAULT 1,
    system_prompt TEXT NOT NULL,
    perf_score   FLOAT,             -- composite reward score when this was created
    active       BOOLEAN NOT NULL DEFAULT true,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(agent_name, version)
);

-- ml_signals: every ML inference result
CREATE TABLE IF NOT EXISTS ml_signals (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    regime_pred     TEXT NOT NULL,
    direction_pred  TEXT NOT NULL,
    short_direction TEXT NOT NULL,
    confidence      FLOAT NOT NULL,
    size_adjustment TEXT NOT NULL,   -- hold / scale_up / scale_down
    model_version   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ml_signals_symbol_ts ON ml_signals(symbol, ts DESC);

-- training_runs: audit log of every ML retrain
CREATE TABLE IF NOT EXISTS training_runs (
    id           BIGSERIAL PRIMARY KEY,
    model_type   TEXT NOT NULL,      -- xgboost | lstm | ppo
    started_at   TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    sample_count INT,
    metrics      JSONB               -- accuracy, loss, reward_mean, etc.
);
```

---

## 7. New Bus Messages

```python
class NewsSentimentResult(BaseMsg):
    symbol: str
    score: float           # -1.0 to 1.0 (avg weighted by relevance)
    article_count: int
    top_headline: str
    source_breakdown: dict  # {"coindesk": 3, "reddit": 7, ...}

class MLSignal(BaseMsg):
    symbol: str
    regime_pred: Literal["trending_up", "trending_down", "ranging", "volatile"]
    direction_pred: Literal["up", "down"]       # XGBoost 1h prediction
    short_direction: Literal["up", "down"]      # LSTM 15m prediction
    size_adjustment: Literal["hold", "scale_up", "scale_down"]
    confidence: float                           # 0.0–1.0
    reasoning: str
```

---

## 8. New Dependencies

```toml
# pyproject.toml additions
"scrapegraphai>=1.13",          # LLM-powered web scraping
"stable-baselines3>=2.3",       # PPO reinforcement learning
"xgboost>=2.0",                 # gradient boosted trees
"torch>=2.3",                   # LSTM + SB3 backend
"scikit-learn>=1.5",            # preprocessing, metrics
"pandas>=2.2",                  # feature engineering DataFrames
"numpy>=1.26",                  # numerical ops
"apscheduler>=3.10",            # batch retraining scheduler
"playwright>=1.44",             # required by ScrapeGraphAI for JS-heavy pages
```

---

## 9. New File Tree

```
backend/src/cryptoswarm/
├── scraper/
│   ├── __init__.py
│   ├── runner.py         # ScraperRunner (asyncio task, 30min cycle)
│   ├── sources.py        # SOURCES list: url, type, symbols hint
│   ├── scorer.py         # OllamaScorer (Qwen 2.5 7B relevance + sentiment)
│   └── writer.py         # NewsWriter (DB inserts + bus publish)
├── ml/
│   ├── __init__.py
│   ├── features.py       # FeatureEngine
│   ├── xgboost_model.py  # XGBoostModel (regime + direction)
│   ├── lstm_model.py     # LSTMModel (short-horizon)
│   ├── ppo_policy.py     # PPOPolicy (Stable-Baselines3 wrapper)
│   ├── reward.py         # RewardComputer
│   ├── trainer.py        # MLTrainer (batch 6h scheduler)
│   └── model_store.py    # ModelStore (disk save/load)
├── learning/
│   ├── __init__.py
│   ├── prompt_evolution.py   # PromptEvolutionEngine (7h cycle)
│   └── prompt_store.py       # PromptStore (DB read/write)
├── agents/
│   ├── ml_agent.py       # MLAgent (5th signal, subscribes to bus)
│   └── reward_computer.py # RewardComputer agent (subscribes position.update)
tests/
├── unit/
│   ├── test_scraper.py
│   ├── test_features.py
│   ├── test_xgboost_model.py
│   ├── test_lstm_model.py
│   ├── test_ppo_policy.py
│   ├── test_reward.py
│   ├── test_ml_agent.py
│   ├── test_prompt_evolution.py
│   └── test_prompt_store.py
```

---

## 10. Data Flow — Full Cycle

```
Every 30min:
  ScraperRunner → scrapes all sources → OllamaScorer scores per symbol
  → NewsWriter → news_items + news_sentiment tables
  → publishes news.sentiment.{symbol} to bus

Every director_interval_s (60s):
  DirectorAgent → broadcasts AnalyzeRequest to 5 agents
  QuantAgent    → klines + LLM → QuantResult
  RiskAgent     → Kelly + LLM → RiskResult
  SentimentAgent → FNG + news → SentimentResult (richer with news)
  PortfolioAgent → positions + LLM → PortfolioResult
  MLAgent       → features + XGBoost + LSTM + PPO → MLSignal

  DirectorAgent collects all 5 → calls LLM with evolved prompt
  → DirectorDecision → Signal → PaperTradeEngine

On trade close:
  RewardComputer → computes composite reward
  → updates rl_tuples(reward, next_state)
  → PPOPolicy.online_update(s, a, r, s') — immediate learning

Every 6h (MLTrainer batch):
  fetch 30d klines + news → FeatureEngine → retrain XGBoost + LSTM
  → save models to disk → insert training_runs record

Every 7h (PromptEvolutionEngine):
  fetch last 50 closed trades + decisions → score per agent
  → Teacher LLM critiques worst agent → new system_prompt
  → save to agent_prompts table → agents reload next cycle
```

---

## 11. Testing Strategy

- **Unit tests (no network, no DB):** All ML models, FeatureEngine, RewardComputer, PromptEvolutionEngine tested with mocked data. XGBoostModel and LSTMModel tested with synthetic feature arrays. PPOPolicy tested with fake `(s,a,r,s')` tuples.
- **Scraper tests:** ScrapeGraphAI calls mocked. OllamaScorer mocked via `unittest.mock`. Test that writer correctly deduplicates by URL.
- **Integration tests:** ScraperRunner → NewsWriter → DB roundtrip (requires Docker stack). MLTrainer end-to-end with minimal synthetic data (50 rows). Director 5-agent roundtrip via bus.
- **No real API calls in unit tests.** All LLM calls (Teacher, OllamaScorer) patched at SDK level.

---

## 12. Deferred (Not Phase 3)

- **GPU fine-tuning** of LLM weights (requires significant compute infrastructure)
- **Live trading** (Phase 5)
- **Dashboard** showing learning curves and prompt diffs (Phase 4)
- **Multi-exchange** data for news correlation
- **Backtesting** the trained models on historical data (could be Phase 3b)
- **Model A/B testing** (running two prompt versions simultaneously and comparing)

---

## 13. Open Design Decisions

| Decision | Resolution |
|---|---|
| Twitter/X scraping | Use public nitter instances; if all fail, skip silently |
| PPO state size | Fixed 25-feature vector (defined in FeatureEngine) |
| LSTM warm-up | Requires 500 bars minimum before first training; MLAgent returns neutral signal until trained |
| Prompt evolution bootstrap | On first run with no agent_prompts entries, use hardcoded defaults |
| Model versioning | Version = ISO timestamp string, stored in training_runs + ml_signals |
| Ollama model for scraping | `qwen2.5:7b` — must be pulled locally: `ollama pull qwen2.5:7b` |
