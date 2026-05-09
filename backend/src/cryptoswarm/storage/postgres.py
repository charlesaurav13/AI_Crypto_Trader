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

    # ------------------------------------------------------------------
    # News pipeline
    # ------------------------------------------------------------------

    async def insert_news_item(
        self, source: str, url: str, title: str | None, body: str | None
    ) -> int:
        """Insert a news article. Returns the new row id (or existing id on conflict)."""
        assert self._pool
        return await self._pool.fetchval(
            """
            INSERT INTO news_items (source, url, title, body)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (url) DO UPDATE SET title=EXCLUDED.title
            RETURNING id
            """,
            source, url, title, body,
        )

    async def insert_news_sentiment(
        self,
        news_item_id: int,
        symbol: str,
        model: str,
        relevance: float,
        score: float,
        summary: str | None,
    ) -> None:
        assert self._pool
        await self._pool.execute(
            """
            INSERT INTO news_sentiment
              (news_item_id, symbol, model, relevance, score, summary)
            VALUES ($1,$2,$3,$4,$5,$6)
            """,
            news_item_id, symbol, model, relevance, score, summary,
        )

    async def get_news_sentiment_for_symbol(
        self, symbol: str, hours: int = 6
    ) -> list:
        assert self._pool
        return await self._pool.fetch(
            """
            SELECT score, relevance, summary, ts
            FROM news_sentiment
            WHERE symbol=$1
              AND ts >= now() - ($2 || ' hours')::interval
            ORDER BY ts DESC
            """,
            symbol, str(hours),
        )

    # ------------------------------------------------------------------
    # Agent prompts
    # ------------------------------------------------------------------

    async def get_agent_prompt(self, agent_name: str) -> str | None:
        """Return active system_prompt for an agent, or None if not set."""
        assert self._pool
        row = await self._pool.fetchrow(
            """
            SELECT system_prompt FROM agent_prompts
            WHERE agent_name=$1 AND active=true
            ORDER BY version DESC LIMIT 1
            """,
            agent_name,
        )
        return row["system_prompt"] if row else None

    async def save_agent_prompt(
        self, agent_name: str, system_prompt: str, perf_score: float | None = None
    ) -> None:
        assert self._pool
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE agent_prompts SET active=false WHERE agent_name=$1",
                    agent_name,
                )
                await conn.execute(
                    """
                    INSERT INTO agent_prompts (agent_name, version, system_prompt, perf_score, active)
                    VALUES (
                        $1,
                        COALESCE((SELECT MAX(version) FROM agent_prompts WHERE agent_name=$1), 0) + 1,
                        $2, $3, true
                    )
                    """,
                    agent_name, system_prompt, perf_score,
                )

    # ------------------------------------------------------------------
    # RL tuple reward update
    # ------------------------------------------------------------------

    async def update_rl_tuple_reward(
        self, correlation_id: str, reward: float, next_state: dict
    ) -> None:
        assert self._pool
        await self._pool.execute(
            """
            UPDATE rl_tuples SET reward=$2, next_state=$3
            WHERE action->>'correlation_id'=$1
              AND reward IS NULL
            """,
            correlation_id, reward, json.dumps(next_state),
        )

    # ------------------------------------------------------------------
    # ML signals
    # ------------------------------------------------------------------

    async def insert_ml_signal(
        self,
        symbol: str,
        regime_pred: str,
        direction_pred: str,
        short_direction: str,
        confidence: float,
        size_adjustment: str,
        model_version: str,
    ) -> None:
        assert self._pool
        await self._pool.execute(
            """
            INSERT INTO ml_signals
              (symbol, regime_pred, direction_pred, short_direction,
               confidence, size_adjustment, model_version)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            """,
            symbol, regime_pred, direction_pred, short_direction,
            confidence, size_adjustment, model_version,
        )

    # ------------------------------------------------------------------
    # Training runs
    # ------------------------------------------------------------------

    async def insert_training_run(
        self, model_type: str, started_at: "datetime"
    ) -> int:
        assert self._pool
        return await self._pool.fetchval(
            """
            INSERT INTO training_runs (model_type, started_at)
            VALUES ($1,$2) RETURNING id
            """,
            model_type, started_at,
        )

    async def update_training_run(
        self,
        run_id: int,
        completed_at: "datetime",
        sample_count: int,
        metrics: dict,
    ) -> None:
        assert self._pool
        await self._pool.execute(
            """
            UPDATE training_runs
            SET completed_at=$2, sample_count=$3, metrics=$4
            WHERE id=$1
            """,
            run_id, completed_at, sample_count, json.dumps(metrics),
        )

    async def get_recent_closed_trades(self, limit: int = 50) -> list:
        """Fetch recent closed trades with their rl_tuple rewards for prompt evolution."""
        assert self._pool
        return await self._pool.fetch(
            """
            SELECT t.correlation_id, t.symbol, t.side, t.realized_pnl,
                   t.opened_ts, t.closed_ts, t.exit_reason,
                   r.reward, r.state, r.action
            FROM trades t
            LEFT JOIN rl_tuples r ON r.action->>'correlation_id' = t.correlation_id
            WHERE t.closed_ts IS NOT NULL
              AND r.reward IS NOT NULL
            ORDER BY t.closed_ts DESC
            LIMIT $1
            """,
            limit,
        )
