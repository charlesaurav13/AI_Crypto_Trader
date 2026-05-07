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
