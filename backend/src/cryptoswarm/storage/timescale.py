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

    async def fetch_klines(self, symbol: str, limit: int = 100) -> list[dict]:
        """Return the most recent `limit` 1m candles for symbol, chronological order."""
        assert self._pool, "TimescaleWriter not connected"
        rows = await self._pool.fetch(
            """
            SELECT ts, open, high, low, close, volume
            FROM klines_1m
            WHERE symbol = $1
            ORDER BY ts DESC
            LIMIT $2
            """,
            symbol,
            limit,
        )
        # rows arrive newest-first; reverse to oldest-first for indicators
        return [
            {
                "open":   float(r["open"]),
                "high":   float(r["high"]),
                "low":    float(r["low"]),
                "close":  float(r["close"]),
                "volume": float(r["volume"]),
            }
            for r in reversed(rows)
        ]
