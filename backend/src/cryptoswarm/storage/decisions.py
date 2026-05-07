"""Persists DirectorDecision records to the `decisions` PostgreSQL table.

The `decisions` table schema (already created in infra/migrations/postgres/001_init.sql):
    id BIGSERIAL PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    input_state JSONB,
    output JSONB,
    reasoning TEXT,
    confidence DOUBLE PRECISION,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
"""
from __future__ import annotations

import asyncpg
import json
import logging

from cryptoswarm.bus.messages import DirectorDecision

logger = logging.getLogger(__name__)


class DecisionWriter:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=3)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def insert(self, decision: DirectorDecision) -> None:
        assert self._pool, "DecisionWriter not connected"
        input_state = json.dumps({
            "symbol":            decision.symbol,
            "quant_summary":     decision.quant_summary,
            "risk_summary":      decision.risk_summary,
            "sentiment_summary": decision.sentiment_summary,
            "portfolio_summary": decision.portfolio_summary,
        })
        output = json.dumps({
            "action":      decision.action,
            "side":        decision.side,
            "size_pct":    decision.size_pct,
            "sl_pct":      decision.sl_pct,
            "tp_pct":      decision.tp_pct,
            "entry_price": decision.entry_price,
        })
        await self._pool.execute(
            """
            INSERT INTO decisions (correlation_id, agent_name, input_state, output, reasoning, confidence)
            VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6)
            """,
            decision.correlation_id,
            "director",
            input_state,
            output,
            decision.reasoning,
            decision.confidence,
        )
        logger.debug(
            "DecisionWriter: saved decision %s action=%s",
            decision.correlation_id, decision.action,
        )
