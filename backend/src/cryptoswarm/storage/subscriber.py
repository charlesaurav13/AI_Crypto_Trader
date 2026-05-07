"""
Subscribes to all bus topics and routes to TimescaleDB / PostgreSQL writers.
This is the only module that writes to the databases at runtime.
"""
from __future__ import annotations
import asyncio
import logging

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
        async for topic, data in self._bus.psubscribe("trade.*"):
            try:
                if topic == "trade.executed":
                    msg = TradeExecuted.model_validate_json(data)
                    await self._pg.insert_trade_open(
                        correlation_id=msg.original_correlation_id,
                        symbol=msg.symbol, side=msg.side,
                        qty=msg.qty, entry_price=msg.entry_price,
                        leverage=msg.leverage, sl=msg.sl, tp=msg.tp,
                        fees=msg.fees,
                        entry_state={},   # Phase 2 populates full market snapshot
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
