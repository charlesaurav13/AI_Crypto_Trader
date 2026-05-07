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
        # Funding rate is embedded in the mark price stream
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
