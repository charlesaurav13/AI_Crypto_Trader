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

            logger.info(
                "Connecting to %d streams for %d symbols",
                len(streams), len(self._cfg.symbol_list),
            )
            async with bm.futures_multiplex_socket(streams) as ws:
                async for msg in ws:
                    await self._handler.handle(msg)
        finally:
            await client.close_connection()
