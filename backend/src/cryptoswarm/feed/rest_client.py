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
