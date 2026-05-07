from __future__ import annotations
import asyncio
from typing import AsyncIterator
import valkey.asyncio as valkey_aio
from .messages import BaseMsg


class BusClient:
    """Thin Valkey pub/sub wrapper. All inter-module communication goes through here."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: valkey_aio.Valkey | None = None

    async def connect(self) -> None:
        self._client = await valkey_aio.from_url(self._url, decode_responses=True)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def publish(self, topic: str, msg: BaseMsg) -> None:
        assert self._client, "BusClient not connected"
        await self._client.publish(topic, msg.model_dump_json())

    async def subscribe(self, *topics: str) -> AsyncIterator[tuple[str, str]]:
        """Exact-match subscribe. Yields (topic, raw_json) pairs."""
        assert self._client, "BusClient not connected"
        ps = self._client.pubsub()
        await ps.subscribe(*topics)
        try:
            async for message in ps.listen():
                if message["type"] == "message":
                    yield message["channel"], message["data"]
        finally:
            await ps.aclose()

    async def psubscribe(self, *patterns: str) -> AsyncIterator[tuple[str, str]]:
        """Pattern subscribe (e.g. 'market.*'). Yields (channel, raw_json) pairs."""
        assert self._client, "BusClient not connected"
        ps = self._client.pubsub()
        await ps.psubscribe(*patterns)
        try:
            async for message in ps.listen():
                if message["type"] == "pmessage":
                    yield message["channel"], message["data"]
        finally:
            await ps.aclose()
