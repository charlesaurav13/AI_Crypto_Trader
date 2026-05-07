"""Sentiment Agent — fetches the Fear & Greed Index and maps it to a [-1, 1] score.

Primary source: alternative.me/fng API (no key required).
If the API is unreachable, returns a neutral score (0.0) with source="neutral_fallback".
"""
from __future__ import annotations

import logging

import httpx

from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import AnalyzeRequest, SentimentResult

logger = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/?limit=1"


def _fng_to_score(value: int) -> float:
    """Map Fear & Greed 0–100 to sentiment score -1.0–1.0.

    0   = Extreme Fear  → -1.0
    50  = Neutral       →  0.0
    100 = Extreme Greed → +1.0
    """
    return round((value - 50) / 50.0, 4)


class SentimentAgent:
    def __init__(self, bus: BusClient, timeout_s: float = 5.0) -> None:
        self._bus = bus
        self._timeout = timeout_s

    async def run(self) -> None:
        async for topic, data in self._bus.psubscribe("agent.analyze.*"):
            req = AnalyzeRequest.model_validate_json(data)
            try:
                await self._handle(req)
            except Exception as exc:
                logger.error("SentimentAgent error for %s: %s", req.symbol, exc)

    async def _handle(self, req: AnalyzeRequest) -> None:
        score, source, summary = await self._fetch_sentiment()
        result = SentimentResult(
            symbol=req.symbol,
            correlation_id=req.correlation_id,
            score=score,
            source=source,
            summary=summary,
        )
        await self._bus.publish(f"agent.result.sentiment.{req.symbol}", result)
        logger.info("SentimentAgent: %s score=%.2f source=%s", req.symbol, score, source)

    async def _fetch_sentiment(self) -> tuple[float, str, str]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_FNG_URL)
                resp.raise_for_status()
                data = resp.json()["data"][0]
                value = int(data["value"])
                label = data.get("value_classification", "Unknown")
                return _fng_to_score(value), "fear_greed_api", f"{label} ({value}/100)"
        except Exception as exc:
            logger.warning("SentimentAgent: Fear & Greed API unavailable: %s", exc)
            return 0.0, "neutral_fallback", "API unavailable — neutral assumed"
