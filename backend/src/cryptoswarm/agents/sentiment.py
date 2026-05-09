"""Sentiment Agent — combines Fear & Greed Index with scraped news sentiment."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import AnalyzeRequest, SentimentResult

if TYPE_CHECKING:
    from cryptoswarm.storage.postgres import PostgresWriter

logger = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/?limit=1"


def _fng_to_score(value: int) -> float:
    return round((value - 50) / 50.0, 4)


class SentimentAgent:
    def __init__(
        self,
        bus: BusClient,
        pg: "PostgresWriter | None" = None,
        timeout_s: float = 5.0,
        news_lookback_hours: int = 6,
    ) -> None:
        self._bus = bus
        self._pg = pg
        self._timeout = timeout_s
        self._news_hours = news_lookback_hours

    async def run(self) -> None:
        async for topic, data in self._bus.psubscribe("agent.analyze.*"):
            req = AnalyzeRequest.model_validate_json(data)
            try:
                await self._handle(req)
            except Exception as exc:
                logger.error("SentimentAgent error for %s: %s", req.symbol, exc)

    async def _handle(self, req: AnalyzeRequest) -> None:
        fng_score, fng_source, fng_summary = await self._fetch_fng()
        news_score, news_count = await self._fetch_news_score(req.symbol)

        if news_count > 0:
            combined = round(0.5 * fng_score + 0.5 * news_score, 4)
            source = "combined"
            summary = f"FNG: {fng_summary} | News ({news_count} articles): {news_score:+.2f}"
        else:
            combined = fng_score
            source = fng_source
            summary = fng_summary

        result = SentimentResult(
            symbol=req.symbol,
            correlation_id=req.correlation_id,
            score=combined,
            source=source,
            summary=summary,
        )
        await self._bus.publish(f"agent.result.sentiment.{req.symbol}", result)
        logger.info("SentimentAgent: %s score=%.2f source=%s", req.symbol, combined, source)

    async def _fetch_fng(self) -> tuple[float, str, str]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_FNG_URL)
                resp.raise_for_status()
                data = resp.json()["data"][0]
                value = int(data["value"])
                label = data.get("value_classification", "Unknown")
                return _fng_to_score(value), "fear_greed_api", f"{label} ({value}/100)"
        except Exception as exc:
            logger.warning("SentimentAgent: FNG API unavailable: %s", exc)
            return 0.0, "neutral_fallback", "API unavailable"

    async def _fetch_news_score(self, symbol: str) -> tuple[float, int]:
        """Return (weighted_avg_sentiment, article_count). Falls back to (0.0, 0) on error."""
        if self._pg is None:
            return 0.0, 0
        try:
            rows = await self._pg.get_news_sentiment_for_symbol(symbol, self._news_hours)
            if not rows:
                return 0.0, 0
            total_weight = sum(float(r["relevance"]) for r in rows)
            if total_weight == 0:
                return 0.0, 0
            weighted = sum(float(r["score"]) * float(r["relevance"]) for r in rows)
            return round(weighted / total_weight, 4), len(rows)
        except Exception as exc:
            logger.warning("SentimentAgent: news fetch error for %s: %s", symbol, exc)
            return 0.0, 0
