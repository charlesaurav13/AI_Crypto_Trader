"""NewsWriter — persists scraped articles and publishes high-relevance sentiment to bus."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cryptoswarm.bus.messages import NewsSentimentResult
from cryptoswarm.scraper.scorer import ScoredArticle

if TYPE_CHECKING:
    from cryptoswarm.bus.client import BusClient
    from cryptoswarm.storage.postgres import PostgresWriter

logger = logging.getLogger(__name__)


class NewsWriter:
    def __init__(
        self,
        pg: "PostgresWriter",
        bus: "BusClient",
        min_relevance: float = 0.3,
    ) -> None:
        self._pg = pg
        self._bus = bus
        self._min_relevance = min_relevance

    async def write(
        self,
        source: str,
        url: str,
        title: str | None,
        body: str | None,
        scores: list[ScoredArticle],
    ) -> None:
        """Insert article + all sentiment scores; publish to bus for high-relevance items."""
        news_id = await self._pg.insert_news_item(
            source=source, url=url, title=title, body=body
        )

        for scored in scores:
            await self._pg.insert_news_sentiment(
                news_item_id=news_id,
                symbol=scored.symbol,
                model="qwen2.5:7b",
                relevance=scored.relevance,
                score=scored.sentiment,
                summary=scored.summary,
            )

        # Publish to bus only for symbols with enough relevance
        for scored in scores:
            if scored.relevance >= self._min_relevance:
                msg = NewsSentimentResult(
                    symbol=scored.symbol,
                    score=scored.sentiment,
                    article_count=1,
                    top_headline=title or "",
                    source_breakdown={source: 1},
                )
                await self._bus.publish(f"news.sentiment.{scored.symbol}", msg)
                logger.debug(
                    "NewsWriter: published %s relevance=%.2f sentiment=%.2f",
                    scored.symbol, scored.relevance, scored.sentiment,
                )
