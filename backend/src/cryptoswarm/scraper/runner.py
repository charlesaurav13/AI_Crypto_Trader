"""ScraperRunner — scrapes all news sources every 30 minutes."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cryptoswarm.scraper.scorer import OllamaScorer
from cryptoswarm.scraper.sources import SOURCES
from cryptoswarm.scraper.writer import NewsWriter

if TYPE_CHECKING:
    from cryptoswarm.bus.client import BusClient
    from cryptoswarm.config.settings import Settings
    from cryptoswarm.storage.postgres import PostgresWriter

logger = logging.getLogger(__name__)


class ScraperRunner:
    def __init__(
        self,
        pg: "PostgresWriter",
        bus: "BusClient",
        settings: "Settings",
    ) -> None:
        self._pg = pg
        self._bus = bus
        self._cfg = settings
        self._scorer = OllamaScorer(
            ollama_url=settings.scraper_ollama_url,
            model=settings.scraper_ollama_model,
            symbols=settings.symbol_list,
        )
        self._writer = NewsWriter(
            pg=pg,
            bus=bus,
            min_relevance=settings.scraper_min_relevance,
            model=settings.scraper_ollama_model,
        )

    async def run(self) -> None:
        """Run forever: scrape all sources, sleep, repeat."""
        while True:
            await self._scrape_all()
            await asyncio.sleep(self._cfg.scraper_interval_s)

    async def _scrape_all(self) -> None:
        logger.info("ScraperRunner: starting scrape cycle (%d sources)", len(SOURCES))
        tasks = [self._scrape_source(src) for src in SOURCES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for src, result in zip(SOURCES, results):
            if isinstance(result, Exception):
                logger.warning("ScraperRunner: source=%s error=%s", src.name, result)
        errors = sum(1 for r in results if isinstance(r, Exception))
        logger.info(
            "ScraperRunner: cycle complete — %d sources, %d errors",
            len(SOURCES), errors,
        )

    async def _scrape_source(self, source) -> None:
        articles = await self._fetch_articles(source)
        for article in articles:
            scores = await self._scorer.score(
                title=article.get("title", ""),
                body=article.get("body", ""),
            )
            await self._writer.write(
                source=source.name,
                url=article.get("url", f"unknown-{source.name}"),
                title=article.get("title"),
                body=article.get("body"),
                scores=scores,
            )

    async def _fetch_articles(self, source) -> list[dict]:
        """Use ScrapeGraphAI for article sources; raw JSON fetch for Reddit."""
        if source.kind == "reddit":
            return await self._fetch_reddit(source)
        return await self._fetch_with_scrapegraph(source)

    async def _fetch_reddit(self, source) -> list[dict]:
        import httpx
        async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": "CryptoSwarm/1.0"}) as client:
            resp = await client.get(source.url)
            resp.raise_for_status()
            posts = resp.json()["data"]["children"]
            return [
                {
                    "title": p["data"]["title"],
                    "url": f"https://reddit.com{p['data']['permalink']}",
                    "body": p["data"].get("selftext", "")[:500],
                }
                for p in posts
            ]

    async def _fetch_with_scrapegraph(self, source) -> list[dict]:
        from scrapegraphai.graphs import SmartScraperGraph
        config = {
            "llm": {
                "base_url": self._cfg.scraper_ollama_url,
                "model": f"ollama/{self._cfg.scraper_ollama_model}",
            },
            "embeddings": {
                "base_url": self._cfg.scraper_ollama_url,
                "model": "ollama/nomic-embed-text",
            },
            "verbose": False,
            "headless": True,
        }
        loop = asyncio.get_running_loop()
        # ScrapeGraphAI is sync — run in thread pool
        result = await loop.run_in_executor(
            None,
            lambda: SmartScraperGraph(
                prompt=source.extraction_prompt,
                source=source.url,
                config=config,
            ).run(),
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "articles" in result:
            return result["articles"]
        return []
