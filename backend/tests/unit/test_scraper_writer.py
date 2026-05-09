"""Tests for NewsWriter — DB and bus calls mocked."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from cryptoswarm.scraper.writer import NewsWriter
from cryptoswarm.scraper.scorer import ScoredArticle


def _make_writer():
    mock_pg = MagicMock()
    mock_pg.insert_news_item = AsyncMock(return_value=1)
    mock_pg.insert_news_sentiment = AsyncMock()
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    return NewsWriter(pg=mock_pg, bus=mock_bus, min_relevance=0.3)


async def test_writer_stores_article_and_scores():
    w = _make_writer()
    scores = [
        ScoredArticle(symbol="BTCUSDT", relevance=0.9, sentiment=0.7, summary="Bullish BTC"),
        ScoredArticle(symbol="ETHUSDT", relevance=0.1, sentiment=0.0, summary="Unrelated"),
    ]
    await w.write(
        source="coindesk",
        url="https://coindesk.com/article/1",
        title="BTC up",
        body="Bitcoin rose...",
        scores=scores,
    )
    w._pg.insert_news_item.assert_called_once()
    # Both sentiments inserted regardless of relevance threshold
    assert w._pg.insert_news_sentiment.call_count == 2


async def test_writer_publishes_only_above_min_relevance():
    w = _make_writer()
    scores = [
        ScoredArticle(symbol="BTCUSDT", relevance=0.9, sentiment=0.7, summary="Bullish"),
        ScoredArticle(symbol="ETHUSDT", relevance=0.1, sentiment=0.0, summary="Skip"),
    ]
    await w.write("coindesk", "https://example.com/1", "title", "body", scores)
    # Only BTCUSDT (0.9 >= 0.3) should be published to bus
    published_topics = [c[0][0] for c in w._bus.publish.call_args_list]
    assert "news.sentiment.BTCUSDT" in published_topics
    assert "news.sentiment.ETHUSDT" not in published_topics


async def test_writer_skips_duplicate_url_gracefully():
    """Second write with same URL returns existing id — writer must not raise."""
    w = _make_writer()
    scores = [ScoredArticle(symbol="BTCUSDT", relevance=0.9, sentiment=0.5, summary="ok")]
    url = "https://example.com/dup"
    # Both calls return the same id (ON CONFLICT DO UPDATE RETURNING id)
    w._pg.insert_news_item = AsyncMock(return_value=5)
    await w.write("coindesk", url, "t", "b", scores)
    await w.write("coindesk", url, "t updated", "b updated", scores)
    # insert_news_item called twice; idempotent upsert handles duplicate silently
    assert w._pg.insert_news_item.call_count == 2
    # Sentiments written twice (once per write call)
    assert w._pg.insert_news_sentiment.call_count == 2
