"""Tests for enhanced SentimentAgent combining FNG + news sentiment."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cryptoswarm.agents.sentiment import SentimentAgent
from cryptoswarm.bus.messages import AnalyzeRequest


def _make_agent(recent_news: list[dict] | None = None):
    mock_bus = MagicMock()
    mock_bus.psubscribe = AsyncMock()
    mock_bus.subscribe = AsyncMock()
    mock_bus.publish = AsyncMock()
    mock_pg = MagicMock()
    mock_pg.get_news_sentiment_for_symbol = AsyncMock(return_value=recent_news or [])
    return SentimentAgent(bus=mock_bus, pg=mock_pg)


async def test_combines_fng_and_news_when_news_available():
    agent = _make_agent(recent_news=[
        {"score": 0.8, "relevance": 0.9, "summary": "BTC bullish"},
        {"score": 0.6, "relevance": 0.7, "summary": "BTC up"},
    ])
    req = AnalyzeRequest(symbol="BTCUSDT")
    with patch.object(agent, "_fetch_fng", AsyncMock(return_value=(0.2, "fear_greed_api", "Greed (60/100)"))):
        await agent._handle(req)
    call = agent._bus.publish.call_args
    topic = call[0][0]
    result = call[0][1]
    assert topic == "agent.result.sentiment.BTCUSDT"
    # Combined score: 0.5 * 0.2 + 0.5 * avg_news
    assert result.source == "combined"
    assert -1.0 <= result.score <= 1.0


async def test_falls_back_to_fng_when_no_news():
    agent = _make_agent(recent_news=[])
    req = AnalyzeRequest(symbol="BTCUSDT")
    with patch.object(agent, "_fetch_fng", AsyncMock(return_value=(0.3, "fear_greed_api", "Neutral (65/100)"))):
        await agent._handle(req)
    result = agent._bus.publish.call_args[0][1]
    assert result.source == "fear_greed_api"
    assert result.score == 0.3


async def test_handle_pg_error_falls_back_to_fng():
    agent = _make_agent()
    agent._pg.get_news_sentiment_for_symbol = AsyncMock(side_effect=Exception("DB down"))
    req = AnalyzeRequest(symbol="BTCUSDT")
    with patch.object(agent, "_fetch_fng", AsyncMock(return_value=(0.1, "fear_greed_api", "Fear (45/100)"))):
        await agent._handle(req)
    result = agent._bus.publish.call_args[0][1]
    assert result.source == "fear_greed_api"
