"""Unit tests for all sub-agents (Quant, Risk, Sentiment, Portfolio).
LLMClient and BusClient are always mocked — no real API keys needed.
"""
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cryptoswarm.bus.messages import (
    AnalyzeRequest, QuantResult, RiskResult, SentimentResult, PortfolioResult, PositionUpdate,
)
from cryptoswarm.config.settings import Settings
from cryptoswarm.agents.quant import QuantAgent
from cryptoswarm.agents.risk_agent import RiskAgent
from cryptoswarm.agents.sentiment import SentimentAgent
from cryptoswarm.agents.portfolio import PortfolioAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_klines(n: int = 100) -> list[dict]:
    base = 50000.0
    return [
        {
            "open":   base + i,
            "high":   base + i + 50,
            "low":    base + i - 50,
            "close":  base + i + 10,
            "volume": 100.0,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Quant Agent
# ---------------------------------------------------------------------------

async def test_quant_agent_publishes_result():
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_ts = MagicMock()
    mock_ts.fetch_klines = AsyncMock(return_value=_fake_klines())

    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock(return_value={
        "regime": "trending_up",
        "signal_strength": 0.65,
        "confidence": 0.75,
        "reasoning": "RSI 58, EMA bullish cross",
    })

    agent = QuantAgent(bus=mock_bus, ts=mock_ts, llm=mock_llm)
    req = AnalyzeRequest(symbol="BTCUSDT")
    await agent._handle(req)

    mock_bus.publish.assert_called_once()
    topic, msg = mock_bus.publish.call_args[0]
    assert topic == "agent.result.quant.BTCUSDT"
    assert isinstance(msg, QuantResult)
    assert msg.symbol == "BTCUSDT"
    assert msg.correlation_id == req.correlation_id
    assert msg.regime == "trending_up"


async def test_quant_agent_skips_insufficient_bars():
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_ts = MagicMock()
    mock_ts.fetch_klines = AsyncMock(return_value=_fake_klines(30))  # too few

    mock_llm = MagicMock()

    agent = QuantAgent(bus=mock_bus, ts=mock_ts, llm=mock_llm)
    req = AnalyzeRequest(symbol="BTCUSDT")
    await agent._handle(req)

    mock_bus.publish.assert_not_called()
    mock_llm.ask.assert_not_called()


# ---------------------------------------------------------------------------
# Risk Agent
# ---------------------------------------------------------------------------

async def test_risk_agent_publishes_result():
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock(return_value={
        "kelly_fraction": 0.05,
        "max_loss_usdt": 30.0,
        "reasoning": "Low volatility, moderate confidence, recommend 5% kelly",
    })

    settings = Settings(paper_trading=True)
    agent = RiskAgent(bus=mock_bus, llm=mock_llm, settings=settings)
    req = AnalyzeRequest(symbol="BTCUSDT")
    await agent._handle(req)

    mock_bus.publish.assert_called_once()
    topic, msg = mock_bus.publish.call_args[0]
    assert topic == "agent.result.risk.BTCUSDT"
    assert isinstance(msg, RiskResult)
    assert msg.correlation_id == req.correlation_id
    assert 0.0 <= msg.kelly_fraction <= 1.0


async def test_risk_agent_clamps_kelly_to_max():
    """kelly_fraction > max_position_pct should be clamped."""
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock(return_value={
        "kelly_fraction": 0.99,  # unreasonably high
        "max_loss_usdt": 500.0,
        "reasoning": "test",
    })

    settings = Settings(paper_trading=True)
    agent = RiskAgent(bus=mock_bus, llm=mock_llm, settings=settings)
    req = AnalyzeRequest(symbol="ETHUSDT")
    await agent._handle(req)

    _, msg = mock_bus.publish.call_args[0]
    assert msg.kelly_fraction == settings.risk.max_position_pct


# ---------------------------------------------------------------------------
# Sentiment Agent
# ---------------------------------------------------------------------------

async def test_sentiment_agent_fear_greed():
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    agent = SentimentAgent(bus=mock_bus)
    req = AnalyzeRequest(symbol="BTCUSDT")

    with patch("cryptoswarm.agents.sentiment.httpx.AsyncClient") as MockHTTP:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"value": "25", "value_classification": "Extreme Fear"}]
        }
        mock_resp.raise_for_status = MagicMock()
        MockHTTP.return_value.__aenter__ = AsyncMock(return_value=MockHTTP.return_value)
        MockHTTP.return_value.__aexit__ = AsyncMock(return_value=False)
        MockHTTP.return_value.get = AsyncMock(return_value=mock_resp)

        await agent._handle(req)

    mock_bus.publish.assert_called_once()
    topic, msg = mock_bus.publish.call_args[0]
    assert topic == "agent.result.sentiment.BTCUSDT"
    assert isinstance(msg, SentimentResult)
    assert msg.score < 0.0   # fear → negative score
    assert msg.correlation_id == req.correlation_id


async def test_sentiment_agent_fallback_on_error():
    """When API is unreachable, SentimentAgent returns neutral score and doesn't crash."""
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    agent = SentimentAgent(bus=mock_bus)
    req = AnalyzeRequest(symbol="ETHUSDT")

    with patch("cryptoswarm.agents.sentiment.httpx.AsyncClient") as MockHTTP:
        MockHTTP.return_value.__aenter__ = AsyncMock(return_value=MockHTTP.return_value)
        MockHTTP.return_value.__aexit__ = AsyncMock(return_value=False)
        MockHTTP.return_value.get = AsyncMock(side_effect=Exception("connection refused"))

        await agent._handle(req)

    _, msg = mock_bus.publish.call_args[0]
    assert msg.score == 0.0
    assert msg.source == "neutral_fallback"


# ---------------------------------------------------------------------------
# Portfolio Agent
# ---------------------------------------------------------------------------

async def test_portfolio_agent_publishes_result():
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock(return_value={
        "approved": True,
        "correlation_penalty": 0.9,
        "reasoning": "Low correlation with existing positions",
    })

    agent = PortfolioAgent(bus=mock_bus, llm=mock_llm)
    req = AnalyzeRequest(symbol="SOLUSDT")
    await agent._handle(req)

    mock_bus.publish.assert_called_once()
    topic, msg = mock_bus.publish.call_args[0]
    assert topic == "agent.result.portfolio.SOLUSDT"
    assert isinstance(msg, PortfolioResult)
    assert msg.correlation_id == req.correlation_id
    assert msg.approved is True


async def test_portfolio_agent_tracks_positions():
    """PortfolioAgent updates its internal position cache from PositionUpdate messages."""
    mock_bus = MagicMock()
    mock_llm = MagicMock()
    agent = PortfolioAgent(bus=mock_bus, llm=mock_llm)

    update = PositionUpdate(
        symbol="BTCUSDT", side="LONG", qty=0.01, entry_price=60000.0,
        mark_price=61000.0, unrealized_pnl=10.0, isolated_margin=120.0,
        liq_price=50000.0, is_closed=False,
    )
    agent._on_position_update(update)
    assert "BTCUSDT" in agent._positions
    assert agent._positions["BTCUSDT"]["side"] == "LONG"


async def test_portfolio_agent_removes_closed_positions():
    mock_bus = MagicMock()
    mock_llm = MagicMock()
    agent = PortfolioAgent(bus=mock_bus, llm=mock_llm)

    open_update = PositionUpdate(
        symbol="BTCUSDT", side="LONG", qty=0.01, entry_price=60000.0,
        mark_price=61000.0, unrealized_pnl=10.0, isolated_margin=120.0,
        liq_price=50000.0, is_closed=False,
    )
    agent._on_position_update(open_update)
    assert "BTCUSDT" in agent._positions

    close_update = PositionUpdate(
        symbol="BTCUSDT", side="LONG", qty=0.0, entry_price=60000.0,
        mark_price=61000.0, unrealized_pnl=0.0, isolated_margin=0.0,
        liq_price=0.0, is_closed=True, close_reason="sl",
    )
    agent._on_position_update(close_update)
    assert "BTCUSDT" not in agent._positions
