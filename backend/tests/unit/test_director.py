"""Tests for DirectorAgent orchestration logic."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from cryptoswarm.agents.director import DirectorAgent
from cryptoswarm.bus.messages import (
    AnalyzeRequest, QuantResult, RiskResult, SentimentResult, PortfolioResult, Signal,
)
from cryptoswarm.config.settings import Settings


def _make_settings(**overrides) -> Settings:
    return Settings(
        paper_trading=True,
        director_interval_s=9999,   # prevent auto-scheduler from firing in tests
        agent_timeout_s=2,
        symbols="BTCUSDT",
        **overrides,
    )


def _make_results(symbol: str, cid: str) -> dict:
    return {
        "quant": QuantResult(
            symbol=symbol, correlation_id=cid,
            regime="trending_up", signal_strength=0.7, confidence=0.8,
            reasoning="EMA cross", indicators={"rsi": 62.0, "close": 65000.0},
        ),
        "risk": RiskResult(
            symbol=symbol, correlation_id=cid,
            kelly_fraction=0.05, max_loss_usdt=50.0, reasoning="moderate risk",
        ),
        "sentiment": SentimentResult(
            symbol=symbol, correlation_id=cid,
            score=0.2, source="fear_greed_api", summary="Greed (60/100)",
        ),
        "portfolio": PortfolioResult(
            symbol=symbol, correlation_id=cid,
            approved=True, correlation_penalty=1.0, reasoning="no existing",
        ),
    }


async def test_director_synthesizes_buy_signal():
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock(return_value={
        "action": "buy",
        "side": "LONG",
        "confidence": 0.82,
        "size_pct": 0.05,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
        "reasoning": "Strong confluence across all agents",
    })

    mock_decisions = MagicMock()
    mock_decisions.insert = AsyncMock()

    settings = _make_settings()
    agent = DirectorAgent(bus=mock_bus, llm=mock_llm, decisions=mock_decisions, settings=settings)

    req = AnalyzeRequest(symbol="BTCUSDT")
    results = _make_results("BTCUSDT", req.correlation_id)

    await agent._analyze_symbol_with_results("BTCUSDT", req, results)

    # Signal should be published to signal.execute
    published_topics = [call[0][0] for call in mock_bus.publish.call_args_list]
    assert "signal.execute" in published_topics

    signal_call = next(c for c in mock_bus.publish.call_args_list if c[0][0] == "signal.execute")
    signal: Signal = signal_call[0][1]
    assert signal.symbol == "BTCUSDT"
    assert signal.side == "LONG"
    assert signal.leverage == settings.risk.max_leverage
    assert signal.sl > 0
    assert signal.tp > signal.sl  # for LONG: tp > entry > sl

    mock_decisions.insert.assert_called_once()


async def test_director_hold_does_not_publish_signal():
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock(return_value={
        "action": "hold",
        "side": "LONG",
        "confidence": 0.3,
        "size_pct": 0.0,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
        "reasoning": "Mixed signals, staying out",
    })

    mock_decisions = MagicMock()
    mock_decisions.insert = AsyncMock()

    settings = _make_settings()
    agent = DirectorAgent(bus=mock_bus, llm=mock_llm, decisions=mock_decisions, settings=settings)

    req = AnalyzeRequest(symbol="BTCUSDT")
    results = _make_results("BTCUSDT", req.correlation_id)
    # make it a ranging/hold scenario
    results["quant"].regime = "ranging"
    results["quant"].signal_strength = 0.1

    await agent._analyze_symbol_with_results("BTCUSDT", req, results)

    published_topics = [call[0][0] for call in mock_bus.publish.call_args_list]
    assert "signal.execute" not in published_topics
    mock_decisions.insert.assert_called_once()


async def test_director_blocked_by_portfolio():
    """When portfolio agent says not approved, Director holds without calling LLM."""
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock()  # should NOT be called

    mock_decisions = MagicMock()
    mock_decisions.insert = AsyncMock()

    settings = _make_settings()
    agent = DirectorAgent(bus=mock_bus, llm=mock_llm, decisions=mock_decisions, settings=settings)

    req = AnalyzeRequest(symbol="BTCUSDT")
    results = _make_results("BTCUSDT", req.correlation_id)
    results["portfolio"].approved = False
    results["portfolio"].reasoning = "Too many correlated positions"

    await agent._analyze_symbol_with_results("BTCUSDT", req, results)

    mock_llm.ask.assert_not_called()
    published_topics = [call[0][0] for call in mock_bus.publish.call_args_list]
    assert "signal.execute" not in published_topics


async def test_director_short_signal():
    """Director publishes SHORT signal with inverted sl/tp."""
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock(return_value={
        "action": "sell",
        "side": "SHORT",
        "confidence": 0.75,
        "size_pct": 0.04,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
        "reasoning": "Bearish breakdown confirmed",
    })

    mock_decisions = MagicMock()
    mock_decisions.insert = AsyncMock()

    settings = _make_settings()
    agent = DirectorAgent(bus=mock_bus, llm=mock_llm, decisions=mock_decisions, settings=settings)

    req = AnalyzeRequest(symbol="ETHUSDT")
    results = _make_results("ETHUSDT", req.correlation_id)
    results["quant"].regime = "trending_down"
    results["quant"].signal_strength = -0.7

    await agent._analyze_symbol_with_results("ETHUSDT", req, results)

    signal_call = next(c for c in mock_bus.publish.call_args_list if c[0][0] == "signal.execute")
    signal: Signal = signal_call[0][1]
    assert signal.side == "SHORT"
    assert signal.sl > signal.tp  # for SHORT: sl > entry > tp
