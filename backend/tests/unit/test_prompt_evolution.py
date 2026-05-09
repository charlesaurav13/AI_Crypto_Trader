"""Tests for PromptEvolutionEngine."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from cryptoswarm.learning.prompt_evolution import PromptEvolutionEngine


def _make_engine(trades=None):
    mock_pg = MagicMock()
    mock_pg.get_recent_closed_trades = AsyncMock(return_value=trades or [])
    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock(return_value={
        "improved_prompt": "You are an improved quant analyst...",
        "changes_summary": "Added focus on trending markets",
        "perf_score": 0.65,
    })
    mock_store = MagicMock()
    mock_store.get = AsyncMock(return_value="original prompt")
    mock_store.save = AsyncMock()
    mock_store.invalidate = MagicMock()
    return PromptEvolutionEngine(
        pg=mock_pg,
        llm=mock_llm,
        prompt_store=mock_store,
        lookback=10,
    )


async def test_run_once_skips_when_no_trades():
    engine = _make_engine(trades=[])
    await engine.run_once()
    engine._llm.ask.assert_not_called()


async def test_run_once_evolves_worst_agent():
    trades = [
        {
            "correlation_id": f"cid-{i}",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "realized_pnl": -10.0,
            "reward": -0.5,
            "state": "{}",
            "action": '{"agent": "quant"}',
            "exit_reason": "sl",
        }
        for i in range(10)
    ]
    engine = _make_engine(trades=trades)
    await engine.run_once()
    engine._llm.ask.assert_called_once()
    engine._store.save.assert_called_once()


async def test_run_once_saves_evolved_prompt():
    trades = [
        {
            "correlation_id": f"cid-{i}", "symbol": "BTCUSDT",
            "side": "LONG", "realized_pnl": -5.0, "reward": -0.3,
            "state": "{}", "action": '{"agent": "risk"}', "exit_reason": "sl",
        }
        for i in range(10)
    ]
    engine = _make_engine(trades=trades)
    await engine.run_once()
    save_call = engine._store.save.call_args
    assert save_call is not None
    assert "improved" in save_call[0][1].lower() or len(save_call[0][1]) > 10


async def test_run_once_handles_llm_error_gracefully():
    trades = [
        {
            "correlation_id": "cid-1", "symbol": "BTCUSDT", "side": "LONG",
            "realized_pnl": -5.0, "reward": -0.3, "state": "{}",
            "action": '{"agent": "quant"}', "exit_reason": "sl",
        }
        for _ in range(10)
    ]
    engine = _make_engine(trades=trades)
    engine._llm.ask = AsyncMock(side_effect=Exception("LLM timeout"))
    await engine.run_once()  # Must not raise
    engine._store.save.assert_not_called()
