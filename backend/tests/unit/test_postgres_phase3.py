"""Unit tests for Phase 3 PostgresWriter methods — all DB calls mocked."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from cryptoswarm.storage.postgres import PostgresWriter


def _make_writer() -> PostgresWriter:
    w = PostgresWriter.__new__(PostgresWriter)
    w._pool = MagicMock()
    w._pool.execute = AsyncMock()
    w._pool.fetch = AsyncMock(return_value=[])
    w._pool.fetchrow = AsyncMock(return_value=None)
    return w


async def test_insert_news_item_calls_execute():
    w = _make_writer()
    w._pool.fetchval = AsyncMock(return_value=42)
    result = await w.insert_news_item(
        source="coindesk",
        url="https://coindesk.com/article/1",
        title="BTC up",
        body="Bitcoin rose today...",
    )
    assert result == 42
    w._pool.fetchval.assert_called_once()


async def test_insert_news_sentiment_calls_execute():
    w = _make_writer()
    await w.insert_news_sentiment(
        news_item_id=1, symbol="BTCUSDT", model="qwen2.5:7b",
        relevance=0.9, score=0.7, summary="Bullish BTC news",
    )
    w._pool.execute.assert_called_once()


async def test_get_news_sentiment_for_symbol_returns_list():
    w = _make_writer()
    w._pool.fetch = AsyncMock(return_value=[
        {"score": 0.5, "relevance": 0.8, "summary": "Good", "ts": datetime.now(timezone.utc)},
    ])
    rows = await w.get_news_sentiment_for_symbol("BTCUSDT", hours=6)
    assert len(rows) == 1


async def test_save_agent_prompt_inserts():
    w = _make_writer()
    # transaction() is a sync call returning an async CM; acquire() returns an async CM
    txn_cm = AsyncMock()
    txn_cm.__aenter__ = AsyncMock(return_value=None)
    txn_cm.__aexit__ = AsyncMock(return_value=False)
    mock_conn = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=txn_cm)
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    w._pool.acquire = MagicMock(return_value=acquire_cm)
    await w.save_agent_prompt(
        agent_name="quant", system_prompt="You are a quant...", perf_score=0.42
    )
    assert mock_conn.execute.await_count == 2


async def test_get_agent_prompt_returns_none_when_missing():
    w = _make_writer()
    w._pool.fetchrow = AsyncMock(return_value=None)
    result = await w.get_agent_prompt("quant")
    assert result is None


async def test_update_rl_tuple_reward_calls_execute():
    w = _make_writer()
    await w.update_rl_tuple_reward(
        correlation_id="cid-123",
        reward=0.85,
        next_state={"rsi": 62.0},
    )
    w._pool.execute.assert_called_once()


async def test_insert_ml_signal_calls_execute():
    w = _make_writer()
    await w.insert_ml_signal(
        symbol="BTCUSDT", regime_pred="trending_up", direction_pred="up",
        short_direction="up", confidence=0.8,
        size_adjustment="scale_up", model_version="2026-05-08T06:00:00",
    )
    w._pool.execute.assert_called_once()


async def test_insert_training_run_returns_id():
    w = _make_writer()
    w._pool.fetchval = AsyncMock(return_value=7)
    result = await w.insert_training_run(
        model_type="xgboost",
        started_at=datetime.now(timezone.utc),
    )
    assert result == 7


async def test_update_training_run_calls_execute():
    w = _make_writer()
    await w.update_training_run(
        run_id=7,
        completed_at=datetime.now(timezone.utc),
        sample_count=1200,
        metrics={"accuracy": 0.72},
    )
    w._pool.execute.assert_called_once()


async def test_get_recent_closed_trades_returns_list():
    w = _make_writer()
    w._pool.fetch = AsyncMock(return_value=[])
    rows = await w.get_recent_closed_trades(limit=50)
    assert rows == []
