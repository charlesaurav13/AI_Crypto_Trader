"""Tests for FeatureEngine — DB calls mocked."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from cryptoswarm.ml.features import FeatureEngine, FEATURE_SIZE


def _make_engine():
    mock_ts = MagicMock()
    mock_pg = MagicMock()
    mock_ts.fetch_klines = AsyncMock(return_value=_fake_klines(200))
    mock_pg.get_news_sentiment_for_symbol = AsyncMock(return_value=[
        {"score": 0.5, "relevance": 0.8, "summary": "good"},
    ])
    return FeatureEngine(ts=mock_ts, pg=mock_pg)


def _fake_klines(n: int) -> list[dict]:
    import random
    price = 65000.0
    rows = []
    for i in range(n):
        price *= (1 + random.uniform(-0.001, 0.001))
        rows.append({
            "open": price * 0.999, "high": price * 1.001,
            "low": price * 0.998, "close": price,
            "volume": random.uniform(100, 500),
        })
    return rows


def test_feature_size_constant():
    assert FEATURE_SIZE == 25


async def test_build_returns_correct_shape():
    engine = _make_engine()
    vec = await engine.build("BTCUSDT")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (FEATURE_SIZE,)
    assert not np.any(np.isnan(vec))


async def test_build_sequence_returns_3d_array():
    engine = _make_engine()
    seq = await engine.build_sequence("BTCUSDT", lookback=30)
    assert seq.shape == (30, FEATURE_SIZE)


async def test_build_returns_neutral_on_insufficient_data():
    engine = _make_engine()
    engine._ts.fetch_klines = AsyncMock(return_value=_fake_klines(10))
    vec = await engine.build("BTCUSDT")
    assert vec.shape == (FEATURE_SIZE,)
    # All zeros when not enough data
    assert np.all(vec == 0.0)


async def test_build_tolerates_news_fetch_error():
    engine = _make_engine()
    engine._pg.get_news_sentiment_for_symbol = AsyncMock(side_effect=Exception("DB error"))
    vec = await engine.build("BTCUSDT")
    assert vec.shape == (FEATURE_SIZE,)
    assert not np.any(np.isnan(vec))
