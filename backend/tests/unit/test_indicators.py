import math
import pytest
from cryptoswarm.agents.indicators import compute_indicators


def _make_ohlcv(n: int = 100) -> list[dict]:
    """Produce n synthetic candles with a mild uptrend."""
    base = 50000.0
    rows = []
    for i in range(n):
        close = base + i * 10 + math.sin(i * 0.3) * 200
        rows.append({
            "open":   close - 50,
            "high":   close + 100,
            "low":    close - 100,
            "close":  close,
            "volume": 100.0 + i,
        })
    return rows


def test_compute_indicators_keys():
    rows = _make_ohlcv(100)
    result = compute_indicators(rows)
    for key in ["rsi", "macd_hist", "bb_pband", "ema20", "ema50", "atr", "close", "ema_cross"]:
        assert key in result, f"Missing key: {key}"


def test_rsi_range():
    rows = _make_ohlcv(100)
    result = compute_indicators(rows)
    assert 0.0 <= result["rsi"] <= 100.0


def test_ema_cross_values():
    rows = _make_ohlcv(100)
    result = compute_indicators(rows)
    assert result["ema_cross"] in ("bullish", "bearish")


def test_requires_at_least_60_bars():
    with pytest.raises(ValueError, match="at least 60"):
        compute_indicators(_make_ohlcv(30))
