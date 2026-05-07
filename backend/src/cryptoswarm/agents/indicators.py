"""Technical indicator computation using the `ta` library.

Requires at least 60 OHLCV bars (50-period EMA needs warmup).
Returns a flat dict of indicator values at the last (most recent) bar.
"""
from __future__ import annotations

import pandas as pd
import ta


def compute_indicators(ohlcv: list[dict]) -> dict:
    """
    Args:
        ohlcv: list of dicts with keys open, high, low, close, volume
               in chronological order (oldest first).
    Returns:
        Flat dict: rsi, macd_hist, bb_pband, ema20, ema50, atr, close, ema_cross
    Raises:
        ValueError: if fewer than 60 bars are supplied.
    """
    if len(ohlcv) < 60:
        raise ValueError(f"compute_indicators requires at least 60 bars, got {len(ohlcv)}")

    df = pd.DataFrame(ohlcv)
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]

    macd = ta.trend.MACD(close=close)
    macd_hist = macd.macd_diff().iloc[-1]

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_pband = bb.bollinger_pband().iloc[-1]

    ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]

    atr = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range().iloc[-1]

    return {
        "rsi":       round(float(rsi), 2),
        "macd_hist": round(float(macd_hist), 6),
        "bb_pband":  round(float(bb_pband), 4),
        "ema20":     round(float(ema20), 4),
        "ema50":     round(float(ema50), 4),
        "atr":       round(float(atr), 4),
        "close":     round(float(close.iloc[-1]), 4),
        "ema_cross": "bullish" if float(ema20) > float(ema50) else "bearish",
    }
