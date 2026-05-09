"""FeatureEngine — builds fixed-size feature vectors from klines + news sentiment."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from cryptoswarm.storage.postgres import PostgresWriter
    from cryptoswarm.storage.timescale import TimescaleWriter

logger = logging.getLogger(__name__)

# Fixed feature vector size — NEVER change without retraining all models
FEATURE_SIZE = 25
_MIN_BARS = 60   # minimum klines needed for indicator computation


class FeatureEngine:
    def __init__(self, ts: "TimescaleWriter", pg: "PostgresWriter") -> None:
        self._ts = ts
        self._pg = pg

    async def build(self, symbol: str) -> np.ndarray:
        """Build a single FEATURE_SIZE feature vector. Returns zeros if insufficient data."""
        try:
            rows = await self._ts.fetch_klines(symbol, limit=200)
            if len(rows) < _MIN_BARS:
                logger.warning("FeatureEngine: only %d bars for %s", len(rows), symbol)
                return np.zeros(FEATURE_SIZE, dtype=np.float32)
            news_score, news_count = await self._get_news_features(symbol)
            return self._compute(rows, news_score, news_count)
        except Exception as exc:
            logger.error("FeatureEngine.build error for %s: %s", symbol, exc)
            return np.zeros(FEATURE_SIZE, dtype=np.float32)

    async def build_sequence(self, symbol: str, lookback: int = 30) -> np.ndarray:
        """Build (lookback, FEATURE_SIZE) sequence for LSTM. Returns zeros if insufficient."""
        try:
            rows = await self._ts.fetch_klines(symbol, limit=lookback + 200)
            if len(rows) < _MIN_BARS + lookback:
                return np.zeros((lookback, FEATURE_SIZE), dtype=np.float32)
            news_score, news_count = await self._get_news_features(symbol)
            seqs = []
            for i in range(lookback, 0, -1):
                window = rows[: len(rows) - i + 1]
                seqs.append(self._compute(window, news_score, news_count))
            return np.stack(seqs, axis=0).astype(np.float32)
        except Exception as exc:
            logger.error("FeatureEngine.build_sequence error: %s", exc)
            return np.zeros((lookback, FEATURE_SIZE), dtype=np.float32)

    def _compute(self, rows: list[dict], news_score: float, news_count: int) -> np.ndarray:
        import ta
        df = pd.DataFrame(rows)
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        vol   = df["volume"].astype(float)

        # --- Price action (5 features) ---
        rsi = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1] or 50.0)
        macd_obj = ta.trend.MACD(close)
        macd_hist = float(macd_obj.macd_diff().iloc[-1] or 0.0)
        bb = ta.volatility.BollingerBands(close, window=20)
        bb_pct = float(bb.bollinger_pband().iloc[-1] or 0.5)
        ema20 = float(ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1] or close.iloc[-1])
        ema50 = float(ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1] or close.iloc[-1])
        ema_cross = float(ema20 - ema50) / float(close.iloc[-1])  # normalised

        # --- Volatility (2 features) ---
        atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1] or 0.0)
        atr_norm = atr / float(close.iloc[-1])
        bb_width = float(bb.bollinger_wband().iloc[-1] or 0.0)

        # --- Volume (3 features) ---
        vol_mean = float(vol.rolling(20).mean().iloc[-1] or vol.mean())
        vol_ratio = float(vol.iloc[-1]) / (vol_mean + 1e-9)
        obv = float(ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume().iloc[-1] or 0.0)
        obv_norm = obv / (float(close.mean()) * float(vol.mean()) + 1e-9)

        # --- Trend (3 features) ---
        adx = float(ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1] or 0.0) / 100.0
        ema50_slope = float(
            ta.trend.EMAIndicator(close, window=50).ema_indicator().diff(5).iloc[-1] or 0.0
        ) / float(close.iloc[-1])
        close_norm = (float(close.iloc[-1]) - float(close.rolling(50).mean().iloc[-1] or close.mean())) / (
            float(close.rolling(50).std().iloc[-1] or 1.0) + 1e-9
        )

        # --- RSI momentum (2 features) ---
        rsi_norm = (rsi - 50.0) / 50.0
        stoch_rsi_raw = ta.momentum.StochRSIIndicator(close, window=14)
        stoch_rsi = float(stoch_rsi_raw.stochrsi().iloc[-1] or 0.5)

        # --- News (3 features) ---
        news_score_f = float(news_score)
        news_count_norm = min(float(news_count) / 20.0, 1.0)
        news_present = 1.0 if news_count > 0 else 0.0

        # --- Candle pattern (3 features) ---
        last_return = float((close.iloc[-1] - close.iloc[-2]) / (close.iloc[-2] + 1e-9))
        ret_3 = float((close.iloc[-1] - close.iloc[-4]) / (close.iloc[-4] + 1e-9))
        ret_10 = float((close.iloc[-1] - close.iloc[-11]) / (close.iloc[-11] + 1e-9))

        # --- Volatility regime (3 features) ---
        returns_std = float(close.pct_change().rolling(20).std().iloc[-1] or 0.01)
        high_low_ratio = float((high.iloc[-1] - low.iloc[-1]) / (close.iloc[-1] + 1e-9))
        vol_acceleration = float(vol.iloc[-1] / (vol.iloc[-5] + 1e-9))

        features = np.array([
            # Price action (5)
            rsi_norm, macd_hist / (float(close.mean()) + 1e-9),
            bb_pct - 0.5, ema_cross, stoch_rsi - 0.5,
            # Volatility (2)
            min(atr_norm, 0.1) * 10, min(bb_width, 0.2) * 5,
            # Volume (3)
            min(vol_ratio - 1.0, 3.0) / 3.0, min(abs(obv_norm), 1.0) * np.sign(obv_norm), 0.0,
            # Trend (3)
            adx, np.clip(ema50_slope * 100, -1, 1), np.clip(close_norm, -3, 3) / 3,
            # News (3)
            news_score_f, news_count_norm, news_present,
            # Candle (3)
            np.clip(last_return * 100, -1, 1),
            np.clip(ret_3 * 100, -1, 1),
            np.clip(ret_10 * 100, -1, 1),
            # Volatility regime (3)
            min(returns_std * 100, 1.0),
            min(high_low_ratio * 50, 1.0),
            min(vol_acceleration - 1.0, 2.0) / 2.0,
            # Padding to reach FEATURE_SIZE=25
            0.0, 0.0,
        ], dtype=np.float32)

        assert len(features) == FEATURE_SIZE
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    async def _get_news_features(self, symbol: str) -> tuple[float, int]:
        try:
            rows = await self._pg.get_news_sentiment_for_symbol(symbol, hours=6)
            if not rows:
                return 0.0, 0
            total_w = sum(float(r["relevance"]) for r in rows)
            if total_w == 0:
                return 0.0, 0
            weighted = sum(float(r["score"]) * float(r["relevance"]) for r in rows)
            return weighted / total_w, len(rows)
        except Exception:
            return 0.0, 0
