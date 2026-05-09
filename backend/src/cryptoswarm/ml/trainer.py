"""MLTrainer — batch retrains XGBoost + LSTM every 6h using closed trade history."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from cryptoswarm.ml.features import FEATURE_SIZE

if TYPE_CHECKING:
    from cryptoswarm.ml.features import FeatureEngine
    from cryptoswarm.ml.lstm_model import LSTMModel
    from cryptoswarm.ml.model_store import ModelStore
    from cryptoswarm.ml.xgboost_model import XGBoostModel
    from cryptoswarm.storage.postgres import PostgresWriter
    from cryptoswarm.storage.timescale import TimescaleWriter

logger = logging.getLogger(__name__)


class MLTrainer:
    def __init__(
        self,
        ts: "TimescaleWriter",
        pg: "PostgresWriter",
        xgb_model: "XGBoostModel",
        lstm_model: "LSTMModel",
        model_store: "ModelStore",
        features: "FeatureEngine",
        interval_s: int = 21600,
        min_samples: int = 500,
        seq_len: int = 30,
    ) -> None:
        self._ts = ts
        self._pg = pg
        self._xgb = xgb_model
        self._lstm = lstm_model
        self._store = model_store
        self._features = features
        self._interval = interval_s
        self._min_samples = min_samples
        self._seq_len = seq_len

    async def run(self) -> None:
        """Run forever: retrain every interval_s."""
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("MLTrainer.run error: %s", exc)

    async def run_once(self) -> None:
        """Single training cycle. Called directly in tests."""
        trades = await self._pg.get_recent_closed_trades(limit=5000)
        if len(trades) < self._min_samples:
            logger.info(
                "MLTrainer: only %d closed trades (need %d) — skipping",
                len(trades), self._min_samples,
            )
            return

        started = datetime.now(timezone.utc)
        run_id = await self._pg.insert_training_run(
            model_type="xgboost+lstm", started_at=started
        )

        X_flat, y_regime, y_dir, X_seq, y_seq_dir = self._build_training_data(trades)

        metrics: dict = {}

        # Train XGBoost
        if len(X_flat) >= self._min_samples:
            try:
                self._xgb.fit(X_flat, y_regime, y_dir)
                self._store.save("xgboost", self._xgb)
                metrics["xgb_samples"] = len(X_flat)
                metrics["xgb_version"] = self._xgb.version
                logger.info("MLTrainer: XGBoost trained on %d samples", len(X_flat))
            except Exception as exc:
                logger.warning("MLTrainer: XGBoost training failed: %s", exc)

        # Train LSTM
        if len(X_seq) >= self._min_samples:
            try:
                self._lstm.fit(X_seq, y_seq_dir)
                self._store.save("lstm", self._lstm)
                metrics["lstm_samples"] = len(X_seq)
                metrics["lstm_version"] = self._lstm.version
                logger.info("MLTrainer: LSTM trained on %d sequences", len(X_seq))
            except Exception as exc:
                logger.warning("MLTrainer: LSTM training failed: %s", exc)

        await self._pg.update_training_run(
            run_id=run_id,
            completed_at=datetime.now(timezone.utc),
            sample_count=len(trades),
            metrics=metrics,
        )

    def _build_training_data(
        self, trades: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse trades into feature arrays and labels."""
        X_flat, y_regime, y_dir = [], [], []

        for t in trades:
            try:
                state = json.loads(t["state"]) if isinstance(t["state"], str) else dict(t["state"])
                features = np.array(
                    [float(state.get(f"f{i}", 0.0)) for i in range(FEATURE_SIZE)],
                    dtype=np.float32,
                )
                # Direction label: 1 if trade was profitable, else 0
                direction = 1 if float(t["realized_pnl"]) > 0 else 0
                # Regime label: derive from P&L magnitude (approximate)
                pnl = float(t["realized_pnl"])
                if pnl > 0.05:
                    regime = 0   # trending_up
                elif pnl < -0.05:
                    regime = 1   # trending_down
                elif abs(pnl) < 0.01:
                    regime = 2   # ranging
                else:
                    regime = 3   # volatile
                X_flat.append(features)
                y_regime.append(regime)
                y_dir.append(direction)
            except Exception:
                continue

        if not X_flat:
            empty = np.empty((0, FEATURE_SIZE), dtype=np.float32)
            return empty, np.array([]), np.array([]), np.empty((0, self._seq_len, FEATURE_SIZE)), np.array([])

        X_arr = np.stack(X_flat)
        y_r_arr = np.array(y_regime, dtype=np.int32)
        y_d_arr = np.array(y_dir, dtype=np.int32)

        # Build sequences for LSTM (rolling windows)
        X_seq, y_seq_dir = [], []
        for i in range(self._seq_len, len(X_arr)):
            X_seq.append(X_arr[i - self._seq_len:i])
            y_seq_dir.append(y_d_arr[i])

        if X_seq:
            X_seq_arr = np.stack(X_seq).astype(np.float32)
            y_seq_arr = np.array(y_seq_dir, dtype=np.int32)
        else:
            X_seq_arr = np.empty((0, self._seq_len, FEATURE_SIZE), dtype=np.float32)
            y_seq_arr = np.array([], dtype=np.int32)

        return X_arr, y_r_arr, y_d_arr, X_seq_arr, y_seq_arr
