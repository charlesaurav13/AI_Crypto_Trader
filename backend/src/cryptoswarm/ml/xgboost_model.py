"""XGBoostModel — regime classification + 1h direction prediction."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from cryptoswarm.ml.features import FEATURE_SIZE

logger = logging.getLogger(__name__)

_REGIME_LABELS = ["trending_up", "trending_down", "ranging", "volatile"]
_MIN_SAMPLES = 100


class XGBoostModel:
    def __init__(self) -> None:
        self._regime_clf = None
        self._direction_clf = None
        self.version: str | None = None

    def fit(
        self,
        X: np.ndarray,           # (N, FEATURE_SIZE)
        y_regime: np.ndarray,    # (N,) int 0-3
        y_direction: np.ndarray, # (N,) int 0=up 1=down
    ) -> None:
        if len(X) < _MIN_SAMPLES:
            raise ValueError(f"XGBoostModel minimum {_MIN_SAMPLES} samples, got {len(X)}")
        import xgboost as xgb
        self._regime_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", verbosity=0,
        )
        self._direction_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0,
        )
        self._regime_clf.fit(X, y_regime)
        self._direction_clf.fit(X, y_direction)
        self.version = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        logger.info("XGBoostModel: trained on %d samples version=%s", len(X), self.version)

    def predict(
        self, features: np.ndarray
    ) -> tuple[str, Literal["up", "down"], float]:
        """Return (regime, direction, confidence). Neutral if not trained."""
        if self._regime_clf is None or self._direction_clf is None:
            return "ranging", "up", 0.0
        x = features.reshape(1, -1)
        regime_idx = int(self._regime_clf.predict(x)[0])
        regime = _REGIME_LABELS[regime_idx]
        dir_proba = self._direction_clf.predict_proba(x)[0]
        direction: Literal["up", "down"] = "up" if dir_proba[0] >= 0.5 else "down"
        confidence = float(max(dir_proba))
        return regime, direction, round(confidence, 4)
