"""Tests for XGBoostModel — no real training data needed."""
import numpy as np
import pytest
from cryptoswarm.ml.xgboost_model import XGBoostModel
from cryptoswarm.ml.features import FEATURE_SIZE


def _fake_features(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, (n, FEATURE_SIZE)).astype(np.float32)


def test_predict_before_training_returns_neutral():
    model = XGBoostModel()
    vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
    regime, direction, confidence = model.predict(vec)
    assert regime == "ranging"
    assert direction == "up"
    assert confidence == 0.0


def test_fit_and_predict_regime():
    model = XGBoostModel()
    X = _fake_features(200)
    # Labels: 0=trending_up 1=trending_down 2=ranging 3=volatile
    y_regime = np.random.randint(0, 4, 200)
    y_dir = np.random.randint(0, 2, 200)
    model.fit(X, y_regime, y_dir)
    regime, direction, confidence = model.predict(X[0])
    assert regime in ["trending_up", "trending_down", "ranging", "volatile"]
    assert direction in ["up", "down"]
    assert 0.0 <= confidence <= 1.0


def test_fit_requires_minimum_samples():
    model = XGBoostModel()
    X = _fake_features(5)
    y_regime = np.array([0, 1, 2, 3, 0])
    y_dir = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="minimum"):
        model.fit(X, y_regime, y_dir)


def test_model_version_updates_after_fit():
    model = XGBoostModel()
    assert model.version is None
    X = _fake_features(200)
    y_r = np.random.randint(0, 4, 200)
    y_d = np.random.randint(0, 2, 200)
    model.fit(X, y_r, y_d)
    assert model.version is not None
