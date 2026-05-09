"""Tests for LSTMModel — no GPU needed."""
import numpy as np
import pytest
from cryptoswarm.ml.lstm_model import LSTMModel
from cryptoswarm.ml.features import FEATURE_SIZE


def _fake_seq(batch: int, seq_len: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, (batch, seq_len, FEATURE_SIZE)).astype(np.float32)


def test_predict_before_training_returns_neutral():
    model = LSTMModel(seq_len=30)
    seq = np.zeros((30, FEATURE_SIZE), dtype=np.float32)
    direction, confidence = model.predict(seq)
    assert direction in ["up", "down"]
    assert confidence == 0.0


def test_fit_and_predict():
    model = LSTMModel(seq_len=10, hidden_size=16, num_layers=1, epochs=2)
    X = _fake_seq(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    direction, confidence = model.predict(X[0])
    assert direction in ["up", "down"]
    assert 0.0 <= confidence <= 1.0


def test_fit_requires_minimum_samples():
    model = LSTMModel(seq_len=10)
    X = _fake_seq(5, 10)
    y = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="minimum"):
        model.fit(X, y)
