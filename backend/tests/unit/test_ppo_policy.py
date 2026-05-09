"""Tests for PPOPolicy wrapper."""
import numpy as np
import pytest
from cryptoswarm.ml.ppo_policy import PPOPolicy
from cryptoswarm.ml.features import FEATURE_SIZE


def _state():
    return np.zeros(FEATURE_SIZE, dtype=np.float32)


def test_predict_before_training_returns_hold():
    policy = PPOPolicy(state_size=FEATURE_SIZE)
    action, confidence = policy.predict(_state())
    assert action == "hold"
    assert confidence == 0.0


def test_update_accumulates_experience():
    policy = PPOPolicy(state_size=FEATURE_SIZE)
    for _ in range(5):
        policy.update(
            state=_state(),
            action="scale_up",
            reward=0.5,
            next_state=_state(),
            done=False,
        )
    assert policy.experience_count == 5


def test_train_requires_minimum_experience():
    policy = PPOPolicy(state_size=FEATURE_SIZE, min_train_samples=100)
    for _ in range(10):
        policy.update(_state(), "hold", 0.0, _state(), False)
    # Should not raise — just skips training silently
    policy.maybe_train()
    assert not policy.is_trained


def test_predict_after_training_returns_valid_action():
    policy = PPOPolicy(state_size=FEATURE_SIZE, min_train_samples=10)
    rng = np.random.default_rng(1)
    for _ in range(15):
        policy.update(
            state=rng.uniform(-1, 1, FEATURE_SIZE).astype(np.float32),
            action="scale_up" if rng.random() > 0.5 else "hold",
            reward=float(rng.uniform(-1, 1)),
            next_state=rng.uniform(-1, 1, FEATURE_SIZE).astype(np.float32),
            done=True,
        )
    policy.maybe_train()
    if policy.is_trained:
        action, conf = policy.predict(_state())
        assert action in ["hold", "scale_up", "scale_down"]
        assert 0.0 <= conf <= 1.0
