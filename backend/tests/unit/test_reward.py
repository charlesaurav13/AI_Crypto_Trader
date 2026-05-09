"""Tests for composite reward function."""
import pytest
from cryptoswarm.ml.reward import compute_reward, RewardConfig


def test_profitable_trade_positive_reward():
    cfg = RewardConfig()
    r = compute_reward(
        realized_pnl=50.0, position_size_usd=1000.0,
        tp_pct=0.04, sl_pct=0.02,
        seconds_in_loss=0, total_seconds=3600,
    )
    assert r > 0.0


def test_losing_trade_negative_reward():
    cfg = RewardConfig()
    r = compute_reward(
        realized_pnl=-30.0, position_size_usd=1000.0,
        tp_pct=0.04, sl_pct=0.02,
        seconds_in_loss=3600, total_seconds=3600,
    )
    assert r < 0.0


def test_reward_bounded():
    cfg = RewardConfig()
    r = compute_reward(
        realized_pnl=999999.0, position_size_usd=100.0,
        tp_pct=0.1, sl_pct=0.01,
        seconds_in_loss=0, total_seconds=100,
    )
    assert r <= 5.0   # capped


def test_big_loss_penalised_harder_than_small_loss():
    big = compute_reward(
        realized_pnl=-200.0, position_size_usd=1000.0,
        tp_pct=0.02, sl_pct=0.02,
        seconds_in_loss=1000, total_seconds=1000,
    )
    small = compute_reward(
        realized_pnl=-20.0, position_size_usd=1000.0,
        tp_pct=0.02, sl_pct=0.02,
        seconds_in_loss=100, total_seconds=1000,
    )
    assert big < small


def test_good_rr_ratio_boosts_reward():
    good_rr = compute_reward(
        realized_pnl=10.0, position_size_usd=500.0,
        tp_pct=0.06, sl_pct=0.01,
        seconds_in_loss=0, total_seconds=500,
    )
    bad_rr = compute_reward(
        realized_pnl=10.0, position_size_usd=500.0,
        tp_pct=0.02, sl_pct=0.02,
        seconds_in_loss=0, total_seconds=500,
    )
    assert good_rr > bad_rr
