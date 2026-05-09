"""Tests for composite reward function."""
import pytest
from cryptoswarm.ml.reward import compute_reward, RewardConfig


def test_profitable_trade_positive_reward():
    r = compute_reward(
        realized_pnl=50.0, position_size_usd=1000.0,
        tp_pct=0.04, sl_pct=0.02,
        seconds_in_loss=0, total_seconds=3600,
    )
    assert r > 0.0


def test_losing_trade_negative_reward():
    r = compute_reward(
        realized_pnl=-30.0, position_size_usd=1000.0,
        tp_pct=0.04, sl_pct=0.02,
        seconds_in_loss=3600, total_seconds=3600,
    )
    assert r < 0.0


def test_reward_upper_bound():
    """Positive reward cannot exceed the structural max ~0.80 from weight design."""
    r = compute_reward(
        realized_pnl=999999.0, position_size_usd=100.0,
        tp_pct=0.1, sl_pct=0.01,
        seconds_in_loss=0, total_seconds=100,
    )
    assert r <= 5.0      # hard clamp
    assert r <= 1.0      # structural max (weights sum to 0.80 positive)


def test_reward_lower_bound():
    """Lower bound clamp at -5.0."""
    r = compute_reward(
        realized_pnl=-999999.0, position_size_usd=100.0,
        tp_pct=0.02, sl_pct=0.02,
        seconds_in_loss=10000, total_seconds=100,
    )
    assert r >= -5.0


def test_break_even_is_not_penalised():
    r = compute_reward(
        realized_pnl=0.0, position_size_usd=1000.0,
        tp_pct=0.02, sl_pct=0.02,
        seconds_in_loss=0, total_seconds=600,
    )
    # win_contribution=0.0, should be mildly positive from rr_contribution
    assert r >= 0.0


def test_custom_config_changes_output():
    heavy_drawdown = RewardConfig(w4=0.50, w1=0.10, w2=0.10, w3=0.10, w5=0.05, drawdown_scale=5.0)
    light_drawdown = RewardConfig(w4=0.05, w1=0.40, w2=0.20, w3=0.20, w5=0.05, drawdown_scale=1.0)
    r_heavy = compute_reward(-50.0, 1000.0, 0.02, 0.02, 500, 1000, cfg=heavy_drawdown)
    r_light = compute_reward(-50.0, 1000.0, 0.02, 0.02, 500, 1000, cfg=light_drawdown)
    assert r_heavy < r_light


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
