import pytest
from cryptoswarm.risk.breakers import CircuitBreakerState, DailyLossBreaker, MaxDrawdownBreaker


def test_daily_loss_not_tripped_initially():
    breaker = DailyLossBreaker(starting_balance=1000.0, threshold_pct=0.03)
    assert breaker.is_tripped() is False


def test_daily_loss_trips_at_threshold():
    breaker = DailyLossBreaker(starting_balance=1000.0, threshold_pct=0.03)
    breaker.update_pnl(-30.0)   # exactly -3%
    assert breaker.is_tripped() is True


def test_daily_loss_not_tripped_below_threshold():
    breaker = DailyLossBreaker(starting_balance=1000.0, threshold_pct=0.03)
    breaker.update_pnl(-29.99)
    assert breaker.is_tripped() is False


def test_daily_loss_manual_reset():
    breaker = DailyLossBreaker(starting_balance=1000.0, threshold_pct=0.03)
    breaker.update_pnl(-50.0)
    assert breaker.is_tripped() is True
    breaker.reset()
    assert breaker.is_tripped() is False


def test_daily_loss_cumulative():
    """Multiple small losses can accumulate to trip the breaker."""
    breaker = DailyLossBreaker(starting_balance=1000.0, threshold_pct=0.03)
    breaker.update_pnl(-10.0)
    breaker.update_pnl(-10.0)
    breaker.update_pnl(-10.0)   # cumulative = -30.0 → trip
    assert breaker.is_tripped() is True


def test_drawdown_breaker():
    breaker = MaxDrawdownBreaker(threshold_pct=0.15)
    breaker.update_equity(1000.0)  # peak = 1000
    breaker.update_equity(1100.0)  # peak = 1100
    breaker.update_equity(935.0)   # 935/1100 ≈ 85% → 15% drawdown → trip
    assert breaker.is_tripped() is True


def test_drawdown_not_tripped_small_drop():
    breaker = MaxDrawdownBreaker(threshold_pct=0.15)
    breaker.update_equity(1000.0)
    breaker.update_equity(870.0)   # 13% drawdown → not tripped
    assert breaker.is_tripped() is False


def test_drawdown_reset():
    breaker = MaxDrawdownBreaker(threshold_pct=0.15)
    breaker.update_equity(1000.0)
    breaker.update_equity(800.0)   # 20% drawdown → trip
    assert breaker.is_tripped() is True
    breaker.reset()
    assert breaker.is_tripped() is False
