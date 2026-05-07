import pytest
from cryptoswarm.risk.guards import SignalGuard, GuardResult
from cryptoswarm.bus.messages import Signal
from cryptoswarm.config.settings import Settings, RiskConfig


def make_settings(**kwargs) -> Settings:
    risk = RiskConfig(**kwargs)
    return Settings(symbols="BTCUSDT,ETHUSDT", risk=risk)


def make_signal(**kwargs) -> Signal:
    defaults = dict(symbol="BTCUSDT", side="LONG", size_usd=100.0,
                    sl=45000.0, tp=55000.0, leverage=5)
    defaults.update(kwargs)
    return Signal(**defaults)


def test_valid_signal_passes():
    guard = SignalGuard(settings=make_settings(), open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal())
    assert result.allowed is True


def test_too_many_positions():
    guard = SignalGuard(settings=make_settings(max_concurrent_positions=5),
                       open_positions=5, current_equity=1000.0)
    result = guard.check(make_signal())
    assert result.allowed is False
    assert "concurrent" in result.reason.lower()


def test_position_size_too_large():
    # 10% of $1000 = $100 max. size_usd=101 should fail.
    guard = SignalGuard(settings=make_settings(), open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal(size_usd=101.0))
    assert result.allowed is False
    assert "size" in result.reason.lower()


def test_leverage_too_high():
    guard = SignalGuard(settings=make_settings(max_leverage=5),
                       open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal(leverage=10))
    assert result.allowed is False
    assert "leverage" in result.reason.lower()


def test_unknown_symbol():
    guard = SignalGuard(settings=make_settings(), open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal(symbol="UNKNOWN"))
    assert result.allowed is False
    assert "symbol" in result.reason.lower()


def test_sl_zero_rejected():
    guard = SignalGuard(settings=make_settings(), open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal(sl=0.0))
    assert result.allowed is False


def test_tp_zero_rejected():
    guard = SignalGuard(settings=make_settings(), open_positions=0, current_equity=1000.0)
    result = guard.check(make_signal(tp=0.0))
    assert result.allowed is False
