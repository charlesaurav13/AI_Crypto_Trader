import pytest
from cryptoswarm.papertrade.math import (
    calc_qty, calc_liq_price, calc_unrealized_pnl,
    calc_realized_pnl, calc_entry_fee, calc_funding,
    calc_isolated_margin, calc_exit_fee,
)

MMR = 0.004  # 0.4% maintenance margin rate


# --- calc_qty ---
def test_calc_qty_long():
    # $100 notional at $50,000/BTC → 0.002 BTC
    assert calc_qty(size_usd=100.0, entry_price=50_000.0) == pytest.approx(0.002, rel=1e-6)


def test_calc_qty_zero_price_raises():
    with pytest.raises(ValueError, match="entry_price must be > 0"):
        calc_qty(100.0, 0.0)


# --- calc_isolated_margin ---
def test_isolated_margin():
    # $100 notional / 5x = $20 margin
    assert calc_isolated_margin(size_usd=100.0, leverage=5) == pytest.approx(20.0)


# --- calc_liq_price ---
def test_liq_price_long():
    # entry=50000, leverage=5, mmr=0.004
    # liq = 50000 * (1 - 1/5 + 0.004) = 50000 * 0.804 = 40200
    expected = 50_000.0 * (1 - 1 / 5 + MMR)
    result = calc_liq_price(entry_price=50_000.0, side="LONG", leverage=5, mmr=MMR)
    assert result == pytest.approx(expected, rel=1e-6)


def test_liq_price_short():
    # liq = 50000 * (1 + 1/5 - 0.004) = 50000 * 1.196 = 59800
    expected = 50_000.0 * (1 + 1 / 5 - MMR)
    result = calc_liq_price(entry_price=50_000.0, side="SHORT", leverage=5, mmr=MMR)
    assert result == pytest.approx(expected, rel=1e-6)


# --- calc_unrealized_pnl ---
def test_unrealized_pnl_long_profit():
    # qty=0.002, entry=50000, mark=51000 → pnl = 0.002 * 1000 = 2.0
    assert calc_unrealized_pnl(qty=0.002, entry_price=50_000.0, mark_price=51_000.0, side="LONG") == pytest.approx(2.0)


def test_unrealized_pnl_short_profit():
    # qty=0.002, entry=50000, mark=49000 → pnl = 0.002 * 1000 = 2.0
    assert calc_unrealized_pnl(qty=0.002, entry_price=50_000.0, mark_price=49_000.0, side="SHORT") == pytest.approx(2.0)


def test_unrealized_pnl_long_loss():
    assert calc_unrealized_pnl(qty=0.002, entry_price=50_000.0, mark_price=49_000.0, side="LONG") == pytest.approx(-2.0)


# --- calc_realized_pnl ---
def test_realized_pnl_long():
    assert calc_realized_pnl(qty=0.002, entry_price=50_000.0, exit_price=51_000.0, side="LONG") == pytest.approx(2.0)


def test_realized_pnl_short():
    assert calc_realized_pnl(qty=0.002, entry_price=50_000.0, exit_price=49_000.0, side="SHORT") == pytest.approx(2.0)


# --- calc_entry_fee ---
def test_entry_fee_taker():
    # $100 notional * 0.04% = $0.04
    assert calc_entry_fee(size_usd=100.0, taker_rate=0.0004) == pytest.approx(0.04)


# --- calc_exit_fee ---
def test_exit_fee():
    # qty=0.002, exit=51000, taker=0.0004 → 0.002 * 51000 * 0.0004 = 0.0408
    assert calc_exit_fee(qty=0.002, exit_price=51_000.0, taker_rate=0.0004) == pytest.approx(0.0408)


# --- calc_funding ---
def test_funding_positive_rate_long_pays():
    # qty=0.002 BTC, mark=50000, rate=0.0001 → funding = 0.002*50000*0.0001 = 0.01 (long pays)
    assert calc_funding(qty=0.002, mark_price=50_000.0, rate=0.0001, side="LONG") == pytest.approx(-0.01)


def test_funding_positive_rate_short_receives():
    assert calc_funding(qty=0.002, mark_price=50_000.0, rate=0.0001, side="SHORT") == pytest.approx(0.01)


def test_funding_negative_rate_long_receives():
    assert calc_funding(qty=0.002, mark_price=50_000.0, rate=-0.0001, side="LONG") == pytest.approx(0.01)
