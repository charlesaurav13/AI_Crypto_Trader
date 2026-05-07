import pytest
from cryptoswarm.papertrade.account import Account, OpenPosition


def make_pos(**kwargs) -> OpenPosition:
    defaults = dict(
        symbol="BTCUSDT", side="LONG", qty=0.002,
        entry_price=50_000.0, leverage=5, sl=40_200.0, tp=60_000.0,
        isolated_margin=20.0, liq_price=40_200.0, fees=0.04,
    )
    defaults.update(kwargs)
    return OpenPosition(**defaults)


def test_account_initial_state():
    acc = Account(starting_balance=1000.0)
    assert acc.balance == 1000.0
    assert acc.open_positions == {}
    assert acc.equity == 1000.0


def test_account_open_position():
    acc = Account(starting_balance=1000.0)
    pos = make_pos()
    acc.open(pos)
    assert "BTCUSDT" in acc.open_positions
    assert acc.balance == pytest.approx(1000.0 - 20.0 - 0.04)  # margin + entry fees


def test_account_close_position():
    acc = Account(starting_balance=1000.0)
    pos = make_pos()
    acc.open(pos)
    acc.close("BTCUSDT", exit_price=51_000.0, exit_reason="tp", exit_fees=0.04)
    assert "BTCUSDT" not in acc.open_positions
    # pnl = 0.002 * 1000 = 2.0
    # net = (margin returned) + pnl - exit_fees; entry fees already taken on open
    # balance_after_open = 1000 - 20 - 0.04 = 979.96
    # balance_after_close = 979.96 + 20 + 2.0 - 0.04 = 1001.88 (gross)
    # net_pnl = 2.0 - 0.04 - 0.04 = 1.92
    expected_balance = (1000.0 - 20.0 - 0.04) + 20.0 + 2.0 - 0.04
    assert acc.balance == pytest.approx(expected_balance)


def test_account_equity_includes_unrealized():
    acc = Account(starting_balance=1000.0)
    pos = make_pos()
    acc.open(pos)
    acc.update_mark("BTCUSDT", 51_000.0)
    # unrealized = 0.002 * 1000 = 2.0
    assert acc.equity == pytest.approx(acc.balance + 2.0)


def test_account_funding_applied():
    acc = Account(starting_balance=1000.0)
    pos = make_pos()
    acc.open(pos)
    bal_before = acc.balance
    acc.apply_funding("BTCUSDT", -0.5)  # long pays 0.5
    assert acc.balance == pytest.approx(bal_before - 0.5)


def test_close_short_profit():
    acc = Account(starting_balance=1000.0)
    pos = make_pos(side="SHORT", sl=60_000.0, tp=40_000.0)
    acc.open(pos)
    acc.close("BTCUSDT", exit_price=49_000.0, exit_reason="tp", exit_fees=0.04)
    # pnl = 0.002 * (50000 - 49000) = 2.0
    expected_balance = (1000.0 - 20.0 - 0.04) + 20.0 + 2.0 - 0.04
    assert acc.balance == pytest.approx(expected_balance)
