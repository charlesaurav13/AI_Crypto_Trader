"""
Pure functions for paper trade calculations.
No side effects, no I/O — easy to unit-test and safe to audit.
"""
from typing import Literal

Side = Literal["LONG", "SHORT"]


def calc_qty(size_usd: float, entry_price: float) -> float:
    """Base asset quantity for a given USD notional and entry price."""
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    return size_usd / entry_price


def calc_isolated_margin(size_usd: float, leverage: int) -> float:
    """Initial isolated margin = notional / leverage."""
    return size_usd / leverage


def calc_liq_price(
    entry_price: float,
    side: Side,
    leverage: int,
    mmr: float = 0.004,
) -> float:
    """
    Simplified Binance USDM isolated-margin liquidation price.

    LONG:  liq = entry * (1 - 1/leverage + mmr)
    SHORT: liq = entry * (1 + 1/leverage - mmr)
    """
    if side == "LONG":
        return entry_price * (1 - 1 / leverage + mmr)
    else:
        return entry_price * (1 + 1 / leverage - mmr)


def calc_unrealized_pnl(
    qty: float,
    entry_price: float,
    mark_price: float,
    side: Side,
) -> float:
    """Unrealized PnL in USDT."""
    if side == "LONG":
        return qty * (mark_price - entry_price)
    else:
        return qty * (entry_price - mark_price)


def calc_realized_pnl(
    qty: float,
    entry_price: float,
    exit_price: float,
    side: Side,
) -> float:
    """Realized PnL in USDT (before fees/funding)."""
    if side == "LONG":
        return qty * (exit_price - entry_price)
    else:
        return qty * (entry_price - exit_price)


def calc_entry_fee(size_usd: float, taker_rate: float) -> float:
    """Entry fee. Always taker for market orders."""
    return size_usd * taker_rate


def calc_exit_fee(qty: float, exit_price: float, taker_rate: float) -> float:
    """Exit fee for market close."""
    return qty * exit_price * taker_rate


def calc_funding(
    qty: float,
    mark_price: float,
    rate: float,
    side: Side,
) -> float:
    """
    Funding payment for one 8h interval.
    Positive rate → longs pay shorts.
    Returns signed amount from position holder's perspective.
    """
    payment = qty * mark_price * rate
    if side == "LONG":
        return -payment   # long pays when rate > 0
    else:
        return payment    # short receives when rate > 0
