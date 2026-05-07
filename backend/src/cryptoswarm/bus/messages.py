from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field


def _cid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BaseMsg(BaseModel):
    correlation_id: str = Field(default_factory=_cid)
    ts: datetime = Field(default_factory=_now)
    schema_version: int = 1


class MarketTick(BaseMsg):
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


class MarkPrice(BaseMsg):
    symbol: str
    mark_price: float
    index_price: float


class FundingUpdate(BaseMsg):
    symbol: str
    funding_time: datetime
    rate: float


class OpenInterestUpdate(BaseMsg):
    symbol: str
    open_interest: float


class LiquidationEvent(BaseMsg):
    symbol: str
    side: Literal["BUY", "SELL"]
    price: float
    qty: float


class BookTicker(BaseMsg):
    symbol: str
    best_bid: float
    best_ask: float


class Signal(BaseMsg):
    symbol: str
    side: Literal["LONG", "SHORT"]
    size_usd: float
    sl: float
    tp: float
    leverage: int
    reasoning: str = ""


class RiskVeto(BaseMsg):
    original_correlation_id: str
    reason: str
    breaker_name: str


class TradeExecuted(BaseMsg):
    original_correlation_id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    qty: float
    entry_price: float
    leverage: int
    sl: float
    tp: float
    fees: float


class PositionUpdate(BaseMsg):
    symbol: str
    side: Literal["LONG", "SHORT"]
    qty: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    isolated_margin: float
    liq_price: float
    is_closed: bool = False
    close_reason: Optional[Literal["sl", "tp", "liq", "manual"]] = None


class CircuitTripped(BaseMsg):
    breaker_name: str
    value: float
    threshold: float


class SystemHeartbeat(BaseMsg):
    process_id: int
