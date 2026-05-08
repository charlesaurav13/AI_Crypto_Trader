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


class AnalyzeRequest(BaseMsg):
    symbol: str
    interval: str = "1m"
    lookback_bars: int = 100


class QuantResult(BaseMsg):
    symbol: str
    regime: Literal["trending_up", "trending_down", "ranging", "volatile"]
    signal_strength: float          # -1.0 (strong sell) to 1.0 (strong buy)
    confidence: float               # 0.0 to 1.0
    reasoning: str
    indicators: dict                # raw indicator values


class RiskResult(BaseMsg):
    symbol: str
    kelly_fraction: float           # 0.0 to 1.0 recommended account fraction
    max_loss_usdt: float
    reasoning: str


class SentimentResult(BaseMsg):
    symbol: str
    score: float                    # -1.0 (extreme fear) to 1.0 (extreme greed)
    source: str                     # "fear_greed_api" | "neutral_fallback"
    summary: str


class PortfolioResult(BaseMsg):
    symbol: str
    approved: bool
    correlation_penalty: float      # 0.0–1.0 multiplier applied to position size
    reasoning: str


class DirectorDecision(BaseMsg):
    symbol: str
    action: Literal["buy", "sell", "hold"]
    side: Literal["LONG", "SHORT"]  # LONG for buy, SHORT for sell, LONG for hold
    confidence: float               # 0.0 to 1.0
    size_pct: float                 # fraction of account balance (0.0 if hold)
    sl_pct: float                   # stop-loss percentage from entry (e.g. 0.02 = 2%)
    tp_pct: float                   # take-profit percentage from entry (e.g. 0.04 = 4%)
    entry_price: float              # last close price used as reference
    reasoning: str
    quant_summary: str
    risk_summary: str
    sentiment_summary: str
    portfolio_summary: str


class NewsSentimentResult(BaseMsg):
    symbol: str
    score: float                  # -1.0 to 1.0 weighted avg
    article_count: int
    top_headline: str
    source_breakdown: dict        # {"coindesk": 3, "reddit": 7, ...}


class MLSignal(BaseMsg):
    symbol: str
    regime_pred: Literal["trending_up", "trending_down", "ranging", "volatile"]
    direction_pred: Literal["up", "down"]       # XGBoost 1h prediction
    short_direction: Literal["up", "down"]      # LSTM 15m prediction
    size_adjustment: Literal["hold", "scale_up", "scale_down"]
    confidence: float                           # 0.0–1.0
    reasoning: str
