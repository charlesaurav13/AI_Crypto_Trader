from datetime import datetime, timezone
from cryptoswarm.bus.messages import (
    MarketTick, Signal, PositionUpdate, SystemHeartbeat,
    MarkPrice, FundingUpdate, OpenInterestUpdate,
    LiquidationEvent, BookTicker, RiskVeto, TradeExecuted,
    CircuitTripped,
    AnalyzeRequest, QuantResult, RiskResult,
    SentimentResult, PortfolioResult, DirectorDecision,
)


def test_market_tick_round_trip():
    tick = MarketTick(
        symbol="BTCUSDT",
        interval="1m",
        open=50000.0,
        high=51000.0,
        low=49500.0,
        close=50500.0,
        volume=123.45,
        is_closed=True,
    )
    data = tick.model_dump()
    tick2 = MarketTick(**data)
    assert tick2.symbol == "BTCUSDT"
    assert tick2.close == 50500.0
    assert tick2.is_closed is True


def test_signal_round_trip():
    sig = Signal(
        symbol="ETHUSDT",
        side="LONG",
        size_usd=500.0,
        sl=2800.0,
        tp=3200.0,
        leverage=3,
        reasoning="momentum breakout",
    )
    data = sig.model_dump()
    sig2 = Signal(**data)
    assert sig2.side == "LONG"
    assert sig2.size_usd == 500.0
    assert sig2.reasoning == "momentum breakout"


def test_position_update_with_close():
    pos = PositionUpdate(
        symbol="BTCUSDT",
        side="LONG",
        qty=0.01,
        entry_price=50000.0,
        mark_price=49000.0,
        unrealized_pnl=-10.0,
        isolated_margin=100.0,
        liq_price=45000.0,
        is_closed=True,
        close_reason="sl",
    )
    assert pos.is_closed is True
    assert pos.close_reason == "sl"


def test_all_messages_have_correlation_id():
    msgs = [
        MarketTick(symbol="BTCUSDT", interval="1m", open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, is_closed=False),
        MarkPrice(symbol="BTCUSDT", mark_price=50000.0, index_price=50001.0),
        FundingUpdate(symbol="BTCUSDT", funding_time=datetime.now(timezone.utc), rate=0.0001),
        OpenInterestUpdate(symbol="BTCUSDT", open_interest=12345.0),
        LiquidationEvent(symbol="BTCUSDT", side="BUY", price=49000.0, qty=0.5),
        BookTicker(symbol="BTCUSDT", best_bid=49999.0, best_ask=50001.0),
        Signal(symbol="BTCUSDT", side="LONG", size_usd=100.0, sl=48000.0, tp=52000.0, leverage=2),
        RiskVeto(original_correlation_id="abc", reason="daily loss", breaker_name="DailyLossBreaker"),
        TradeExecuted(original_correlation_id="abc", symbol="BTCUSDT", side="LONG",
                      qty=0.01, entry_price=50000.0, leverage=2, sl=48000.0, tp=52000.0, fees=0.5),
        PositionUpdate(symbol="BTCUSDT", side="LONG", qty=0.01, entry_price=50000.0,
                       mark_price=50500.0, unrealized_pnl=5.0, isolated_margin=100.0, liq_price=45000.0),
        CircuitTripped(breaker_name="DailyLossBreaker", value=-35.0, threshold=-30.0),
        SystemHeartbeat(process_id=12345),
    ]
    for msg in msgs:
        assert hasattr(msg, "correlation_id"), f"{type(msg).__name__} missing correlation_id"
        assert isinstance(msg.correlation_id, str)
        assert len(msg.correlation_id) == 36  # UUID4 string length
        assert hasattr(msg, "ts")
        assert hasattr(msg, "schema_version")
        assert msg.schema_version == 1


def test_analyze_request_defaults():
    msg = AnalyzeRequest(symbol="BTCUSDT")
    assert msg.symbol == "BTCUSDT"
    assert msg.interval == "1m"
    assert msg.lookback_bars == 100
    assert msg.correlation_id


def test_quant_result_roundtrip():
    msg = QuantResult(
        symbol="BTCUSDT",
        regime="trending_up",
        signal_strength=0.75,
        confidence=0.8,
        reasoning="EMA cross bullish",
        indicators={"rsi": 62.0},
    )
    restored = QuantResult.model_validate_json(msg.model_dump_json())
    assert restored.regime == "trending_up"
    assert restored.signal_strength == 0.75


def test_director_decision_hold():
    msg = DirectorDecision(
        symbol="ETHUSDT",
        action="hold",
        side="LONG",
        confidence=0.3,
        size_pct=0.0,
        sl_pct=0.02,
        tp_pct=0.04,
        entry_price=3000.0,
        reasoning="mixed signals",
        quant_summary="ranging",
        risk_summary="low kelly",
        sentiment_summary="neutral",
        portfolio_summary="approved",
    )
    assert msg.action == "hold"
    assert msg.size_pct == 0.0
