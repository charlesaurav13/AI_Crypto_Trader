"""Tests for Phase 3 bus message types."""
from cryptoswarm.bus.messages import NewsSentimentResult, MLSignal


def test_news_sentiment_result_serialises():
    msg = NewsSentimentResult(
        symbol="BTCUSDT",
        score=0.45,
        article_count=5,
        top_headline="BTC breaks resistance",
        source_breakdown={"coindesk": 2, "reddit": 3},
    )
    data = msg.model_dump_json()
    restored = NewsSentimentResult.model_validate_json(data)
    assert restored.symbol == "BTCUSDT"
    assert restored.score == 0.45
    assert restored.article_count == 5


def test_ml_signal_serialises():
    msg = MLSignal(
        symbol="ETHUSDT",
        correlation_id="test-cid",
        regime_pred="trending_up",
        direction_pred="up",
        short_direction="up",
        size_adjustment="scale_up",
        confidence=0.75,
        reasoning="XGBoost+LSTM agree bullish",
    )
    data = msg.model_dump_json()
    restored = MLSignal.model_validate_json(data)
    assert restored.regime_pred == "trending_up"
    assert restored.size_adjustment == "scale_up"


def test_ml_signal_neutral_defaults():
    msg = MLSignal(
        symbol="SOLUSDT",
        correlation_id="cid",
        regime_pred="ranging",
        direction_pred="down",
        short_direction="down",
        size_adjustment="hold",
        confidence=0.0,
        reasoning="model not trained yet",
    )
    assert msg.confidence == 0.0
    assert msg.size_adjustment == "hold"
