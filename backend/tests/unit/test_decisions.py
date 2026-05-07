import pytest
from unittest.mock import AsyncMock, MagicMock
from cryptoswarm.storage.decisions import DecisionWriter
from cryptoswarm.bus.messages import DirectorDecision


async def test_decision_writer_insert():
    mock_pool = MagicMock()
    mock_pool.execute = AsyncMock(return_value="INSERT 0 1")

    writer = DecisionWriter.__new__(DecisionWriter)
    writer._pool = mock_pool

    decision = DirectorDecision(
        symbol="BTCUSDT",
        action="buy",
        side="LONG",
        confidence=0.8,
        size_pct=0.05,
        sl_pct=0.02,
        tp_pct=0.04,
        entry_price=65000.0,
        reasoning="Strong bullish regime with risk approval",
        quant_summary="trending_up, strength=0.75",
        risk_summary="kelly=0.05",
        sentiment_summary="score=0.2",
        portfolio_summary="approved, penalty=1.0",
    )

    await writer.insert(decision)
    mock_pool.execute.assert_called_once()
    call_args = mock_pool.execute.call_args[0]
    assert "INSERT INTO decisions" in call_args[0]
    assert decision.correlation_id in call_args


async def test_decision_writer_requires_connection():
    writer = DecisionWriter("postgresql://localhost/test")
    decision = DirectorDecision(
        symbol="BTCUSDT", action="hold", side="LONG", confidence=0.1,
        size_pct=0.0, sl_pct=0.02, tp_pct=0.04, entry_price=65000.0,
        reasoning="test", quant_summary="", risk_summary="",
        sentiment_summary="", portfolio_summary="",
    )
    with pytest.raises(AssertionError):
        await writer.insert(decision)
