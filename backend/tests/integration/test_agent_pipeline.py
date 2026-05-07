"""Integration test: Director broadcasts AnalyzeRequest; verify bus roundtrip.

Requires Valkey on localhost:6379 (make up). Uses DB 1 for isolation.
Auto-skips when infrastructure is unavailable.
"""
import asyncio
import pytest
from cryptoswarm.bus.messages import AnalyzeRequest, QuantResult


async def test_analyze_request_roundtrip(bus):
    """Publish an AnalyzeRequest on agent.analyze.BTCUSDT and receive it."""
    received = []

    async def collector():
        async for topic, data in bus.psubscribe("agent.analyze.*"):
            received.append((topic, data))
            break

    task = asyncio.create_task(collector())
    await asyncio.sleep(0.1)

    req = AnalyzeRequest(symbol="BTCUSDT")
    await bus.publish("agent.analyze.BTCUSDT", req)
    await asyncio.wait_for(task, timeout=3.0)

    assert len(received) == 1
    topic, data = received[0]
    assert topic == "agent.analyze.BTCUSDT"
    restored = AnalyzeRequest.model_validate_json(data)
    assert restored.symbol == "BTCUSDT"
    assert restored.correlation_id == req.correlation_id


async def test_quant_result_roundtrip(bus):
    """Publish a QuantResult on agent.result.quant.BTCUSDT and verify recovery."""
    received = []

    async def collector():
        async for topic, data in bus.psubscribe("agent.result.quant.*"):
            received.append((topic, data))
            break

    task = asyncio.create_task(collector())
    await asyncio.sleep(0.1)

    result = QuantResult(
        symbol="BTCUSDT",
        regime="trending_up",
        signal_strength=0.7,
        confidence=0.8,
        reasoning="test",
        indicators={"rsi": 62.0, "close": 65000.0},
    )
    await bus.publish("agent.result.quant.BTCUSDT", result)
    await asyncio.wait_for(task, timeout=3.0)

    assert len(received) == 1
    restored = QuantResult.model_validate_json(received[0][1])
    assert restored.regime == "trending_up"
    assert restored.correlation_id == result.correlation_id
