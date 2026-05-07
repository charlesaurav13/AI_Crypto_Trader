"""Bus client tests — require Valkey running at localhost:6379 (make up).

Tests auto-skip when Valkey is unreachable so the unit suite stays green
without Docker running.
"""
import asyncio
import pytest
import valkey.asyncio as valkey_aio

from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import MarketTick


# ---------------------------------------------------------------------------
# Skip fixture — skips entire module when Valkey is not reachable
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def valkey_available():
    """Return True only if a Valkey/Redis instance answers at localhost:6379."""
    import socket
    try:
        s = socket.create_connection(("localhost", 6379), timeout=1)
        s.close()
        return True
    except OSError:
        return False


@pytest.fixture(autouse=True)
def require_valkey(valkey_available):
    if not valkey_available:
        pytest.skip("Valkey not reachable — run `make up` first")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_publish_subscribe_roundtrip():
    client = BusClient("redis://localhost:6379")
    await client.connect()

    received: list[tuple[str, str]] = []

    async def collector():
        async for topic, data in client.subscribe("market.tick.BTCUSDT.1m"):
            received.append((topic, data))
            break  # stop after first message

    task = asyncio.create_task(collector())
    await asyncio.sleep(0.1)  # let subscribe settle

    msg = MarketTick(
        symbol="BTCUSDT", interval="1m",
        open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, is_closed=True,
    )
    await client.publish("market.tick.BTCUSDT.1m", msg)
    await asyncio.wait_for(task, timeout=3.0)

    assert len(received) == 1
    topic, data = received[0]
    assert topic == "market.tick.BTCUSDT.1m"
    restored = MarketTick.model_validate_json(data)
    assert restored.symbol == "BTCUSDT"
    assert restored.correlation_id == msg.correlation_id

    await client.close()


async def test_pattern_subscribe():
    client = BusClient("redis://localhost:6379")
    await client.connect()
    received = []

    async def collector():
        async for topic, data in client.psubscribe("market.tick.*"):
            received.append(topic)
            break

    task = asyncio.create_task(collector())
    await asyncio.sleep(0.1)

    msg = MarketTick(
        symbol="ETHUSDT", interval="1m",
        open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, is_closed=False,
    )
    await client.publish("market.tick.ETHUSDT.1m", msg)
    await asyncio.wait_for(task, timeout=3.0)
    assert "market.tick.ETHUSDT.1m" in received

    await client.close()
