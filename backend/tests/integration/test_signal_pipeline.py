"""
Integration tests for the full signal pipeline.
Requires: Valkey at :6379, TimescaleDB at :5432, PostgreSQL at :5433
Run: cd backend && pytest tests/integration/ -v
"""
import asyncio
import json
import pytest

from cryptoswarm.bus.messages import (
    Signal, TradeExecuted, RiskVeto, PositionUpdate, MarkPrice,
)
from cryptoswarm.papertrade.engine import PaperTradeEngine
from cryptoswarm.storage.subscriber import StorageSubscriber
from cryptoswarm.storage.timescale import TimescaleWriter

TS_DSN = "postgresql://postgres:postgres@localhost:5432/cryptoswarm_ts"

pytestmark = pytest.mark.asyncio


async def test_valid_signal_opens_position(bus, pg, settings):
    engine = PaperTradeEngine(bus, settings)
    ts = TimescaleWriter(TS_DSN)
    await ts.connect()
    storage = StorageSubscriber(bus, ts, pg)

    received_trades: list[str] = []

    async def collector():
        async for _, data in bus.subscribe("trade.executed"):
            received_trades.append(data)
            break

    tasks = [
        asyncio.create_task(engine.run()),
        asyncio.create_task(storage.run()),
        asyncio.create_task(collector()),
    ]
    await asyncio.sleep(0.1)

    sig = Signal(
        symbol="BTCUSDT", side="LONG", size_usd=100.0,
        sl=45000.0, tp=55000.0, leverage=5,
        reasoning=json.dumps({"entry": 50000.0}),
    )
    await bus.publish("signal.execute", sig)
    await asyncio.wait_for(asyncio.gather(tasks[2]), timeout=5.0)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await ts.close()

    assert len(received_trades) == 1
    executed = TradeExecuted.model_validate_json(received_trades[0])
    assert executed.symbol == "BTCUSDT"
    assert executed.entry_price == 50000.0

    # Verify audit row in PostgreSQL
    rows = await pg._pool.fetch(
        "SELECT * FROM trades WHERE correlation_id=$1", sig.correlation_id
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTCUSDT"


async def test_sl_triggers_position_close(bus, pg, settings):
    engine = PaperTradeEngine(bus, settings)
    closed: list[str] = []

    async def collector():
        async for _, data in bus.subscribe("position.update"):
            upd = PositionUpdate.model_validate_json(data)
            if upd.is_closed:
                closed.append(data)
                break

    tasks = [asyncio.create_task(engine.run()), asyncio.create_task(collector())]
    await asyncio.sleep(0.1)

    # Open a position
    sig = Signal(
        symbol="ETHUSDT", side="LONG", size_usd=100.0,
        sl=1800.0, tp=2400.0, leverage=5,
        reasoning=json.dumps({"entry": 2000.0}),
    )
    await bus.publish("signal.execute", sig)
    await asyncio.sleep(0.3)

    # Simulate mark price dropping below SL
    await bus.publish(
        "market.mark.ETHUSDT",
        MarkPrice(symbol="ETHUSDT", mark_price=1799.0, index_price=1798.0),
    )
    await asyncio.wait_for(asyncio.gather(tasks[1]), timeout=5.0)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert len(closed) == 1
    upd = PositionUpdate.model_validate_json(closed[0])
    assert upd.close_reason == "sl"


async def test_daily_loss_breaker_trips(bus, settings):
    engine = PaperTradeEngine(bus, settings)
    vetoes: list[str] = []

    async def veto_collector():
        async for _, data in bus.subscribe("risk.veto"):
            vetoes.append(data)
            break

    tasks = [asyncio.create_task(engine.run()), asyncio.create_task(veto_collector())]
    await asyncio.sleep(0.1)

    # Force-trip the daily loss breaker
    engine._daily_loss.update_pnl(-100.0)  # well past -3% of $1000
    assert engine._daily_loss.is_tripped()

    # Signal should now be vetoed
    sig = Signal(
        symbol="BTCUSDT", side="LONG", size_usd=50.0,
        sl=45000.0, tp=55000.0, leverage=5,
    )
    await bus.publish("signal.execute", sig)
    await asyncio.wait_for(asyncio.gather(tasks[1]), timeout=5.0)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert len(vetoes) >= 1
    veto = RiskVeto.model_validate_json(vetoes[0])
    assert "circuit" in veto.reason.lower()
