from cryptoswarm.bus.client import BusClient
from cryptoswarm.storage.postgres import PostgresWriter
from cryptoswarm.storage.timescale import TimescaleWriter
from cryptoswarm.papertrade.engine import PaperTradeEngine

# Module-level singletons set by main.py before app starts
_bus: BusClient | None = None
_pg: PostgresWriter | None = None
_ts: TimescaleWriter | None = None
_engine: PaperTradeEngine | None = None


def set_deps(bus: BusClient, pg: PostgresWriter, ts: TimescaleWriter,
             engine: PaperTradeEngine) -> None:
    global _bus, _pg, _ts, _engine
    _bus = bus
    _pg = pg
    _ts = ts
    _engine = engine


def get_bus() -> BusClient:
    assert _bus, "BusClient not initialised"
    return _bus


def get_pg() -> PostgresWriter:
    assert _pg, "PostgresWriter not initialised"
    return _pg


def get_ts() -> TimescaleWriter:
    assert _ts, "TimescaleWriter not initialised"
    return _ts


def get_engine() -> PaperTradeEngine:
    assert _engine, "PaperTradeEngine not initialised"
    return _engine
