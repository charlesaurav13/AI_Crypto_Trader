"""
Integration test fixtures.
Requires docker-compose infra: make up
"""
import asyncio
import pytest
from cryptoswarm.config.settings import Settings
from cryptoswarm.bus.client import BusClient
from cryptoswarm.storage.postgres import PostgresWriter
from cryptoswarm.papertrade.engine import PaperTradeEngine
from cryptoswarm.storage.subscriber import StorageSubscriber
from cryptoswarm.storage.timescale import TimescaleWriter

TS_DSN = "postgresql://postgres:postgres@localhost:5432/cryptoswarm_ts"
PG_DSN = "postgresql://postgres:postgres@localhost:5433/cryptoswarm"
VALKEY_URL = "redis://localhost:6379"


def _infra_available() -> bool:
    """True if all three infra services are reachable."""
    import socket
    checks = [("localhost", 6379), ("localhost", 5432), ("localhost", 5433)]
    for host, port in checks:
        try:
            s = socket.create_connection((host, port), timeout=1)
            s.close()
        except OSError:
            return False
    return True


@pytest.fixture(scope="session", autouse=True)
def require_infra():
    if not _infra_available():
        pytest.skip("Integration infra not reachable — run `make up` first")


@pytest.fixture
async def bus():
    b = BusClient(VALKEY_URL)
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
async def pg():
    w = PostgresWriter(PG_DSN)
    await w.connect()
    yield w
    await w.close()


@pytest.fixture
async def settings():
    return Settings(
        valkey_url=VALKEY_URL,
        timescale_dsn=TS_DSN,
        postgres_dsn=PG_DSN,
        paper_trading=True,
        symbols="BTCUSDT,ETHUSDT",
    )
