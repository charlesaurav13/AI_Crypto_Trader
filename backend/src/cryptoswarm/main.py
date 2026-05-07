import asyncio
import logging
import os
import signal as _signal
import uvicorn

from cryptoswarm.config.settings import get_settings
from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import SystemHeartbeat
from cryptoswarm.feed.rest_client import BinanceRestClient
from cryptoswarm.feed.handler import FrameHandler
from cryptoswarm.feed.ws_client import FeedManager
from cryptoswarm.storage.timescale import TimescaleWriter
from cryptoswarm.storage.postgres import PostgresWriter
from cryptoswarm.storage.subscriber import StorageSubscriber
from cryptoswarm.storage.decisions import DecisionWriter
from cryptoswarm.papertrade.engine import PaperTradeEngine
from cryptoswarm.agents.llm import LLMClient
from cryptoswarm.agents.quant import QuantAgent
from cryptoswarm.agents.risk_agent import RiskAgent
from cryptoswarm.agents.sentiment import SentimentAgent
from cryptoswarm.agents.portfolio import PortfolioAgent
from cryptoswarm.agents.director import DirectorAgent
from cryptoswarm.api.app import create_app
from cryptoswarm.api import deps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def heartbeat_loop(bus: BusClient, interval_s: int) -> None:
    while True:
        await bus.publish("system.heartbeat", SystemHeartbeat(process_id=os.getpid()))
        await asyncio.sleep(interval_s)


async def main() -> None:
    cfg = get_settings()
    logger.info("Starting CryptoSwarm Foundation (paper_trading=%s)", cfg.paper_trading)

    # --- Boot dependencies ---
    bus = BusClient(cfg.valkey_url)
    await bus.connect()
    logger.info("Bus connected: %s", cfg.valkey_url)

    ts_writer = TimescaleWriter(cfg.timescale_dsn)
    await ts_writer.connect()
    pg_writer = PostgresWriter(cfg.postgres_dsn)
    await pg_writer.connect()
    logger.info("Storage connected")

    rest = BinanceRestClient(cfg.binance_api_key, cfg.binance_api_secret, cfg.binance_testnet)
    await rest.connect()

    decisions_writer = DecisionWriter(cfg.postgres_dsn)
    await decisions_writer.connect()
    logger.info("DecisionWriter connected")

    llm = LLMClient(cfg)   # provider selected by cfg.llm_provider

    handler = FrameHandler(bus)
    feed = FeedManager(cfg, handler, rest)
    storage_sub = StorageSubscriber(bus, ts_writer, pg_writer)
    engine = PaperTradeEngine(bus, cfg)

    quant_agent     = QuantAgent(bus=bus, ts=ts_writer, llm=llm)
    risk_agent      = RiskAgent(bus=bus, llm=llm, settings=cfg)
    sentiment_agent = SentimentAgent(bus=bus)
    portfolio_agent = PortfolioAgent(bus=bus, llm=llm)
    director        = DirectorAgent(bus=bus, llm=llm, decisions=decisions_writer, settings=cfg)

    # Wire API deps
    deps.set_deps(bus=bus, pg=pg_writer, ts=ts_writer, engine=engine)

    # --- Launch all tasks ---
    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info"))

    tasks = [
        asyncio.create_task(feed.run(), name="feed"),
        asyncio.create_task(storage_sub.run(), name="storage"),
        asyncio.create_task(engine.run(), name="engine"),
        asyncio.create_task(quant_agent.run(), name="quant_agent"),
        asyncio.create_task(risk_agent.run(), name="risk_agent"),
        asyncio.create_task(sentiment_agent.run(), name="sentiment_agent"),
        asyncio.create_task(portfolio_agent.run(), name="portfolio_agent"),
        asyncio.create_task(director.run(), name="director"),
        asyncio.create_task(
            heartbeat_loop(bus, cfg.risk.heartbeat_interval_s), name="heartbeat"
        ),
        asyncio.create_task(server.serve(), name="api"),
    ]

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_running_loop()
    for sig in (_signal.SIGTERM, _signal.SIGINT):
        loop.add_signal_handler(sig, lambda: [t.cancel() for t in tasks])

    logger.info("All tasks started. CryptoSwarm is running.")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutdown initiated")
    finally:
        await bus.close()
        await ts_writer.close()
        await pg_writer.close()
        await decisions_writer.close()
        await rest.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
