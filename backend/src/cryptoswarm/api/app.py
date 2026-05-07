from fastapi import FastAPI
from cryptoswarm.api.routes import health, positions, trades, circuit, signal, sse


def create_app() -> FastAPI:
    app = FastAPI(title="CryptoSwarm", version="0.1.0")
    app.include_router(health.router)
    app.include_router(positions.router)
    app.include_router(trades.router)
    app.include_router(circuit.router)
    app.include_router(signal.router)
    app.include_router(sse.router)
    return app
