"""Portfolio Agent — tracks open positions and assesses correlation risk for proposed trades."""
from __future__ import annotations

import asyncio
import logging

from cryptoswarm.agents.llm import LLMClient
from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import AnalyzeRequest, PortfolioResult, PositionUpdate

logger = logging.getLogger(__name__)

_SYSTEM = """You are a portfolio risk manager. Given a list of currently open futures positions
and a proposed new trade symbol, assess correlation risk and approve or reject the trade.
Penalize trades that add concentration risk (same sector, high correlation)."""

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {
            "type": "boolean",
            "description": "Whether to approve the proposed trade",
        },
        "correlation_penalty": {
            "type": "number",
            "description": "Size multiplier 0.0–1.0 (1.0 = no penalty, 0.5 = halve the size)",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the portfolio risk assessment",
        },
    },
    "required": ["approved", "correlation_penalty", "reasoning"],
}


class PortfolioAgent:
    def __init__(self, bus: BusClient, llm: LLMClient) -> None:
        self._bus = bus
        self._llm = llm
        self._positions: dict[str, dict] = {}  # symbol → position snapshot

    def _on_position_update(self, update: PositionUpdate) -> None:
        if update.is_closed:
            self._positions.pop(update.symbol, None)
        else:
            self._positions[update.symbol] = {
                "side":           update.side,
                "qty":            update.qty,
                "entry_price":    update.entry_price,
                "unrealized_pnl": update.unrealized_pnl,
            }

    async def run(self) -> None:
        await asyncio.gather(
            self._listen_positions(),
            self._listen_analyze(),
        )

    async def _listen_positions(self) -> None:
        # PaperTradeEngine publishes on "position.update" (no symbol suffix)
        async for topic, data in self._bus.subscribe("position.update"):
            try:
                update = PositionUpdate.model_validate_json(data)
                self._on_position_update(update)
            except Exception as exc:
                logger.error("PortfolioAgent position update error: %s", exc)

    async def _listen_analyze(self) -> None:
        async for topic, data in self._bus.psubscribe("agent.analyze.*"):
            req = AnalyzeRequest.model_validate_json(data)
            try:
                await self._handle(req)
            except Exception as exc:
                logger.error("PortfolioAgent error for %s: %s", req.symbol, exc)

    async def _handle(self, req: AnalyzeRequest) -> None:
        open_positions = [
            f"{sym}: {pos['side']} qty={pos['qty']} pnl=${pos['unrealized_pnl']:.2f}"
            for sym, pos in self._positions.items()
            if sym != req.symbol
        ]

        prompt = (
            f"Proposed trade: {req.symbol}\n"
            f"Open positions ({len(open_positions)}):\n"
            + ("\n".join(open_positions) if open_positions else "  (none)")
        )

        raw = await self._llm.ask(
            system=_SYSTEM,
            prompt=prompt,
            tool_name="portfolio_assessment",
            tool_schema=_TOOL_SCHEMA,
        )

        result = PortfolioResult(
            symbol=req.symbol,
            correlation_id=req.correlation_id,
            approved=bool(raw["approved"]),
            correlation_penalty=float(raw["correlation_penalty"]),
            reasoning=raw["reasoning"],
        )
        await self._bus.publish(f"agent.result.portfolio.{req.symbol}", result)
        logger.info(
            "PortfolioAgent: %s approved=%s penalty=%.2f",
            req.symbol, result.approved, result.correlation_penalty,
        )
