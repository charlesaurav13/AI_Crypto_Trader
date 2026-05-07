"""Risk Agent — evaluates account state and recommends position sizing via Kelly criterion."""
from __future__ import annotations

import logging

from cryptoswarm.agents.llm import LLMClient
from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import AnalyzeRequest, RiskResult
from cryptoswarm.config.settings import Settings

logger = logging.getLogger(__name__)

_SYSTEM = """You are a risk manager for a Binance USDM perpetual futures paper trading system.
Apply Kelly criterion principles to recommend position sizing. Never recommend more than 10% of account balance.
Return conservative sizing — the system is learning, losses are real."""

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "kelly_fraction": {
            "type": "number",
            "description": "Recommended fraction of account to allocate (0.0–0.10)",
        },
        "max_loss_usdt": {
            "type": "number",
            "description": "Maximum acceptable loss in USD for this trade",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief reasoning for the position size recommendation",
        },
    },
    "required": ["kelly_fraction", "max_loss_usdt", "reasoning"],
}


class RiskAgent:
    def __init__(self, bus: BusClient, llm: LLMClient, settings: Settings) -> None:
        self._bus = bus
        self._llm = llm
        self._cfg = settings

    async def run(self) -> None:
        async for topic, data in self._bus.psubscribe("agent.analyze.*"):
            req = AnalyzeRequest.model_validate_json(data)
            try:
                await self._handle(req)
            except Exception as exc:
                logger.error("RiskAgent error for %s: %s", req.symbol, exc)

    async def _handle(self, req: AnalyzeRequest) -> None:
        risk = self._cfg.risk
        prompt = (
            f"Symbol: {req.symbol}\n"
            f"Account balance: ${risk.starting_balance_usd:.2f}\n"
            f"Max position: {risk.max_position_pct * 100:.0f}% of balance\n"
            f"Max leverage: {risk.max_leverage}x\n"
            f"Daily loss limit: {risk.daily_loss_pct * 100:.0f}%\n"
            f"Max drawdown limit: {risk.max_drawdown_pct * 100:.0f}%\n"
            f"Recommend Kelly fraction and max loss in USD for a new {req.symbol} position."
        )

        raw = await self._llm.ask(
            system=_SYSTEM,
            prompt=prompt,
            tool_name="risk_assessment",
            tool_schema=_TOOL_SCHEMA,
        )

        # Clamp kelly_fraction to configured max
        kelly = min(float(raw["kelly_fraction"]), risk.max_position_pct)

        result = RiskResult(
            symbol=req.symbol,
            correlation_id=req.correlation_id,
            kelly_fraction=kelly,
            max_loss_usdt=float(raw["max_loss_usdt"]),
            reasoning=raw["reasoning"],
        )
        await self._bus.publish(f"agent.result.risk.{req.symbol}", result)
        logger.info(
            "RiskAgent: %s kelly=%.3f max_loss=$%.2f",
            req.symbol, result.kelly_fraction, result.max_loss_usdt,
        )
