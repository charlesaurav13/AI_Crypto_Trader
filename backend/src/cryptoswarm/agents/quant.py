"""Quant Agent — computes technical indicators and asks Claude for regime + signal strength."""
from __future__ import annotations

import json
import logging

from cryptoswarm.agents.indicators import compute_indicators
from cryptoswarm.agents.llm import LLMClient
from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import AnalyzeRequest, QuantResult
from cryptoswarm.storage.timescale import TimescaleWriter

logger = logging.getLogger(__name__)

_SYSTEM = """You are a quantitative trading analyst for Binance USDM perpetual futures.
Given technical indicator values, identify the market regime and signal strength.
Be concise and data-driven. Signal strength: -1.0 = strong sell, 0.0 = neutral, +1.0 = strong buy."""

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "regime": {
            "type": "string",
            "enum": ["trending_up", "trending_down", "ranging", "volatile"],
            "description": "Current market regime",
        },
        "signal_strength": {
            "type": "number",
            "description": "Signal strength from -1.0 (strong sell) to 1.0 (strong buy)",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the analysis from 0.0 to 1.0",
        },
        "reasoning": {
            "type": "string",
            "description": "One or two sentence explanation referencing specific indicator values",
        },
    },
    "required": ["regime", "signal_strength", "confidence", "reasoning"],
}


class QuantAgent:
    def __init__(self, bus: BusClient, ts: TimescaleWriter, llm: LLMClient) -> None:
        self._bus = bus
        self._ts = ts
        self._llm = llm

    async def run(self) -> None:
        """Subscribe to agent.analyze.* and handle each request."""
        async for topic, data in self._bus.psubscribe("agent.analyze.*"):
            req = AnalyzeRequest.model_validate_json(data)
            try:
                await self._handle(req)
            except Exception as exc:
                logger.error("QuantAgent error for %s: %s", req.symbol, exc)

    async def _handle(self, req: AnalyzeRequest) -> None:
        rows = await self._ts.fetch_klines(req.symbol, req.lookback_bars)
        if len(rows) < 60:
            logger.warning("QuantAgent: insufficient klines for %s (%d bars)", req.symbol, len(rows))
            return

        indicators = compute_indicators(rows)
        prompt = (
            f"Symbol: {req.symbol}\n"
            f"Interval: {req.interval}\n"
            f"Indicators (last bar):\n{json.dumps(indicators, indent=2)}"
        )

        raw = await self._llm.ask(
            system=_SYSTEM,
            prompt=prompt,
            tool_name="quant_analysis",
            tool_schema=_TOOL_SCHEMA,
        )

        result = QuantResult(
            symbol=req.symbol,
            correlation_id=req.correlation_id,
            regime=raw["regime"],
            signal_strength=float(raw["signal_strength"]),
            confidence=float(raw["confidence"]),
            reasoning=raw["reasoning"],
            indicators=indicators,
        )
        await self._bus.publish(f"agent.result.quant.{req.symbol}", result)
        logger.info(
            "QuantAgent: %s regime=%s strength=%.2f",
            req.symbol, result.regime, result.signal_strength,
        )
