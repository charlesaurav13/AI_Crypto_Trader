"""Director Agent — orchestrates all sub-agents and synthesizes final trading decisions.

Cycle (every director_interval_s):
  1. For each symbol: broadcast AnalyzeRequest on agent.analyze.{symbol}
  2. Collect QuantResult + RiskResult + SentimentResult + PortfolioResult
     (matched by correlation_id, timeout = agent_timeout_s)
  3. Call Claude: synthesize → DirectorDecision
  4. If action != hold: publish Signal to signal.execute
  5. Persist DirectorDecision via DecisionWriter
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from cryptoswarm.agents.llm import LLMClient
from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import (
    AnalyzeRequest,
    DirectorDecision,
    PortfolioResult,
    QuantResult,
    RiskResult,
    SentimentResult,
    Signal,
)
from cryptoswarm.config.settings import Settings
from cryptoswarm.storage.decisions import DecisionWriter

logger = logging.getLogger(__name__)

_RESULT_CLASSES = {
    "quant":     QuantResult,
    "risk":      RiskResult,
    "sentiment": SentimentResult,
    "portfolio": PortfolioResult,
}
_REQUIRED_AGENTS = set(_RESULT_CLASSES.keys())

_SYSTEM = """You are the Chief Trading Officer of a multi-agent AI futures trading system.
Sub-agents have analyzed the market from four independent angles. Synthesize their findings
into a single trading decision. Be decisive but conservative — when signals conflict, hold.
Never recommend a trade with confidence below 0.6."""

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "action":     {"type": "string", "enum": ["buy", "sell", "hold"]},
        "side":       {
            "type": "string",
            "enum": ["LONG", "SHORT"],
            "description": "LONG for buy, SHORT for sell, LONG for hold",
        },
        "confidence": {"type": "number", "description": "0.0–1.0"},
        "size_pct":   {"type": "number", "description": "Fraction of balance to allocate, 0.0 if hold"},
        "sl_pct":     {"type": "number", "description": "Stop-loss % from entry (e.g. 0.02 = 2%)"},
        "tp_pct":     {"type": "number", "description": "Take-profit % from entry (e.g. 0.04 = 4%)"},
        "reasoning":  {"type": "string", "description": "2–3 sentence synthesis of all agent inputs"},
    },
    "required": ["action", "side", "confidence", "size_pct", "sl_pct", "tp_pct", "reasoning"],
}


class DirectorAgent:
    def __init__(
        self,
        bus: BusClient,
        llm: LLMClient,
        decisions: DecisionWriter,
        settings: Settings,
    ) -> None:
        self._bus = bus
        self._llm = llm
        self._decisions = decisions
        self._cfg = settings
        self._pending: dict[str, dict[str, Any]] = {}   # cid → {quant, risk, ...}
        self._events:  dict[str, asyncio.Event]     = {}   # cid → Event

    async def run(self) -> None:
        await asyncio.gather(
            self._collect_results(),
            self._scheduler(),
        )

    async def _collect_results(self) -> None:
        async for topic, data in self._bus.psubscribe("agent.result.*"):
            parts = topic.split(".")
            if len(parts) < 4:
                continue
            result_type = parts[2]  # quant | risk | sentiment | portfolio
            msg_cls = _RESULT_CLASSES.get(result_type)
            if msg_cls is None:
                continue
            try:
                msg = msg_cls.model_validate_json(data)
            except Exception as exc:
                logger.warning("DirectorAgent: failed to parse %s result: %s", result_type, exc)
                continue

            cid = msg.correlation_id
            if cid not in self._pending:
                continue  # stale result from a previous cycle

            self._pending[cid][result_type] = msg
            if _REQUIRED_AGENTS.issubset(self._pending[cid].keys()):
                self._events[cid].set()

    async def _scheduler(self) -> None:
        while True:
            for symbol in self._cfg.symbol_list:
                asyncio.create_task(self._analyze_symbol(symbol))
            await asyncio.sleep(self._cfg.director_interval_s)

    async def _analyze_symbol(self, symbol: str) -> None:
        req = AnalyzeRequest(symbol=symbol)
        cid = req.correlation_id
        self._pending[cid] = {}
        self._events[cid] = asyncio.Event()

        await self._bus.publish(f"agent.analyze.{symbol}", req)

        try:
            await asyncio.wait_for(
                self._events[cid].wait(),
                timeout=self._cfg.agent_timeout_s,
            )
            results = dict(self._pending[cid])
        except asyncio.TimeoutError:
            results = dict(self._pending.get(cid, {}))
            got = set(results.keys())
            missing = _REQUIRED_AGENTS - got
            logger.warning(
                "DirectorAgent: timeout for %s, missing=%s", symbol, missing,
            )
        finally:
            self._pending.pop(cid, None)
            self._events.pop(cid, None)

        if len(results) < len(_REQUIRED_AGENTS):
            logger.warning(
                "DirectorAgent: skipping %s — only %d/%d results",
                symbol, len(results), len(_REQUIRED_AGENTS),
            )
            return

        await self._analyze_symbol_with_results(symbol, req, results)

    async def _analyze_symbol_with_results(
        self,
        symbol: str,
        req: AnalyzeRequest,
        results: dict[str, Any],
    ) -> None:
        quant: QuantResult         = results["quant"]
        risk: RiskResult           = results["risk"]
        sentiment: SentimentResult = results["sentiment"]
        portfolio: PortfolioResult = results["portfolio"]

        prompt = (
            f"Symbol: {symbol}\n\n"
            f"QUANT AGENT:\n"
            f"  Regime: {quant.regime}, Signal strength: {quant.signal_strength:.2f}, "
            f"Confidence: {quant.confidence:.2f}\n"
            f"  Reasoning: {quant.reasoning}\n"
            f"  Key indicators: RSI={quant.indicators.get('rsi')}, "
            f"EMA cross={quant.indicators.get('ema_cross')}, "
            f"MACD hist={quant.indicators.get('macd_hist')}\n\n"
            f"RISK AGENT:\n"
            f"  Kelly fraction: {risk.kelly_fraction:.3f}, Max loss: ${risk.max_loss_usdt:.2f}\n"
            f"  Reasoning: {risk.reasoning}\n\n"
            f"SENTIMENT AGENT:\n"
            f"  Score: {sentiment.score:.2f} ({sentiment.summary})\n\n"
            f"PORTFOLIO AGENT:\n"
            f"  Approved: {portfolio.approved}, "
            f"Correlation penalty: {portfolio.correlation_penalty:.2f}\n"
            f"  Reasoning: {portfolio.reasoning}\n\n"
            f"Account config: balance=${self._cfg.risk.starting_balance_usd:.0f}, "
            f"max_leverage={self._cfg.risk.max_leverage}x"
        )

        if not portfolio.approved:
            logger.info("DirectorAgent: %s blocked by portfolio agent", symbol)
            raw: dict[str, Any] = {
                "action": "hold", "side": "LONG", "confidence": 0.0,
                "size_pct": 0.0, "sl_pct": 0.02, "tp_pct": 0.04,
                "reasoning": f"Portfolio agent blocked: {portfolio.reasoning}",
            }
        else:
            raw = await self._llm.ask(
                system=_SYSTEM,
                prompt=prompt,
                tool_name="trading_decision",
                tool_schema=_TOOL_SCHEMA,
            )

        entry_price = float(quant.indicators.get("close", 0.0))
        decision = DirectorDecision(
            symbol=symbol,
            correlation_id=req.correlation_id,
            action=raw["action"],
            side=raw["side"],
            confidence=float(raw["confidence"]),
            size_pct=float(raw["size_pct"]),
            sl_pct=float(raw["sl_pct"]),
            tp_pct=float(raw["tp_pct"]),
            entry_price=entry_price,
            reasoning=raw["reasoning"],
            quant_summary=f"{quant.regime} str={quant.signal_strength:.2f}",
            risk_summary=f"kelly={risk.kelly_fraction:.3f}",
            sentiment_summary=f"score={sentiment.score:.2f}",
            portfolio_summary=(
                f"approved={portfolio.approved} penalty={portfolio.correlation_penalty:.2f}"
            ),
        )

        await self._decisions.insert(decision)
        logger.info(
            "DirectorAgent: %s action=%s confidence=%.2f",
            symbol, decision.action, decision.confidence,
        )

        if decision.action == "hold":
            return

        # Build Signal for PaperTradeEngine
        balance = self._cfg.risk.starting_balance_usd
        size_usd = balance * min(decision.size_pct, self._cfg.risk.max_position_pct)
        leverage = self._cfg.risk.max_leverage

        if decision.side == "LONG":
            sl = entry_price * (1.0 - decision.sl_pct)
            tp = entry_price * (1.0 + decision.tp_pct)
        else:
            sl = entry_price * (1.0 + decision.sl_pct)
            tp = entry_price * (1.0 - decision.tp_pct)

        signal = Signal(
            symbol=symbol,
            correlation_id=req.correlation_id,
            side=decision.side,
            size_usd=round(size_usd, 2),
            sl=round(sl, 4),
            tp=round(tp, 4),
            leverage=leverage,
            reasoning=decision.reasoning,
        )
        await self._bus.publish("signal.execute", signal)
        logger.info(
            "DirectorAgent: published signal %s %s size=$%.2f sl=%.4f tp=%.4f",
            symbol, decision.side, signal.size_usd, signal.sl, signal.tp,
        )
