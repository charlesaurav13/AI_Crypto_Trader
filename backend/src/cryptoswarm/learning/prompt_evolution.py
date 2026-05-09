"""PromptEvolutionEngine — teacher-student critique loop.

Every interval_s:
  1. Fetch last N closed trades + their rewards
  2. Score each agent type by average reward on trades it contributed to
  3. Identify worst-performing agent
  4. Call Teacher LLM: critique worst agent's decisions, rewrite its prompt
  5. Save new prompt version to DB
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptoswarm.learning.prompt_store import PromptStore
    from cryptoswarm.storage.postgres import PostgresWriter

logger = logging.getLogger(__name__)

_AGENT_NAMES = ["quant", "risk", "sentiment", "portfolio", "director"]

_DEFAULT_PROMPTS: dict[str, str] = {
    "quant": (
        "You are a quantitative trading analyst for Binance USDM perpetual futures. "
        "Given technical indicator values, identify the market regime and signal strength. "
        "Be concise and data-driven."
    ),
    "risk": (
        "You are a risk management specialist for crypto futures trading. "
        "Apply Kelly criterion conservatively. Protect capital above all else."
    ),
    "sentiment": (
        "You are a crypto market sentiment analyst. "
        "Combine news sentiment and market fear/greed data to assess market mood."
    ),
    "portfolio": (
        "You are a portfolio manager overseeing crypto futures positions. "
        "Assess correlation risk and concentration limits carefully."
    ),
    "director": (
        "You are the Chief Trading Officer of a multi-agent AI futures trading system. "
        "Synthesize all agent signals into decisive, profitable trading decisions. "
        "Be aggressive on high-conviction signals, hold on conflicted ones."
    ),
}

_EVOLUTION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "improved_prompt": {
            "type": "string",
            "description": "The complete rewritten system prompt for the agent",
        },
        "changes_summary": {
            "type": "string",
            "description": "1-2 sentences describing what was changed and why",
        },
        "perf_score": {
            "type": "number",
            "description": "Estimated performance improvement score 0.0-1.0",
        },
    },
    "required": ["improved_prompt", "changes_summary", "perf_score"],
}


class PromptEvolutionEngine:
    def __init__(
        self,
        pg: "PostgresWriter",
        llm,
        prompt_store: "PromptStore",
        interval_s: int = 25200,
        lookback: int = 50,
    ) -> None:
        self._pg = pg
        self._llm = llm
        self._store = prompt_store
        self._interval = interval_s
        self._lookback = lookback

    async def run(self) -> None:
        """Run forever: evolve prompts every interval_s."""
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("PromptEvolutionEngine.run error: %s", exc)

    async def run_once(self) -> None:
        """Single evolution cycle. Called directly in tests."""
        trades = await self._pg.get_recent_closed_trades(limit=self._lookback)
        if not trades:
            logger.info("PromptEvolutionEngine: no closed trades yet — skipping")
            return

        agent_scores = self._score_agents(trades)
        if not agent_scores:
            return

        worst_agent = min(agent_scores, key=agent_scores.get)  # type: ignore[arg-type]
        worst_score = agent_scores[worst_agent]
        logger.info(
            "PromptEvolutionEngine: worst agent=%s score=%.4f — evolving",
            worst_agent, worst_score,
        )

        default = _DEFAULT_PROMPTS.get(worst_agent, "You are a trading agent.")
        current_prompt = await self._store.get(worst_agent, default=default)

        worst_trades = self._get_worst_trades(trades, worst_agent, n=10)
        critique_prompt = self._build_critique_prompt(
            agent_name=worst_agent,
            current_prompt=current_prompt,
            worst_trades=worst_trades,
            avg_score=worst_score,
        )

        try:
            result = await self._llm.ask(
                system=(
                    "You are an expert AI prompt engineer specialising in crypto trading agents. "
                    "Your goal is to improve agent prompts to make them more profitable and risk-aware."
                ),
                prompt=critique_prompt,
                tool_name="evolve_prompt",
                tool_schema=_EVOLUTION_TOOL_SCHEMA,
            )
            new_prompt = result["improved_prompt"]
            perf_score = float(result.get("perf_score", 0.5))
            await self._store.save(worst_agent, new_prompt, perf_score=perf_score)
            logger.info(
                "PromptEvolutionEngine: evolved %s — %s",
                worst_agent, result.get("changes_summary", "")
            )
        except Exception as exc:
            logger.warning("PromptEvolutionEngine: LLM critique failed: %s", exc)

    def _score_agents(self, trades: list) -> dict[str, float]:
        """Compute mean reward per agent type based on trade history."""
        scores: dict[str, list[float]] = defaultdict(list)
        for t in trades:
            try:
                reward = float(t["reward"]) if t["reward"] is not None else 0.0
                action = json.loads(t["action"]) if isinstance(t["action"], str) else dict(t["action"])
                agent = action.get("agent", "director")
                if agent in _AGENT_NAMES:
                    scores[agent].append(reward)
            except Exception:
                continue
        if not scores:
            rewards = [float(t["reward"]) for t in trades if t.get("reward") is not None]
            return {"director": sum(rewards) / len(rewards)} if rewards else {}
        return {a: sum(v) / len(v) for a, v in scores.items()}

    def _get_worst_trades(self, trades: list, agent_name: str, n: int) -> list[dict]:
        """Return n worst trades (lowest reward) involving this agent."""
        relevant = []
        for t in trades:
            try:
                action = json.loads(t["action"]) if isinstance(t["action"], str) else dict(t["action"])
                if action.get("agent", "director") == agent_name:
                    relevant.append(t)
            except Exception:
                relevant.append(t)
        relevant.sort(key=lambda x: float(x["reward"]) if x.get("reward") else 0.0)
        return relevant[:n]

    def _build_critique_prompt(
        self,
        agent_name: str,
        current_prompt: str,
        worst_trades: list[dict],
        avg_score: float,
    ) -> str:
        examples = []
        for t in worst_trades[:5]:
            examples.append(
                f"  Symbol: {t.get('symbol', '?')} Side: {t.get('side', '?')} "
                f"P&L: {float(t.get('realized_pnl', 0)):.4f} "
                f"Reward: {float(t.get('reward', 0)):.4f} "
                f"Exit: {t.get('exit_reason', '?')}"
            )
        examples_str = "\n".join(examples) if examples else "  (no examples available)"
        return (
            f"Agent: {agent_name}\n"
            f"Current average reward: {avg_score:.4f} (negative = losing money)\n\n"
            f"Current system prompt:\n{current_prompt}\n\n"
            f"Worst performing trades this agent contributed to:\n{examples_str}\n\n"
            f"Rewrite the system prompt to avoid these failure patterns. "
            f"Make the agent more profitable, better at cutting losses, "
            f"and more aggressive on high-conviction opportunities. "
            f"Keep the improved prompt under 400 words."
        )
