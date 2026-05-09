"""PromptStore — reads and writes versioned agent system prompts from agent_prompts table."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptoswarm.storage.postgres import PostgresWriter

logger = logging.getLogger(__name__)


class PromptStore:
    def __init__(self, pg: "PostgresWriter") -> None:
        self._pg = pg
        self._cache: dict[str, str] = {}   # agent_name → current prompt

    async def get(self, agent_name: str, default: str) -> str:
        """Return active prompt from cache → DB → default (in that order)."""
        if agent_name in self._cache:
            return self._cache[agent_name]
        try:
            prompt = await self._pg.get_agent_prompt(agent_name)
            if prompt:
                self._cache[agent_name] = prompt
                return prompt
        except Exception as exc:
            logger.warning("PromptStore.get DB error for %s: %s", agent_name, exc)
        return default

    async def save(
        self, agent_name: str, system_prompt: str, perf_score: float | None = None
    ) -> None:
        """Save new prompt version to DB and invalidate cache."""
        await self._pg.save_agent_prompt(
            agent_name=agent_name,
            system_prompt=system_prompt,
            perf_score=perf_score,
        )
        self.invalidate(agent_name)
        logger.info("PromptStore: saved new prompt for %s (score=%.3f)", agent_name, perf_score or 0)

    def invalidate(self, agent_name: str) -> None:
        """Force cache miss so next get() re-reads from DB."""
        self._cache.pop(agent_name, None)
