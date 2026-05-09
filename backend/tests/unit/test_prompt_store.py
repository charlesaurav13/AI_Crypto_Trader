"""Tests for PromptStore."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from cryptoswarm.learning.prompt_store import PromptStore


def _make_store(prompt: str | None = None):
    mock_pg = MagicMock()
    mock_pg.get_agent_prompt = AsyncMock(return_value=prompt)
    mock_pg.save_agent_prompt = AsyncMock()
    return PromptStore(pg=mock_pg)


async def test_get_returns_db_prompt_when_exists():
    store = _make_store(prompt="You are a quant analyst...")
    result = await store.get("quant", default="default prompt")
    assert result == "You are a quant analyst..."


async def test_get_returns_default_when_no_db_entry():
    store = _make_store(prompt=None)
    result = await store.get("quant", default="default prompt")
    assert result == "default prompt"


async def test_save_calls_pg():
    store = _make_store()
    await store.save("director", "New evolved prompt...", perf_score=0.72)
    store._pg.save_agent_prompt.assert_called_once_with(
        agent_name="director",
        system_prompt="New evolved prompt...",
        perf_score=0.72,
    )


async def test_get_caches_after_first_call():
    store = _make_store(prompt="cached prompt")
    _ = await store.get("quant", default="d")
    _ = await store.get("quant", default="d")
    # DB should only be hit once due to in-memory cache
    assert store._pg.get_agent_prompt.call_count == 1
