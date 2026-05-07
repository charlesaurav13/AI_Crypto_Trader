"""Thin async wrapper over the Anthropic SDK.

Uses tool_choice to guarantee structured JSON output — no free-text parsing needed.
Pass `api_key=""` to use the ANTHROPIC_API_KEY env var (SDK default).
"""
from __future__ import annotations

import anthropic
from typing import Any


class LLMClient:
    def __init__(self, api_key: str = "", model: str = "claude-3-5-sonnet-20241022") -> None:
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._model = model

    async def ask(
        self,
        system: str,
        prompt: str,
        tool_name: str,
        tool_schema: dict,
    ) -> dict[str, Any]:
        """Call Claude with tool_choice=tool_name to get guaranteed structured output.

        Args:
            system: System prompt string.
            prompt: User message content.
            tool_name: Exact name of the tool Claude must call.
            tool_schema: JSON Schema dict for the tool's input_schema.

        Returns:
            The tool's input dict (already validated by Claude against tool_schema).

        Raises:
            RuntimeError: If Claude did not return a tool_use block.
        """
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            tools=[
                {
                    "name": tool_name,
                    "description": f"Return the {tool_name} structured result.",
                    "input_schema": tool_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input  # type: ignore[return-value]
        raise RuntimeError(
            f"LLMClient: Claude did not return tool_use block for '{tool_name}'. "
            f"Content types: {[b.type for b in response.content]}"
        )
