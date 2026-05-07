"""Multi-provider LLM client for CryptoSwarm agents.

All four providers expose the exact same interface:

    result = await client.ask(system, prompt, tool_name, tool_schema)

Provider is selected by ``settings.llm_provider``.  Switch at runtime by
setting the ``LLM_PROVIDER`` environment variable (no code change needed).

Supported providers
-------------------
anthropic   Claude (claude-3-5-sonnet-20241022, claude-3-opus-20240229, …)
openai      GPT-4o, GPT-4-turbo, GPT-3.5-turbo, o1-mini, …
ollama      Any local model via Ollama's OpenAI-compatible API
            (llama3.1, mistral-nemo, qwen2.5, … — model must support tools)
gemini      Gemini 1.5 Pro / Flash (google-genai SDK)

Quick-start .env
-----------------
# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Ollama  (no key needed — must have `ollama serve` running locally)
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-1.5-pro
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base — the contract every provider must satisfy
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Force the model to call a named tool; return the tool's input dict."""

    @abstractmethod
    async def ask(
        self,
        system: str,
        prompt: str,
        tool_name: str,
        tool_schema: dict,
    ) -> dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Anthropic  (Claude)
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Uses Anthropic tool_choice to guarantee a structured tool_use block."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic  # lazy — only imported when this provider is active
        self._client = anthropic.AsyncAnthropic(
            **({"api_key": api_key} if api_key else {})
        )
        self._model = model
        logger.debug("AnthropicProvider model=%s", model)

    async def ask(
        self,
        system: str,
        prompt: str,
        tool_name: str,
        tool_schema: dict,
    ) -> dict[str, Any]:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            tools=[{
                "name": tool_name,
                "description": f"Return the {tool_name} structured result.",
                "input_schema": tool_schema,
            }],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input  # type: ignore[return-value]
        raise RuntimeError(
            f"AnthropicProvider: no tool_use block for '{tool_name}'. "
            f"Block types: {[b.type for b in response.content]}"
        )


# ---------------------------------------------------------------------------
# OpenAI  (GPT-4o, GPT-4-turbo, GPT-3.5-turbo, …)
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """Uses OpenAI function-calling with tool_choice forced to the named function."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI  # lazy import
        self._client = AsyncOpenAI(
            api_key=api_key or "placeholder",
            **({"base_url": base_url} if base_url else {}),
        )
        self._model = model
        logger.debug("OpenAIProvider model=%s base_url=%s", model, base_url)

    async def ask(
        self,
        system: str,
        prompt: str,
        tool_name: str,
        tool_schema: dict,
    ) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self._model,
            tools=[{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Return the {tool_name} structured result.",
                    "parameters": tool_schema,
                },
            }],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls and tool_calls[0].function.name == tool_name:
            return json.loads(tool_calls[0].function.arguments)
        raise RuntimeError(
            f"OpenAIProvider: no function_call for '{tool_name}'. "
            f"finish_reason={response.choices[0].finish_reason}"
        )


# ---------------------------------------------------------------------------
# Ollama  (local models via OpenAI-compatible /v1 endpoint)
# ---------------------------------------------------------------------------

class OllamaProvider(OpenAIProvider):
    """Wraps Ollama's OpenAI-compatible REST API.

    Requires ``ollama serve`` running and the chosen model pulled:
        ollama pull llama3.1
        ollama pull mistral-nemo
        ollama pull qwen2.5

    Models with confirmed tool-calling support: llama3.1, mistral-nemo,
    qwen2.5, deepseek-r1 (>=7b).  Smaller models may silently ignore the
    tool_choice constraint; upgrade to a larger variant if results are empty.
    """

    def __init__(self, base_url: str, model: str) -> None:
        super().__init__(
            api_key="ollama",          # Ollama ignores the key
            model=model,
            base_url=f"{base_url.rstrip('/')}/v1",
        )
        logger.debug("OllamaProvider model=%s base_url=%s/v1", model, base_url.rstrip("/"))


# ---------------------------------------------------------------------------
# Google Gemini  (google-genai SDK >= 1.0)
# ---------------------------------------------------------------------------

# JSON Schema type → Gemini type string
_GEMINI_TYPE = {
    "string":  "STRING",
    "number":  "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array":   "ARRAY",
    "object":  "OBJECT",
}


def _to_gemini_schema(schema: dict) -> dict:
    """Convert a flat JSON Schema object dict to the Gemini Schema dict format."""
    props: dict[str, dict] = {}
    for name, prop in schema.get("properties", {}).items():
        entry: dict[str, Any] = {
            "type": _GEMINI_TYPE.get(prop.get("type", "string"), "STRING"),
        }
        if "description" in prop:
            entry["description"] = prop["description"]
        if "enum" in prop:
            entry["enum"] = prop["enum"]
        props[name] = entry

    result: dict[str, Any] = {"type": "OBJECT", "properties": props}
    if "required" in schema:
        result["required"] = schema["required"]
    return result


class GeminiProvider(LLMProvider):
    """Google Gemini via the google-genai SDK (>=1.0).

    Supports native async via ``client.aio.models.generate_content``.
    Forced function calling via ``tool_config`` mode=ANY.
    """

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai  # lazy import
        self._client = genai.Client(api_key=api_key)
        self._model = model
        logger.debug("GeminiProvider model=%s", model)

    async def ask(
        self,
        system: str,
        prompt: str,
        tool_name: str,
        tool_schema: dict,
    ) -> dict[str, Any]:
        from google import genai
        from google.genai import types

        tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool_name,
                    description=f"Return the {tool_name} structured result.",
                    parameters=_to_gemini_schema(tool_schema),
                )
            ]
        )
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tool_name],
                )
            ),
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )

        for part in response.candidates[0].content.parts:
            if part.function_call and part.function_call.name == tool_name:
                # args is a MapComposite — convert to plain dict
                return {k: v for k, v in part.function_call.args.items()}

        raise RuntimeError(
            f"GeminiProvider: no function_call for '{tool_name}'. "
            f"finish_reason={response.candidates[0].finish_reason}"
        )


# ---------------------------------------------------------------------------
# Factory + public LLMClient
# ---------------------------------------------------------------------------

def _make_provider(settings: Any) -> LLMProvider:
    name: str = settings.llm_provider
    if name == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )
    if name == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
    if name == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_url,
            model=settings.ollama_model,
        )
    if name == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
    raise ValueError(
        f"Unknown llm_provider={name!r}. "
        f"Valid choices: anthropic | openai | ollama | gemini"
    )


class LLMClient:
    """Provider-agnostic LLM client.

    All agents depend on this class; switch the underlying model by changing
    ``LLM_PROVIDER`` (and the matching API key / model env vars) in ``.env``
    — no code changes required anywhere else.

    Usage
    -----
    client = LLMClient(settings)
    result = await client.ask(
        system="You are a quant analyst.",
        prompt="Symbol: BTCUSDT\\nRSI: 62 ...",
        tool_name="quant_analysis",
        tool_schema={...},          # standard JSON Schema
    )
    # result is a plain dict matching tool_schema
    """

    def __init__(self, settings: Any) -> None:
        self._provider: LLMProvider = _make_provider(settings)
        logger.info(
            "LLMClient ready  provider=%-12s model=%s",
            settings.llm_provider,
            getattr(settings, f"{settings.llm_provider}_model",
                    getattr(settings, "anthropic_model", "?")),
        )

    async def ask(
        self,
        system: str,
        prompt: str,
        tool_name: str,
        tool_schema: dict,
    ) -> dict[str, Any]:
        return await self._provider.ask(system, prompt, tool_name, tool_schema)
