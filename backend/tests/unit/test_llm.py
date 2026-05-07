"""Tests for the multi-provider LLMClient.

All network calls are mocked — no real API keys required.
Because each provider does a lazy ``import`` inside __init__, we either:
  (a) bypass the constructor via __new__ and set _client directly, or
  (b) patch the underlying SDK constructor before instantiation.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cryptoswarm.agents.llm import (
    LLMClient,
    AnthropicProvider,
    OpenAIProvider,
    OllamaProvider,
    GeminiProvider,
    _make_provider,
)
from cryptoswarm.config.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA = {
    "type": "object",
    "properties": {
        "regime":     {"type": "string", "enum": ["trending_up", "ranging"]},
        "confidence": {"type": "number"},
    },
    "required": ["regime", "confidence"],
}

_EXPECTED = {"regime": "trending_up", "confidence": 0.8}


def _settings(provider: str, **extra) -> Settings:
    return Settings(paper_trading=True, llm_provider=provider, **extra)


def _make_anthropic_response(tool_name: str, result: dict) -> MagicMock:
    tb = MagicMock()
    tb.type = "tool_use"
    tb.name = tool_name
    tb.input = result
    resp = MagicMock()
    resp.content = [tb]
    return resp


def _make_openai_response(tool_name: str, result: dict) -> MagicMock:
    fn = MagicMock()
    fn.name = tool_name
    fn.arguments = json.dumps(result)
    tc = MagicMock()
    tc.function = fn
    msg = MagicMock()
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "tool_calls"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_gemini_response(tool_name: str, result: dict) -> MagicMock:
    fc = MagicMock()
    fc.name = tool_name
    fc.args = result          # MapComposite-like — items() works on plain dict in tests
    part = MagicMock()
    part.function_call = fc
    cand = MagicMock()
    cand.content.parts = [part]
    cand.finish_reason = "STOP"
    resp = MagicMock()
    resp.candidates = [cand]
    return resp


# ---------------------------------------------------------------------------
# Factory / dispatch
# ---------------------------------------------------------------------------

def test_make_provider_anthropic():
    with patch("anthropic.AsyncAnthropic"):
        p = _make_provider(_settings("anthropic"))
    assert isinstance(p, AnthropicProvider)


def test_make_provider_openai():
    with patch("openai.AsyncOpenAI"):
        p = _make_provider(_settings("openai"))
    assert isinstance(p, OpenAIProvider)


def test_make_provider_ollama():
    with patch("openai.AsyncOpenAI"):
        p = _make_provider(_settings("ollama"))
    assert isinstance(p, OllamaProvider)


def test_make_provider_gemini():
    mock_genai = MagicMock()
    with patch.dict("sys.modules", {"google": MagicMock(genai=mock_genai), "google.genai": mock_genai}):
        p = _make_provider(_settings("gemini"))
    assert isinstance(p, GeminiProvider)


def test_make_provider_invalid_raises():
    s = _settings("anthropic")
    object.__setattr__(s, "llm_provider", "cohere")
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        _make_provider(s)


# ---------------------------------------------------------------------------
# LLMClient delegates to the injected provider
# ---------------------------------------------------------------------------

async def test_llm_client_delegates():
    mock_provider = MagicMock()
    mock_provider.ask = AsyncMock(return_value=_EXPECTED)

    client = LLMClient.__new__(LLMClient)
    client._provider = mock_provider

    result = await client.ask("sys", "prompt", "quant_analysis", _SCHEMA)
    assert result == _EXPECTED
    mock_provider.ask.assert_called_once_with("sys", "prompt", "quant_analysis", _SCHEMA)


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

async def test_anthropic_returns_tool_input():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        return_value=_make_anthropic_response("quant_analysis", _EXPECTED)
    )
    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        provider = AnthropicProvider(api_key="test", model="claude-3-5-sonnet-20241022")

    result = await provider.ask("sys", "prompt", "quant_analysis", _SCHEMA)
    assert result == _EXPECTED


async def test_anthropic_raises_on_missing_block():
    text_block = MagicMock()
    text_block.type = "text"
    resp = MagicMock()
    resp.content = [text_block]

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=resp)
    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        provider = AnthropicProvider(api_key="test", model="claude-3-5-sonnet-20241022")

    with pytest.raises(RuntimeError, match="AnthropicProvider"):
        await provider.ask("sys", "prompt", "quant_analysis", _SCHEMA)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

async def test_openai_returns_tool_call():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("quant_analysis", _EXPECTED)
    )
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

    result = await provider.ask("sys", "prompt", "quant_analysis", _SCHEMA)
    assert result == _EXPECTED


async def test_openai_raises_when_no_tool_call():
    msg = MagicMock()
    msg.tool_calls = []
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=resp)
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

    with pytest.raises(RuntimeError, match="OpenAIProvider"):
        await provider.ask("sys", "prompt", "quant_analysis", _SCHEMA)


# ---------------------------------------------------------------------------
# Ollama provider — inherits OpenAI logic; verify base_url wiring
# ---------------------------------------------------------------------------

def test_ollama_base_url_has_v1():
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch("openai.AsyncOpenAI", side_effect=capture):
        OllamaProvider(base_url="http://localhost:11434", model="llama3.1")

    assert captured["base_url"].endswith("/v1")
    assert captured["api_key"] == "ollama"


async def test_ollama_returns_tool_call():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("quant_analysis", _EXPECTED)
    )
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.1")

    result = await provider.ask("sys", "prompt", "quant_analysis", _SCHEMA)
    assert result == _EXPECTED


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

async def test_gemini_returns_function_call():
    fake_response = _make_gemini_response("quant_analysis", _EXPECTED)

    # Build provider bypassing __init__ (avoids needing real google.genai)
    provider = GeminiProvider.__new__(GeminiProvider)
    provider._model = "gemini-1.5-pro"
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock(return_value=fake_response)

    # Patch the lazy 'from google import genai' inside ask()
    mock_genai = MagicMock()
    mock_genai.types.Tool = MagicMock(return_value=MagicMock())
    mock_genai.types.FunctionDeclaration = MagicMock(return_value=MagicMock())
    mock_genai.types.GenerateContentConfig = MagicMock(return_value=MagicMock())
    mock_genai.types.ToolConfig = MagicMock(return_value=MagicMock())
    mock_genai.types.FunctionCallingConfig = MagicMock(return_value=MagicMock())

    with patch.dict("sys.modules", {
        "google":            MagicMock(genai=mock_genai),
        "google.genai":      mock_genai,
        "google.genai.types": mock_genai.types,
    }):
        result = await provider.ask("sys", "prompt", "quant_analysis", _SCHEMA)

    assert result["regime"] == "trending_up"
    assert result["confidence"] == 0.8


async def test_gemini_raises_when_no_function_call():
    # Part with function_call = None
    part = MagicMock()
    part.function_call = None
    cand = MagicMock()
    cand.content.parts = [part]
    cand.finish_reason = "STOP"
    resp = MagicMock()
    resp.candidates = [cand]

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._model = "gemini-1.5-pro"
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock(return_value=resp)

    mock_genai = MagicMock()
    with patch.dict("sys.modules", {
        "google":            MagicMock(genai=mock_genai),
        "google.genai":      mock_genai,
        "google.genai.types": mock_genai.types,
    }):
        with pytest.raises(RuntimeError, match="GeminiProvider"):
            await provider.ask("sys", "prompt", "quant_analysis", _SCHEMA)
