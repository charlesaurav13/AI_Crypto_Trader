import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from cryptoswarm.agents.llm import LLMClient


@pytest.fixture
def mock_anthropic_response():
    """Build a fake Anthropic messages.create response with a tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "quant_analysis"
    tool_block.input = {
        "regime": "trending_up",
        "signal_strength": 0.7,
        "confidence": 0.8,
        "reasoning": "bullish EMA cross",
    }
    response = MagicMock()
    response.content = [tool_block]
    return response


async def test_ask_returns_tool_input(mock_anthropic_response):
    with patch("cryptoswarm.agents.llm.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(return_value=mock_anthropic_response)

        client = LLMClient(api_key="test-key")
        result = await client.ask(
            system="You are a quant.",
            prompt="Analyze BTCUSDT",
            tool_name="quant_analysis",
            tool_schema={
                "type": "object",
                "properties": {
                    "regime": {"type": "string"},
                    "signal_strength": {"type": "number"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["regime", "signal_strength", "confidence", "reasoning"],
            },
        )

    assert result["regime"] == "trending_up"
    assert result["confidence"] == 0.8


async def test_ask_raises_if_no_tool_use():
    """If Claude returns only text (not a tool_use block), raise RuntimeError."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Here is my analysis..."

    fake_response = MagicMock()
    fake_response.content = [text_block]

    with patch("cryptoswarm.agents.llm.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(return_value=fake_response)

        client = LLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="did not return tool_use"):
            await client.ask(
                system="sys",
                prompt="prompt",
                tool_name="quant_analysis",
                tool_schema={"type": "object", "properties": {}, "required": []},
            )
