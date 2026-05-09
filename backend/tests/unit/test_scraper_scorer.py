"""Tests for OllamaScorer — all HTTP calls mocked."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock, Mock

from cryptoswarm.scraper.scorer import OllamaScorer, ScoredArticle


def _make_scorer() -> OllamaScorer:
    return OllamaScorer(
        ollama_url="http://localhost:11434",
        model="qwen2.5:7b",
        symbols=["BTCUSDT", "ETHUSDT"],
    )


async def test_score_returns_scored_articles():
    scorer = _make_scorer()
    fake_response = {
        "BTCUSDT": {"relevance": 0.9, "sentiment": 0.7, "summary": "Bitcoin bullish"},
        "ETHUSDT": {"relevance": 0.2, "sentiment": 0.0, "summary": "Unrelated"},
    }
    with patch.object(scorer, "_call_ollama", AsyncMock(return_value=fake_response)):
        results = await scorer.score(
            title="BTC breaks $70k",
            body="Bitcoin surpassed 70,000 today...",
        )
    assert len(results) == 2
    btc = next(r for r in results if r.symbol == "BTCUSDT")
    assert btc.relevance == 0.9
    assert btc.sentiment == 0.7


async def test_score_clamps_values():
    scorer = _make_scorer()
    bad_response = {
        "BTCUSDT": {"relevance": 1.5, "sentiment": -2.0, "summary": "test"},
        "ETHUSDT": {"relevance": -0.1, "sentiment": 0.0, "summary": "test"},
    }
    with patch.object(scorer, "_call_ollama", AsyncMock(return_value=bad_response)):
        results = await scorer.score("title", "body")
    btc = next(r for r in results if r.symbol == "BTCUSDT")
    assert btc.relevance <= 1.0
    assert btc.sentiment >= -1.0
    eth = next(r for r in results if r.symbol == "ETHUSDT")
    assert eth.relevance >= 0.0


async def test_score_returns_neutral_on_ollama_error():
    scorer = _make_scorer()
    with patch.object(scorer, "_call_ollama", AsyncMock(side_effect=Exception("timeout"))):
        results = await scorer.score("title", "body")
    # Should return neutral scores rather than raise
    assert all(r.relevance == 0.0 for r in results)
    assert all(r.sentiment == 0.0 for r in results)


async def test_call_ollama_handles_markdown_wrapped_json():
    """_call_ollama strips ```json ... ``` wrapper before parsing."""
    scorer = _make_scorer()
    json_payload = '{"BTCUSDT": {"relevance": 0.8, "sentiment": 0.5, "summary": "BTC up"}}'
    markdown_response = f"```json\n{json_payload}\n```"
    mock_resp = Mock()
    mock_resp.json.return_value = {"response": markdown_response}
    mock_resp.raise_for_status = Mock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await scorer._call_ollama("BTC up", "body text")
    assert result["BTCUSDT"]["relevance"] == 0.8
    assert result["BTCUSDT"]["sentiment"] == 0.5
