"""OllamaScorer — uses local Qwen 2.5 7B to score news relevance + sentiment per symbol."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are a crypto trading relevance scorer.
Given a news article, score its relevance and sentiment for each of these crypto symbols: {symbols}.

Article title: {title}
Article body (first 500 chars): {body}

Return ONLY valid JSON in this exact format (no extra text):
{{
  "SYMBOL1": {{"relevance": 0.0, "sentiment": 0.0, "summary": "..."}},
  "SYMBOL2": {{"relevance": 0.0, "sentiment": 0.0, "summary": "..."}}
}}

relevance: 0.0 (not related) to 1.0 (directly about this asset)
sentiment: -1.0 (very bearish) to 1.0 (very bullish)
summary: max 30 words
"""


@dataclass
class ScoredArticle:
    symbol: str
    relevance: float    # 0.0–1.0
    sentiment: float    # -1.0 to 1.0
    summary: str


class OllamaScorer:
    def __init__(self, ollama_url: str, model: str, symbols: list[str]) -> None:
        self._url = f"{ollama_url.rstrip('/')}/api/generate"
        self._model = model
        self._symbols = symbols

    async def score(self, title: str, body: str) -> list[ScoredArticle]:
        """Score an article for all symbols. Returns neutral on any error."""
        try:
            raw = await self._call_ollama(title, body[:500])
            return [
                ScoredArticle(
                    symbol=sym,
                    relevance=max(0.0, min(1.0, float(raw.get(sym, {}).get("relevance", 0.0)))),
                    sentiment=max(-1.0, min(1.0, float(raw.get(sym, {}).get("sentiment", 0.0)))),
                    summary=str(raw.get(sym, {}).get("summary", "")),
                )
                for sym in self._symbols
            ]
        except Exception as exc:
            logger.warning("OllamaScorer error: %s — returning neutral scores", exc)
            return [
                ScoredArticle(symbol=sym, relevance=0.0, sentiment=0.0, summary="")
                for sym in self._symbols
            ]

    async def _call_ollama(self, title: str, body: str) -> dict:
        prompt = _PROMPT_TEMPLATE.format(
            symbols=", ".join(self._symbols),
            title=title,
            body=body,
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._url,
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            text = resp.json()["response"].strip()
            # Extract JSON block if wrapped in markdown
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            return json.loads(text)
