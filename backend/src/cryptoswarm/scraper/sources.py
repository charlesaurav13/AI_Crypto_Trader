"""Static list of news sources for ScrapeGraphAI."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class NewsSource:
    name: str
    url: str
    kind: Literal["article", "reddit", "social"]
    extraction_prompt: str


SOURCES: list[NewsSource] = [
    NewsSource(
        name="coindesk",
        url="https://www.coindesk.com/markets/",
        kind="article",
        extraction_prompt="Extract all news article titles, URLs, and first 300 chars of body text. Return as JSON list: [{title, url, body}]",
    ),
    NewsSource(
        name="cointelegraph",
        url="https://cointelegraph.com/",
        kind="article",
        extraction_prompt="Extract all news article titles, URLs, and first 300 chars of body text. Return as JSON list: [{title, url, body}]",
    ),
    NewsSource(
        name="decrypt",
        url="https://decrypt.co/news",
        kind="article",
        extraction_prompt="Extract all news article titles, URLs, and first 300 chars of body text. Return as JSON list: [{title, url, body}]",
    ),
    NewsSource(
        name="theblock",
        url="https://www.theblock.co/latest",
        kind="article",
        extraction_prompt="Extract all news article titles, URLs, and first 300 chars of body text. Return as JSON list: [{title, url, body}]",
    ),
    NewsSource(
        name="cryptoslate",
        url="https://cryptoslate.com/news/",
        kind="article",
        extraction_prompt="Extract all news article titles, URLs, and first 300 chars of body text. Return as JSON list: [{title, url, body}]",
    ),
    NewsSource(
        name="reddit_crypto",
        url="https://www.reddit.com/r/CryptoMarkets/new/.json?limit=25",
        kind="reddit",
        extraction_prompt="Extract post titles and selftext. Return as JSON list: [{title, url, body}]",
    ),
    NewsSource(
        name="reddit_bitcoin",
        url="https://www.reddit.com/r/Bitcoin/new/.json?limit=25",
        kind="reddit",
        extraction_prompt="Extract post titles and selftext. Return as JSON list: [{title, url, body}]",
    ),
    NewsSource(
        name="reddit_ethtrader",
        url="https://www.reddit.com/r/ethtrader/new/.json?limit=25",
        kind="reddit",
        extraction_prompt="Extract post titles and selftext. Return as JSON list: [{title, url, body}]",
    ),
    NewsSource(
        name="reuters_crypto",
        url="https://www.reuters.com/technology/",
        kind="article",
        extraction_prompt="Extract cryptocurrency and blockchain related news: titles, URLs, and first 300 chars of body. Return as JSON list: [{title, url, body}]",
    ),
    NewsSource(
        name="twitter_nitter",
        url="https://nitter.privacydev.net/search?q=bitcoin+OR+ethereum+crypto&f=tweets",
        kind="social",
        extraction_prompt="Extract tweet texts from the timeline. Return as JSON list: [{title, url, body}] where title=username, body=tweet text, url=tweet link",
    ),
]
