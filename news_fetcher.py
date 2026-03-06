"""Fetch and process news articles from trusted international media.

Primary:  NewsAPI.org (title + description + content, domain filtering)
Fallback: Google News RSS (title only, no API key needed)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from resilience import with_resilience

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strict trusted domain whitelist — only these are accepted
# ---------------------------------------------------------------------------

TRUSTED_SOURCES = [
    "reuters.com",
    "bbc.com",
    "apnews.com",
    "cnn.com",
    "nytimes.com",
]

NEWSAPI_URL = "https://newsapi.org/v2/everything"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


@dataclass
class Article:
    """A single news article."""

    title: str
    description: str
    source: str
    published: datetime | None
    link: str
    content: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_tz(dt: datetime) -> datetime:
    """Make sure a datetime is timezone-aware (defaults to UTC)."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _domain_matches(source_name: str, domain: str) -> bool:
    """Check if a source name matches a trusted domain."""
    name_part = domain.split(".")[0].lower()
    return name_part in source_name.lower()


def _is_trusted(source_name: str, whitelist: list[str]) -> bool:
    """Check whether *source_name* matches any whitelisted domain."""
    return any(_domain_matches(source_name, d) for d in whitelist)


def _deduplicate(articles: list[Article]) -> list[Article]:
    """Remove duplicate articles by title (case-insensitive)."""
    seen: set[str] = set()
    unique: list[Article] = []
    for a in articles:
        key = a.title.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def _parse_iso_date(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 date string into a timezone-aware datetime."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return _ensure_tz(dt)
    except Exception:
        return None


def _parse_rss_date(raw: str | None) -> datetime | None:
    """Parse an RFC-822 date from RSS into a timezone-aware datetime."""
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# NewsAPI (primary)
# ---------------------------------------------------------------------------

@with_resilience
def _fetch_newsapi(
    subject: str,
    event: str,
    deadline: datetime,
    trusted_sources: list[str],
    api_key: str,
    max_results: int = 4,
) -> list[Article]:
    """Fetch articles from NewsAPI.org — returns title + description + content."""
    query = f"{subject} {event}"
    dl = _ensure_tz(deadline)

    # Only use domains that are in the trusted whitelist
    safe_domains = [d for d in trusted_sources if d in TRUSTED_SOURCES]
    if not safe_domains:
        logger.warning("No trusted domains in request — using full whitelist.")
        safe_domains = TRUSTED_SOURCES

    params = {
        "q": query,
        "domains": ",".join(safe_domains),
        "to": dl.strftime("%Y-%m-%dT%H:%M:%S"),
        "sortBy": "relevancy",
        "pageSize": max_results,
        "language": "en",
        "apiKey": api_key,
    }

    # Exception bubbling up is required for @with_resilience retries
    resp = requests.get(NEWSAPI_URL, params=params, timeout=2.5)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        logger.error("NewsAPI error: %s", data.get("message", "unknown"))
        return []

    articles: list[Article] = []
    for item in data.get("articles", []):
        pub_date = _parse_iso_date(item.get("publishedAt"))
        source_name = item.get("source", {}).get("name", "Unknown")

        # Safety: reject articles from untrusted sources
        if not _is_trusted(source_name, TRUSTED_SOURCES):
            logger.debug("Skipping untrusted source: %s", source_name)
            continue

        # Reject articles published AFTER deadline
        if pub_date is not None and pub_date > dl:
            continue

        articles.append(
            Article(
                title=item.get("title", "").strip(),
                description=(item.get("description") or "").strip(),
                source=source_name,
                published=pub_date,
                link=item.get("url", ""),
                content=(item.get("content") or "").strip(),
            )
        )

    # Deduplicate
    articles = _deduplicate(articles)

    logger.info(
        "NewsAPI: %d articles from trusted sources for '%s' (domains: %s, up to: %s)",
        len(articles), query, ",".join(safe_domains),
        dl.strftime("%Y-%m-%d"),
    )
    return articles


# ---------------------------------------------------------------------------
# Google News RSS (fallback)
# ---------------------------------------------------------------------------

@with_resilience
def _fetch_google_news(
    subject: str,
    event: str,
    deadline: datetime,
    max_results: int = 30,
) -> list[Article]:
    """Fallback: fetch from Google News RSS when NewsAPI is unavailable."""
    query = quote_plus(f"{subject} {event}")
    url = GOOGLE_NEWS_RSS.format(query=query)

    # Let exceptions bubble up for @with_resilience
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml-xml")
    items = soup.find_all("item", limit=max_results)

    dl = _ensure_tz(deadline)

    articles: list[Article] = []
    for item in items:
        source_tag = item.find("source")
        source_name = source_tag.text.strip() if source_tag else ""
        pub_date = _parse_rss_date(
            item.findNext("pubDate").text if item.find("pubDate") else None
        )

        # Only accept trusted sources
        if not _is_trusted(source_name, TRUSTED_SOURCES):
            continue
        if pub_date is not None and pub_date > dl:
            continue

        articles.append(
            Article(
                title=item.title.text.strip() if item.title else "",
                description="",
                source=source_name,
                published=pub_date,
                link=item.link.text.strip() if item.link else "",
                content="",
            )
        )

    articles = _deduplicate(articles)

    logger.info(
        "Google RSS fallback: %d trusted articles for '%s %s'",
        len(articles), subject, event,
    )
    return articles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_articles(
    subject: str,
    event: str,
    deadline: datetime,
    trusted_sources: list[str],
    max_results: int = 15,
) -> list[Article]:
    """Fetch articles — tries NewsAPI first, falls back to Google News RSS.

    All returned articles are guaranteed to be:
    - From TRUSTED_SOURCES whitelist only
    - Published on or before the deadline
    - Deduplicated by title
    """
    news_api_key = os.getenv("NEWS_API_KEY")

    if news_api_key:
        try:
            articles = _fetch_newsapi(
                subject, event, deadline, trusted_sources, news_api_key, max_results
            )
            if articles:
                return articles
            logger.warning("NewsAPI returned 0 results — falling back to Google News RSS.")
        except Exception as exc:
            logger.warning("NewsAPI failed completely (%s) — falling back to Google RSS.", type(exc).__name__)

    try:
        return _fetch_google_news(subject, event, deadline, max_results)
    except Exception as exc:
        logger.error("Google News RSS also failed completely: %s", type(exc).__name__)
        return []
