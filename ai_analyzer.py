"""Analyze fetched articles with OpenAI to classify them and determine event outcome."""

from __future__ import annotations

import json
import logging
import os
from typing import TypedDict

from openai import OpenAI

from news_fetcher import Article
from resilience import with_resilience
from oracle_constants import ORACLE_POLICY_VERSION

logger = logging.getLogger(__name__)




NORMALIZATION_PROMPT = """\
You are an AI Event Normalizer for a prediction market.

Your task is to take a raw market question and convert it into a strictly factual, standardized JSON object.

RULES:
1. Remove all emotional language, adjectives, and bias.
2. Convert to a factual measurable claim.
3. Standardize numbers and units.
4. Clarify if binary or threshold-based.
5. You must output STRICTLY valid JSON matching the format below.

JSON FORMAT:
{
  "subject": "...",
  "action": "...",
  "object": "...",
  "quantifier": "...",
  "deadline": "..."
}
"""

@with_resilience
def normalize_event(client: OpenAI, subject: str, event: str, deadline: str) -> str:
    """Normalize the raw market event into a structured factual claim."""
    raw_query = f"Subject: {subject}\nEvent: {event}\nDeadline: {deadline}"
    
    # Exception bubbling up is required for @with_resilience retries
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": NORMALIZATION_PROMPT},
            {"role": "user", "content": raw_query},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "normalized_event",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "action": {"type": "string"},
                        "object": {"type": "string"},
                        "quantifier": {"type": "string"},
                        "deadline": {"type": "string"}
                    },
                    "required": ["subject", "action", "object", "quantifier", "deadline"],
                    "additionalProperties": False
                }
            }
        },
        timeout=3.5,
    )
    
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse normalization GPT response as JSON: %s. Raw response: %r", e, raw)
        raise
    
    logger.debug("Original Event: '%s' '%s'", subject, event)
    logger.debug("Normalized Event: %s", json.dumps(parsed))
    
    return json.dumps(parsed, indent=2)


# --- New Direct Settlement Prompt ---

SETTLEMENT_PROMPT = """\
You are an impartial AI oracle tasked with resolving a binary prediction market based on verifiable real-world events. Your goal is to determine if the specified event occurred before the deadline, using only the provided news articles as evidence. Do not use external knowledge or assumptions—base your verdict solely on the articles.

Market Title: {subject} — {event}
Event Description: {normalized_event}
Deadline: {deadline} (The event must have occurred BEFORE this timestamp)

Fetched Articles:
{articles_json}

Resolution Criteria:
- Output YES (1) if multiple reliable sources confirm the event happened exactly as described before the deadline.
- Output NO (0) if no confirmation, conflicting evidence, or the event did not occur.
- Require at least 2 corroborating sources for YES to avoid misinformation.
- For terms like "struck down" or "out as leader": Interpret strictly.
- If evidence is ambiguous or insufficient, lower your confidence but still decide based on preponderance.
- If zero articles were provided, you MAY fallback to your own knowledge, but cap confidence below 0.75.

Step-by-Step Reasoning:
1. Summarize key facts from the articles relevant to the event.
2. Check if the event matches the title/description and occurred before the deadline.
3. Evaluate source reliability and cross-corroboration.
4. Decide: YES or NO.

Output Format (JSON only, no extra text):
{
  "result": 1 or 0,
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of decision, citing specific articles"
}
"""

@with_resilience
def analyze(
    subject: str,
    event: str,
    articles: list[Article],
    deadline: str = "",
) -> dict:
    """Analyze articles directly using the full-context GPT prompt."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return {"result": 0, "confidence": 0.0, "reason": "API Key missing"}

    client = OpenAI(api_key=api_key)

    # 1. Normalize the event (keeps our standardized claim structure)
    normalized_event_json = normalize_event(client, subject, event, deadline)
    logger.info("Event normalized successfully.")

    # 2. Format articles for the prompt
    articles_data = []
    # Take top 10 articles max to fit within context window
    for a in articles[:10]:
        pub = a.published.strftime("%Y-%m-%d %H:%M UTC") if a.published else "unknown date"
        articles_data.append({
            "title": a.title,
            "source": a.source,
            "date": pub,
            "content": a.description or a.content or "No content available."
        })
    articles_json = json.dumps(articles_data, indent=2)

    # 3. Fill the template
    user_prompt = SETTLEMENT_PROMPT.format(
        subject=subject,
        event=event,
        normalized_event=normalized_event_json,
        deadline=deadline,
        articles_json=articles_json,
    )

    logger.info("Calling GPT-4o-mini for direct settlement verdict with %d articles...", len(articles_data))

    # 4. Request the verdict using Strict JSON Schema to guarantee format
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "settlement_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "integer", "description": "1 for YES, 0 for NO"},
                        "confidence": {"type": "number", "description": "0.0 to 1.0 confidence score"},
                        "reasoning": {"type": "string", "description": "Brief explanation of decision"}
                    },
                    "required": ["result", "confidence", "reasoning"],
                    "additionalProperties": False
                }
            }
        },
        timeout=10.0,
    )
    
    raw = response.choices[0].message.content
    
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse GPT response as JSON: %s. Raw response: %r", e, raw)
        raise  # Lets @with_resilience retry

    result = int(parsed.get("result", 0))
    confidence = float(parsed.get("confidence", 0.0))
    reasoning = parsed.get("reasoning", "No reasoning provided.")
    
    unique_sources = len({a["source"] for a in articles_data})

    # Optional: Log the article info that gets returned
    article_logs = [
        {"title": a["title"], "source": a["source"], "classification": "DIRECT_PROMPT", "confidence": confidence}
        for a in articles_data
    ]

    verdict = {
        "policy_version": ORACLE_POLICY_VERSION,
        "result": result,
        "confidence": round(confidence, 4),
        "evidence_count": len(articles_data),
        "confirm_count": len(articles_data) if result == 1 else 0,
        "deny_count": len(articles_data) if result == 0 else 0,
        "conditional_count": 0,
        "future_intent_count": 0,
        "opinion_count": 0,
        "weighted_confirm_score": float(len(articles_data)) if result == 1 else 0.0,
        "weighted_deny_score": float(len(articles_data)) if result == 0 else 0.0,
        "conflict": False,
        "reason": f"gpt_verdict: {reasoning}",
        "unique_source_count": unique_sources,
        "article_logs": article_logs,
        "normalized_event_json": normalized_event_json,
    }

    logger.info(
        "Direct Verdict Complete: result=%d, conf=%.2f, reason='%s'",
        result, confidence, reasoning
    )

    return verdict
