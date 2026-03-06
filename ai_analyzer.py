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
        timeout=3.5,
    )
    raw = response.choices[0].message.content.strip()
    
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse normalization GPT response as JSON. Falling back to regex. Raw: {raw!r}")
        import re
        parsed = {
            "subject": subject,
            "action": "occurred",
            "object": event,
            "quantifier": "",
            "deadline": deadline
        }
        # Try to pluck some fields out textually
        subj_match = re.search(r'"subject"\s*:\s*"([^"]+)"', raw)
        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', raw)
        obj_match = re.search(r'"object"\s*:\s*"([^"]+)"', raw)
        
        if subj_match: parsed["subject"] = subj_match.group(1)
        if action_match: parsed["action"] = action_match.group(1)
        if obj_match: parsed["object"] = obj_match.group(1)
    
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

    # 4. Request the verdict
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        timeout=10.0,
    )
    
    raw = response.choices[0].message.content.strip()
    
    # Clean up markdown code blocks if the model included them
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    # Model sometimes hallucinates weird unescaped newlines inside the JSON keys themselves
    import re
    raw = re.sub(r'{\s*"\s+', '{"', raw)
    raw = re.sub(r'\n\s*"result"', '"result"', raw)
    
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed standard parse. Trying repair wrapper. Raw was: {raw!r}")
        try:
            # Sometime structured outputs omit the opening bracket if it thinks the prompt implicitly provides it
            repaired = raw
            if not repaired.startswith("{"):
                repaired = "{" + repaired
            if not repaired.endswith("}"):
                repaired = repaired + "}"
                
            parsed = json.loads(repaired)
        except json.JSONDecodeError as deeper_e:
            logger.error("Failed to parse GPT response as JSON even after repair: %s", deeper_e)
            raise  # Lets @with_resilience retry

    # GPT sometimes adds leading newlines to the JSON keys themselves if poorly formatted
    # e.g., {'\n  "result"': 1} instead of {'result': 1}. Clean all keys:
    clean_parsed = {}
    for k, v in parsed.items():
        clean_k = k.strip().strip('"').strip("'")
        clean_parsed[clean_k] = v

    try:
        result = int(clean_parsed.get("result", 0))
        confidence = float(clean_parsed.get("confidence", 0.0))
        reasoning = clean_parsed.get("reasoning", "No reasoning provided.")
    except Exception as extract_err:
        logger.error(f"Failed to extract keys from GPT response. Parsed dict: {parsed}. Cleaned dict: {clean_parsed}. Raw was: {raw!r}")
        raise
    
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
