"""Analyze fetched articles with OpenAI to classify them and determine event outcome."""

from __future__ import annotations

import json
import logging
import os
from typing import TypedDict

from openai import OpenAI

from news_fetcher import Article
from resilience import with_resilience
from oracle_constants import (
    ORACLE_POLICY_VERSION,
    TIER_WEIGHTS,
    DEFAULT_TIER_WEIGHT,
    VALID_CLASSIFICATIONS,
    MIN_CONFIRMATIONS,
    MIN_UNIQUE_SOURCES,
    CONFLICT_PENALTY,
    AMBIGUITY_PENALTY,
    DIVERSITY_DIVISOR,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    FALLBACK_BASE_CONFIDENCE,
    SAFETY_FALLBACK_CONFIDENCE,
    DEFAULT_FALLBACK_CONFIDENCE,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an AI Event Classifier for a prediction market.

Your task is to classify whether a specific article CONFIRMs, DENYs, CONDITIONALs, FUTURE_INTENTs, OPINIONs, or is IRRELEVANT to the given event.

CRITICAL RULES:
1. You must ONLY use the provided article content.
2. You must NOT use prior knowledge or speculate.
3. The event must have occurred BEFORE the provided deadline.

DEFINITIONS:
CONFIRM: The event has definitively occurred before the deadline.
DENY: The event definitively did not occur before the deadline, or the opposite happened.
CONDITIONAL: The article uses "if", "may", "could", "pending", "subject to", etc.
FUTURE_INTENT: The article says it will happen, plans to happen, expected to happen, but has NOT yet occurred.
OPINION: Analysis, commentary, or prediction without factual confirmation.
IRRELEVANT: The article does not contain enough clear information about the event.

JSON FORMAT:
{
  "classification": "CONFIRM" | "DENY" | "CONDITIONAL" | "FUTURE_INTENT" | "OPINION" | "IRRELEVANT",
  "confidence": 0.0 to 1.0
}
"""

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

DIRECT_VERDICT_PROMPT = """\
You are an AI Oracle for a prediction market. No news articles were found for this event.

Based on your training data and knowledge, determine whether the following event has occurred.

RULES:
1. Answer based ONLY on widely-reported, factual events you are confident about.
2. If you are not sure, set result to 0 and confidence below 0.5.
3. If the event clearly happened based on well-known facts, set result to 1 and confidence to 0.7-0.9.
4. If the event clearly did NOT happen, set result to 0 and confidence to 0.7-0.9.

Respond in STRICT JSON:
{
  "result": 0 or 1,
  "confidence": 0.0 to 1.0,
  "reason": "short explanation"
}
"""


@with_resilience
def _gpt_direct_verdict(client: OpenAI, normalized_event_json: str, subject: str, event: str, deadline: str) -> dict:
    """Fallback: ask GPT directly when no news articles were found."""
    user_msg = f"Normalized Event:\\n{normalized_event_json}\\n\\nOriginal: {subject} — {event}\\nDeadline: {deadline}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": DIRECT_VERDICT_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        timeout=5.0,
    )
    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)
    
    result = int(parsed.get("result", 0))
    confidence = float(parsed.get("confidence", 0.3))
    reason = parsed.get("reason", "gpt_direct_knowledge")
    
    # Cap confidence for knowledge-based verdicts (no article evidence)
    confidence = min(confidence, 0.75)
    confidence = max(confidence, MIN_CONFIDENCE)
    
    logger.info("GPT direct verdict: result=%d, confidence=%.2f, reason=%s", result, confidence, reason)
    
    return {
        "policy_version": ORACLE_POLICY_VERSION,
        "result": result,
        "confidence": round(confidence, 4),
        "evidence_count": 0,
        "confirm_count": 0,
        "deny_count": 0,
        "weighted_confirm_score": 0.0,
        "weighted_deny_score": 0.0,
        "conditional_count": 0,
        "future_intent_count": 0,
        "opinion_count": 0,
        "unique_source_count": 0,
        "conflict": False,
        "reason": f"gpt_direct: {reason}",
    }



class ClassificationResult(TypedDict):
    classification: str
    confidence: float


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
        response_format={"type": "json_object"},
        timeout=3.5,
    )
    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)
    
    logger.debug("Original Event: '%s' '%s'", subject, event)
    logger.debug("Normalized Event: %s", json.dumps(parsed))
    
    return json.dumps(parsed, indent=2)


def _build_article_prompt(normalized_event_json: str, article: Article) -> str:
    """Build the prompt for a single article."""
    pub = article.published.strftime("%Y-%m-%d %H:%M UTC") if article.published else "unknown date"
    
    parts = [
        "TARGET EVENT (Normalized JSON):",
        normalized_event_json,
        "",
        "Article Details:",
        f"Source: {article.source}",
        f"Title: {article.title}",
        f"Published: {pub}",
    ]
    if article.description:
        parts.append(f"Description: {article.description}")
    if article.content:
        parts.append(f"Content: {article.content}")
        
    parts.extend([
        "",
        "Classify the article relative to the event. Return JSON ONLY."
    ])
    return "\n".join(parts)


@with_resilience
def classify_article(client: OpenAI, normalized_event_json: str, article: Article) -> ClassificationResult:
    """Classify a single article using OpenAI."""
    prompt = _build_article_prompt(normalized_event_json, article)
    
    # Let exceptions bubble up for @with_resilience
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        timeout=3.5,
    )
    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)
    
    c_val = str(parsed.get("classification", "IRRELEVANT")).upper()
    if c_val not in VALID_CLASSIFICATIONS:
        c_val = "IRRELEVANT"
        
    return {
        "classification": c_val,
        "confidence": float(parsed.get("confidence", 0.0)),
    }


# --- Source Tiers (imported from oracle_constants) ---

def _get_article_weight(source: str) -> float:
    """Get the weight of an article based on its source domain."""
    source_lower = source.lower()
    for domain, weight in TIER_WEIGHTS.items():
        if domain.split(".")[0] in source_lower:
            return weight
    return DEFAULT_TIER_WEIGHT

def _count_unique_sources(articles: list[Article]) -> int:
    """Count the number of unique trusted sources."""
    return len({a.source.lower() for a in articles})


# Structure matching the expected return JSON format
def analyze(
    subject: str,
    event: str,
    articles: list[Article],
    deadline: str = "",
) -> dict:
    """
    Classify each article individually, aggregate results using weighted tiers,
    and return final verdict.
    """
    # Base fallback structure
    base_verdict = {
        "policy_version": ORACLE_POLICY_VERSION,
        "result": 0,
        "confidence": DEFAULT_FALLBACK_CONFIDENCE,
        "evidence_count": 0,
        "confirm_count": 0,
        "deny_count": 0,
        "conditional_count": 0,
        "future_intent_count": 0,
        "opinion_count": 0,
        "weighted_confirm_score": 0.0,
        "weighted_deny_score": 0.0,
        "conflict": False,
        "reason": "insufficient_evidence"
    }

    # --- API key check (needed for normalization) ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return base_verdict

    client = OpenAI(api_key=api_key)

    # 1. Normalize the event (always runs so we always log both)
    logger.debug("ORIGINAL question: subject='%s' event='%s'", subject, event)
    normalized_event_json = normalize_event(client, subject, event, deadline)
    logger.debug("NORMALIZED event: %s", normalized_event_json)
    logger.info("Event normalized successfully.")

    # --- Safety: no articles — try direct GPT knowledge fallback ---
    if not articles:
        logger.info("0 news articles found — attempting GPT direct knowledge fallback.")
        try:
            fallback_result = _gpt_direct_verdict(client, normalized_event_json, subject, event, deadline)
            fallback_result["normalized_event_json"] = normalized_event_json
            fallback_result["article_logs"] = []
            return fallback_result
        except Exception as exc:
            logger.warning("GPT direct fallback failed: %s — returning default verdict.", exc)
            base_verdict["normalized_event_json"] = normalized_event_json
            base_verdict["article_logs"] = []
            return base_verdict

    # --- Safety: fewer than MIN_UNIQUE_SOURCES unique trusted sources ---
    unique_sources = _count_unique_sources(articles)
    if unique_sources < MIN_UNIQUE_SOURCES:
        logger.info(
            "SAFETY: Only %d unique source(s) — forcing fallback (need ≥%d).",
            unique_sources, MIN_UNIQUE_SOURCES,
        )
        base_verdict["confidence"] = SAFETY_FALLBACK_CONFIDENCE
        return base_verdict
    
    confirm_count = 0
    deny_count = 0
    conditional_count = 0
    future_intent_count = 0
    opinion_count = 0
    irrelevant_count = 0
    evidence_count = 0
    
    weighted_confirm_score = 0.0
    weighted_deny_score = 0.0
    
    article_logs = []
    
    # Curation: Select top 4 articles maximizing unique source diversity
    articles_to_process = []
    seen_sources = set()
    for a in articles:
        if a.source not in seen_sources and len(articles_to_process) < 4:
            articles_to_process.append(a)
            seen_sources.add(a.source)
    for a in articles:
        if len(articles_to_process) >= 4:
            break
        if a not in articles_to_process:
            articles_to_process.append(a)
    
    logger.info("Classifying %d curated articles against normalized event...", len(articles_to_process))
    
    for a in articles_to_process:
        res = classify_article(client, normalized_event_json, a)
        c = res["classification"]
        weight = _get_article_weight(a.source)
        
        # Only CONFIRM and DENY affect weighted score. CONDITIONAL, FUTURE_INTENT, OPINION increase evidence count
        if c == "CONFIRM":
            confirm_count += 1
            evidence_count += 1
            weighted_confirm_score += weight
        elif c == "DENY":
            deny_count += 1
            evidence_count += 1
            weighted_deny_score += weight
        elif c == "CONDITIONAL":
            conditional_count += 1
            evidence_count += 1
        elif c == "FUTURE_INTENT":
            future_intent_count += 1
            evidence_count += 1
        elif c == "OPINION":
            opinion_count += 1
            evidence_count += 1
        else:
            irrelevant_count += 1
            
        article_logs.append({
            "title": a.title,
            "source": a.source,
            "classification": c,
            "weight": weight,
            "confidence": res["confidence"]
        })
            
        logger.debug("Article '%s' -> %s (conf: %.2f, weight: %.1f)", a.title, c, res["confidence"], weight)

    logger.info(
        "Classification complete: CONFIRM=%d (%.1f) DENY=%d (%.1f) CONDITIONAL=%d FUTURE=%d OPINION=%d IRRELEVANT=%d",
        confirm_count, weighted_confirm_score, deny_count, weighted_deny_score, 
        conditional_count, future_intent_count, opinion_count, irrelevant_count
    )
    
    verdict = base_verdict.copy()
    verdict["confirm_count"] = confirm_count
    verdict["deny_count"] = deny_count
    verdict["conditional_count"] = conditional_count
    verdict["future_intent_count"] = future_intent_count
    verdict["opinion_count"] = opinion_count
    verdict["weighted_confirm_score"] = round(weighted_confirm_score, 2)
    verdict["weighted_deny_score"] = round(weighted_deny_score, 2)
    verdict["evidence_count"] = evidence_count
    
    conflict = (confirm_count >= 1 and deny_count >= 1)
    verdict["conflict"] = conflict
    
    # --- Strengthened Aggregation Rules & Deterministic Confidence Calibration ---
    
    # 1. Base Result
    if confirm_count >= MIN_CONFIRMATIONS and confirm_count > deny_count and not conflict:
        verdict["result"] = 1
        verdict["reason"] = "multi_source_confirmation"
    elif deny_count >= MIN_CONFIRMATIONS and deny_count > confirm_count:
        verdict["result"] = 0
        verdict["reason"] = "multi_source_denial"
    elif conflict:
        verdict["result"] = 0
        verdict["reason"] = "conflicting_evidence"
    else:
        verdict["result"] = 0
        verdict["reason"] = "insufficient_confirmations"
        
    # 2. Base Confidence (Using Weighted Scores)
    if verdict["result"] == 1:
        base_conf = weighted_confirm_score / (weighted_confirm_score + weighted_deny_score + 1.0)
    elif verdict["result"] == 0 and deny_count > confirm_count:
        base_conf = weighted_deny_score / (weighted_confirm_score + weighted_deny_score + 1.0)
    else:
        base_conf = FALLBACK_BASE_CONFIDENCE
        
    # 3. Apply Diversity Multiplier
    diversity_multiplier = min(1.0, unique_sources / DIVERSITY_DIVISOR)
    confidence = base_conf * diversity_multiplier
    
    # 4. Conflict Penalty
    if conflict:
        confidence = confidence * CONFLICT_PENALTY
        
    # 5. Ambiguity Penalty
    ambiguity_total = conditional_count + future_intent_count + opinion_count
    if ambiguity_total > confirm_count:
        confidence = confidence * AMBIGUITY_PENALTY
        
    # 6. Clamp Confidence
    confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))
    
    verdict["confidence"] = round(confidence, 4)
    verdict["unique_source_count"] = unique_sources
    verdict["article_logs"] = article_logs
    verdict["normalized_event_json"] = normalized_event_json
    
    return verdict
