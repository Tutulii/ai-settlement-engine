"""AI Settlement Microservice — FastAPI entry point."""

import logging
import hashlib
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from ai_analyzer import analyze
from models import SettleRequest, SettleResponse
from news_fetcher import fetch_articles
from resilience import CircuitBreakerOpenException, APIRetryExhaustedException, global_circuit_breaker
from oracle_constants import ORACLE_POLICY_VERSION, MIN_CONFIDENCE

# ---------------------------------------------------------------------------
# Globals & State
# ---------------------------------------------------------------------------

# In-memory idempotency lock matching market_id to SettleResponse
SETTLED_MARKETS: dict[str, SettleResponse] = {}

# Immutable audit log
LOG_FILE = "settlement_audit.jsonl"

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Settlement Service",
    description="Analyzes trusted news sources to settle prediction market queries.",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Liveness probe reporting circuit breaker status and policy version."""
    if global_circuit_breaker.state == "open":
        api_status = "down"
    elif global_circuit_breaker.state == "half-open":
        api_status = "degraded"
    else:
        api_status = "healthy"
        
    return {
        "status": "ok",
        "policy_version": ORACLE_POLICY_VERSION,
        "api_status": api_status,
        "circuit_breaker": global_circuit_breaker.state
    }


@app.post("/settle", response_model=SettleResponse)
def settle(req: SettleRequest):
    """Settle a prediction market query.

    1. Fetch articles from trusted sources.
    2. Filter by deadline.
    3. Analyze with AI.
    4. Return structured verdict.
    """
    logger.info("Settlement request: market=%s subject=%s event=%s deadline=%s", 
                req.market_id, req.subject, req.event, req.deadline)

    # --- Idempotent Lock ---
    if req.market_id in SETTLED_MARKETS:
        logger.info("Idempotent lock hit for market %s. Returning cached result.", req.market_id)
        return SETTLED_MARKETS[req.market_id]

    try:
        # --- Step 1 & 2: Fetch + filter articles ---
        articles = fetch_articles(
            subject=req.subject,
            event=req.event,
            deadline=req.deadline,
            trusted_sources=req.trusted_sources,
        )

        # --- Step 3: AI analysis ---
        logger.info("Passing %d evidence articles to AI for analysis", len(articles))
        verdict = analyze(
            subject=req.subject,
            event=req.event,
            articles=articles,
            deadline=req.deadline.isoformat(),
        )
        
    except (CircuitBreakerOpenException, APIRetryExhaustedException) as exc:
        logger.error("External API resilience failure: %s", exc, exc_info=True)
        safe_fallback = SettleResponse(
            policy_version=ORACLE_POLICY_VERSION,
            result=0,
            confidence=MIN_CONFIDENCE,
            evidence_count=0,
            confirm_count=0,
            deny_count=0,
            weighted_confirm_score=0.0,
            weighted_deny_score=0.0,
            conditional_count=0,
            future_intent_count=0,
            opinion_count=0,
            unique_source_count=0,
            conflict=False,
            reason="external_api_failure"
        )
        return safe_fallback
    except Exception as exc:
        logger.error("Unexpected error during settlement: %s", exc)
        raise HTTPException(status_code=502, detail="Settlement failed unexpectedly.") from exc

    # --- Step 4: Deterministic Hashing and Persistent Logging ---
    article_logs = verdict.pop("article_logs", [])
    normalized_json_str = verdict.pop("normalized_event_json", "{}")
    
    evidence_str = "".join(f"{al['source']}|{al['title']}|{al['classification']}" for al in article_logs)
    hash_payload = f"{normalized_json_str}|{evidence_str}|{ORACLE_POLICY_VERSION}"
    settlement_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": req.market_id,
        "settlement_hash": settlement_hash,
        "policy_version": ORACLE_POLICY_VERSION,
        "normalized_event": json.loads(normalized_json_str) if normalized_json_str != "{}" else {},
        "evidence_used": article_logs,
        "final_output": verdict,
    }
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except IOError as e:
        logger.error("Failed to write to persistent log: %s", e)

    # --- Step 5: Respond and Cache ---
    resp = SettleResponse(
        policy_version=verdict.get("policy_version", ORACLE_POLICY_VERSION),
        result=verdict["result"],
        confidence=round(verdict["confidence"], 4),
        evidence_count=verdict.get("evidence_count", len(articles)),
        confirm_count=verdict.get("confirm_count", 0),
        deny_count=verdict.get("deny_count", 0),
        weighted_confirm_score=verdict.get("weighted_confirm_score", 0.0),
        weighted_deny_score=verdict.get("weighted_deny_score", 0.0),
        conditional_count=verdict.get("conditional_count", 0),
        future_intent_count=verdict.get("future_intent_count", 0),
        opinion_count=verdict.get("opinion_count", 0),
        unique_source_count=verdict.get("unique_source_count", 0),
        conflict=verdict.get("conflict", False),
        reason=verdict.get("reason", "unknown")
    )
    
    SETTLED_MARKETS[req.market_id] = resp
    return resp
