"""
Phase 8 — Red Team Attack Simulation Framework

A controlled test harness that injects fake articles with controlled metadata,
source tiers, publish dates, and pre-determined classifications to run
deterministic adversarial attack scenarios against the settlement engine.

NO external APIs are called. NO production logic is modified.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Inline copy of production constants (DO NOT modify production files)
# ---------------------------------------------------------------------------

TIER_WEIGHTS = {
    "reuters.com": 1.0,   # Tier 1
    "apnews.com": 1.0,    # Tier 1
    "bbc.com": 0.8,       # Tier 2
    "nytimes.com": 0.6,   # Tier 3
    "cnn.com": 0.4,       # Tier 4
}


@dataclass
class FakeArticle:
    """Controlled test article with injected classification."""
    title: str
    source: str
    classification: str  # Pre-determined: CONFIRM, DENY, CONDITIONAL, FUTURE_INTENT, OPINION, IRRELEVANT
    published: datetime | None = None
    content: str = ""
    tier: float = 0.4
    weight: float = 0.4


# ---------------------------------------------------------------------------
# Deterministic Engine (mirrors production logic exactly)
# ---------------------------------------------------------------------------

def _get_article_weight(source: str) -> float:
    source_lower = source.lower()
    for domain, weight in TIER_WEIGHTS.items():
        if domain.split(".")[0] in source_lower:
            return weight
    return 0.4


def _count_unique_sources(articles: list[FakeArticle]) -> int:
    return len({a.source.lower() for a in articles})


def run_simulation(
    test_name: str,
    articles: list[FakeArticle],
    deadline: str = "2026-03-03T23:59:59Z",
) -> dict:
    """
    Run a single attack simulation using injected articles.
    Mirrors the production aggregation + confidence logic exactly.
    No API calls. Fully deterministic.
    """

    # --- Filter: ignore articles published AFTER deadline ---
    dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    filtered = []
    for a in articles:
        if a.published and a.published > dl:
            continue  # Post-deadline — silently ignored
        filtered.append(a)

    # Counts
    confirm_count = 0
    deny_count = 0
    conditional_count = 0
    future_intent_count = 0
    opinion_count = 0
    irrelevant_count = 0
    evidence_count = 0
    weighted_confirm_score = 0.0
    weighted_deny_score = 0.0

    for a in filtered:
        c = a.classification
        weight = _get_article_weight(a.source)

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

    unique_sources = _count_unique_sources(filtered)
    conflict = (confirm_count >= 1 and deny_count >= 1)

    # --- Result Decision ---
    if confirm_count >= 2 and confirm_count > deny_count and not conflict:
        result = 1
        reason = "multi_source_confirmation"
    elif deny_count >= 2 and deny_count > confirm_count:
        result = 0
        reason = "multi_source_denial"
    elif conflict:
        result = 0
        reason = "conflicting_evidence"
    else:
        result = 0
        reason = "insufficient_confirmations"

    # --- Base Confidence (Weighted) ---
    if result == 1:
        base_conf = weighted_confirm_score / (weighted_confirm_score + weighted_deny_score + 1.0)
    elif result == 0 and deny_count > confirm_count:
        base_conf = weighted_deny_score / (weighted_confirm_score + weighted_deny_score + 1.0)
    else:
        base_conf = 0.3

    # Diversity Multiplier
    diversity_multiplier = min(1.0, unique_sources / 4)
    confidence = base_conf * diversity_multiplier

    # Conflict Penalty
    if conflict:
        confidence = confidence * 0.6

    # Ambiguity Penalty
    ambiguity_total = conditional_count + future_intent_count + opinion_count
    if ambiguity_total > confirm_count:
        confidence = confidence * 0.7

    # Clamp
    confidence = max(0.05, min(0.95, confidence))

    verdict = {
        "test_name": test_name,
        "result": result,
        "confidence": round(confidence, 4),
        "confirm_count": confirm_count,
        "deny_count": deny_count,
        "weighted_confirm_score": round(weighted_confirm_score, 2),
        "weighted_deny_score": round(weighted_deny_score, 2),
        "conditional_count": conditional_count,
        "future_intent_count": future_intent_count,
        "opinion_count": opinion_count,
        "evidence_count": evidence_count,
        "unique_source_count": unique_sources,
        "conflict": conflict,
        "reason": reason,
    }
    return verdict


# ---------------------------------------------------------------------------
# Attack Scenarios
# ---------------------------------------------------------------------------

DL = "2026-03-03T23:59:59Z"
BEFORE_DL = datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc)
AFTER_DL  = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)


def test_a_single_high_tier_false_confirmation():
    """TEST A — Single High-Tier False Confirmation.
    1 Reuters article confirms, no other sources.
    Expected: result=0, confidence low, reason=insufficient_confirmations
    """
    articles = [
        FakeArticle(
            title="Breaking: Event confirmed by Reuters",
            source="reuters.com",
            classification="CONFIRM",
            published=BEFORE_DL,
            content="The event has occurred.",
        ),
    ]
    return run_simulation("TEST_A: Single High-Tier False Confirmation", articles, DL)


def test_b_many_low_tier_confirmations():
    """TEST B — Many Low-Tier Confirmations.
    5 Tier-4 (cnn.com) sources confirm, 0 Tier-1 sources.
    Expected: result may be 1 but confidence significantly reduced.
    """
    articles = [
        FakeArticle(
            title=f"CNN Report #{i}: Event confirmed",
            source=f"cnn{i}.com" if i > 0 else "cnn.com",
            classification="CONFIRM",
            published=BEFORE_DL,
            content="Breaking news: the event has occurred.",
        )
        for i in range(5)
    ]
    # Make sources unique for diversity but all Tier 4
    articles[0].source = "cnn.com"
    articles[1].source = "cnn1.com"
    articles[2].source = "cnn2.com"
    articles[3].source = "cnn3.com"
    articles[4].source = "cnn4.com"
    return run_simulation("TEST_B: Many Low-Tier Confirmations", articles, DL)


def test_c_conflict_between_tier1_sources():
    """TEST C — Conflict Between Tier 1 Sources.
    2 Reuters confirm, 2 AP deny.
    Expected: conflict=true, confidence heavily penalized.
    """
    articles = [
        FakeArticle(title="Reuters: Event confirmed (1)", source="reuters.com", classification="CONFIRM", published=BEFORE_DL),
        FakeArticle(title="Reuters: Event confirmed (2)", source="reuters.com", classification="CONFIRM", published=BEFORE_DL),
        FakeArticle(title="AP: Event denied (1)", source="apnews.com", classification="DENY", published=BEFORE_DL),
        FakeArticle(title="AP: Event denied (2)", source="apnews.com", classification="DENY", published=BEFORE_DL),
    ]
    return run_simulation("TEST_C: Conflict Between Tier-1 Sources", articles, DL)


def test_d_future_intent_trap():
    """TEST D — Future Intent Trap.
    Articles say 'Fed is expected to cut 100bp next week'.
    Expected: future_intent_count > 0, result=0, confidence reduced.
    """
    articles = [
        FakeArticle(
            title="Fed expected to cut rates next week",
            source="reuters.com",
            classification="FUTURE_INTENT",
            published=BEFORE_DL,
            content="The Federal Reserve is expected to cut interest rates by 100 basis points next week.",
        ),
        FakeArticle(
            title="Markets anticipate major rate cut",
            source="bbc.com",
            classification="FUTURE_INTENT",
            published=BEFORE_DL,
            content="Markets are pricing in a 100bp emergency cut expected by next Wednesday.",
        ),
        FakeArticle(
            title="Rate cut expected soon says analyst",
            source="cnn.com",
            classification="FUTURE_INTENT",
            published=BEFORE_DL,
            content="Analysts expect the Fed to announce a rate cut in the coming days.",
        ),
    ]
    return run_simulation("TEST_D: Future Intent Trap", articles, DL)


def test_e_conditional_language_trap():
    """TEST E — Conditional Language Trap.
    'Bill could pass by March 31' / 'Court may strike down ruling'.
    Expected: conditional_count > 0, no CONFIRM classification.
    """
    articles = [
        FakeArticle(
            title="Bill could pass by March 31",
            source="reuters.com",
            classification="CONDITIONAL",
            published=BEFORE_DL,
            content="The bill could pass the Senate by March 31 if enough votes are gathered.",
        ),
        FakeArticle(
            title="Court may strike down ruling",
            source="apnews.com",
            classification="CONDITIONAL",
            published=BEFORE_DL,
            content="The Supreme Court may strike down the ruling, subject to further deliberation.",
        ),
        FakeArticle(
            title="Possible passage pending committee review",
            source="bbc.com",
            classification="CONDITIONAL",
            published=BEFORE_DL,
            content="Passage is pending committee review and could happen if bipartisan support holds.",
        ),
    ]
    return run_simulation("TEST_E: Conditional Language Trap", articles, DL)


def test_f_post_deadline_confirmation():
    """TEST F — Post-Deadline Confirmation.
    Article published AFTER deadline confirming event.
    Expected: Ignored completely.
    """
    articles = [
        FakeArticle(
            title="Event confirmed after deadline",
            source="reuters.com",
            classification="CONFIRM",
            published=AFTER_DL,
            content="The event was confirmed today.",
        ),
        FakeArticle(
            title="AP: Also confirmed after deadline",
            source="apnews.com",
            classification="CONFIRM",
            published=AFTER_DL,
            content="AP confirms the event took place.",
        ),
    ]
    return run_simulation("TEST_F: Post-Deadline Confirmation", articles, DL)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_a_single_high_tier_false_confirmation,
    test_b_many_low_tier_confirmations,
    test_c_conflict_between_tier1_sources,
    test_d_future_intent_trap,
    test_e_conditional_language_trap,
    test_f_post_deadline_confirmation,
]

EXPECTED = {
    "TEST_A: Single High-Tier False Confirmation": {
        "result": 0,
        "confidence_max": 0.35,
        "reason": "insufficient_confirmations",
    },
    "TEST_B: Many Low-Tier Confirmations": {
        "result": 1,
        "confidence_max": 0.70,
        "reason": "multi_source_confirmation",
    },
    "TEST_C: Conflict Between Tier-1 Sources": {
        "conflict": True,
        "result": 0,
        "confidence_max": 0.35,
        "reason": "conflicting_evidence",
    },
    "TEST_D: Future Intent Trap": {
        "result": 0,
        "future_intent_min": 1,
        "confirm_count": 0,
    },
    "TEST_E: Conditional Language Trap": {
        "result": 0,
        "conditional_min": 1,
        "confirm_count": 0,
    },
    "TEST_F: Post-Deadline Confirmation": {
        "result": 0,
        "confirm_count": 0,
        "evidence_count": 0,
    },
}


def check_expectation(verdict: dict, expected: dict) -> tuple[bool, list[str]]:
    """Check if the verdict matches the expected conditions."""
    failures = []
    name = verdict["test_name"]

    if "result" in expected and verdict["result"] != expected["result"]:
        failures.append(f"  result: expected {expected['result']}, got {verdict['result']}")

    if "confidence_max" in expected and verdict["confidence"] > expected["confidence_max"]:
        failures.append(f"  confidence: expected <= {expected['confidence_max']}, got {verdict['confidence']}")

    if "reason" in expected and verdict["reason"] != expected["reason"]:
        failures.append(f"  reason: expected '{expected['reason']}', got '{verdict['reason']}'")

    if "conflict" in expected and verdict["conflict"] != expected["conflict"]:
        failures.append(f"  conflict: expected {expected['conflict']}, got {verdict['conflict']}")

    if "future_intent_min" in expected and verdict["future_intent_count"] < expected["future_intent_min"]:
        failures.append(f"  future_intent_count: expected >= {expected['future_intent_min']}, got {verdict['future_intent_count']}")

    if "conditional_min" in expected and verdict["conditional_count"] < expected["conditional_min"]:
        failures.append(f"  conditional_count: expected >= {expected['conditional_min']}, got {verdict['conditional_count']}")

    if "confirm_count" in expected and verdict["confirm_count"] != expected["confirm_count"]:
        failures.append(f"  confirm_count: expected {expected['confirm_count']}, got {verdict['confirm_count']}")

    if "evidence_count" in expected and verdict["evidence_count"] != expected["evidence_count"]:
        failures.append(f"  evidence_count: expected {expected['evidence_count']}, got {verdict['evidence_count']}")

    return (len(failures) == 0, failures)


def main():
    print("=" * 70)
    print("  PHASE 8 — RED TEAM ATTACK SIMULATION FRAMEWORK")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for test_fn in ALL_TESTS:
        verdict = test_fn()
        name = verdict["test_name"]
        expected = EXPECTED.get(name, {})

        ok, failures = check_expectation(verdict, expected)

        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{'─' * 70}")
        print(f"{status}  {name}")
        print(f"{'─' * 70}")
        print(json.dumps(verdict, indent=2))

        if not ok:
            print("  EXPECTATION FAILURES:")
            for f in failures:
                print(f)
            failed += 1
        else:
            passed += 1

        print()

    print("=" * 70)
    print(f"  RESULTS: {passed} PASSED / {failed} FAILED / {len(ALL_TESTS)} TOTAL")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
