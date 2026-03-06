# Oracle Settlement Policy Specification — v1.0

> **Policy Version:** `v1.0`
> **Effective Date:** 2026-03-03
> **Status:** FROZEN — No runtime modification permitted.

---

## 1. Trusted Domain List

Only articles from these domains are accepted for evidence:

| Domain        | Tier   | Weight |
|---------------|--------|--------|
| reuters.com   | Tier 1 | 1.0    |
| apnews.com    | Tier 1 | 1.0    |
| bbc.com       | Tier 2 | 0.8    |
| nytimes.com   | Tier 3 | 0.6    |
| cnn.com       | Tier 4 | 0.4    |

All other sources are rejected.

---

## 2. Source Tier Definitions

| Tier | Weight | Description                            |
|------|--------|----------------------------------------|
| 1    | 1.0    | Wire services, highest factual trust   |
| 2    | 0.8    | Major international broadcasters       |
| 3    | 0.6    | Major national newspapers              |
| 4    | 0.4    | Major cable news networks              |

Unknown but accepted sources default to Tier 4 (0.4).

---

## 3. Classification Categories

Each article is classified as one of:

| Category       | Definition                                                                 |
|----------------|----------------------------------------------------------------------------|
| CONFIRM        | Event definitively occurred **before** the deadline                        |
| DENY           | Event definitively did **not** occur, or the opposite happened             |
| CONDITIONAL    | Uses "if", "may", "could", "pending", "subject to"                        |
| FUTURE_INTENT  | States event will/plans to/expected to happen, but has **not yet** occurred|
| OPINION        | Analysis, commentary, or prediction without factual confirmation           |
| IRRELEVANT     | Insufficient information about the event                                   |

---

## 4. Aggregation Thresholds

| Parameter              | Value | Description                                      |
|------------------------|-------|--------------------------------------------------|
| MIN_CONFIRMATIONS      | 2     | Minimum CONFIRM count for result=1               |
| MIN_DENIALS            | 2     | Minimum DENY count for denial path               |
| MIN_UNIQUE_SOURCES     | 2     | Minimum unique trusted sources before analysis    |

---

## 5. Result Decision Rules

```
IF confirm_count >= 2 AND confirm_count > deny_count AND conflict == false:
    result = 1, reason = "multi_source_confirmation"

ELIF deny_count >= 2 AND deny_count > confirm_count:
    result = 0, reason = "multi_source_denial"

ELIF conflict == true:
    result = 0, reason = "conflicting_evidence"

ELSE:
    result = 0, reason = "insufficient_confirmations"
```

---

## 6. Conflict Definition

```
conflict = (confirm_count >= 1 AND deny_count >= 1)
```

---

## 7. Confidence Formula

### Step 1 — Base Confidence

```
IF result == 1:
    base = weighted_confirm_score / (weighted_confirm_score + weighted_deny_score + 1.0)

ELIF result == 0 AND deny_count > confirm_count:
    base = weighted_deny_score / (weighted_confirm_score + weighted_deny_score + 1.0)

ELSE:
    base = 0.3
```

### Step 2 — Diversity Multiplier

```
diversity_multiplier = min(1.0, unique_source_count / 4)
confidence = base * diversity_multiplier
```

### Step 3 — Conflict Penalty

```
IF conflict == true:
    confidence = confidence * 0.6
```

### Step 4 — Ambiguity Penalty

```
ambiguity_total = conditional_count + future_intent_count + opinion_count

IF ambiguity_total > confirm_count:
    confidence = confidence * 0.7
```

### Step 5 — Clamp

```
confidence = max(0.05, min(0.95, confidence))
```

---

## 8. Deadline Enforcement

- Articles published **after** the deadline are silently discarded.
- CONFIRM classification requires explicit statement that the event occurred **before** the deadline.
- Future-oriented language ("will happen", "expected") triggers FUTURE_INTENT, not CONFIRM.

---

## 9. Fallback Behavior

| Condition                       | Result | Confidence | Reason                    |
|---------------------------------|--------|------------|---------------------------|
| No OPENAI_API_KEY               | 0      | 0.2        | insufficient_evidence      |
| 0 articles found                | 0      | 0.2        | insufficient_evidence      |
| < 2 unique sources              | 0      | 0.3        | insufficient_evidence      |
| Insufficient confirmations      | 0      | computed   | insufficient_confirmations |

---

## 10. Structured Output Format

```json
{
  "policy_version": "v1.0",
  "result": 0,
  "confidence": 0.0,
  "confirm_count": 0,
  "deny_count": 0,
  "weighted_confirm_score": 0.0,
  "weighted_deny_score": 0.0,
  "conditional_count": 0,
  "future_intent_count": 0,
  "opinion_count": 0,
  "unique_source_count": 0,
  "evidence_count": 0,
  "conflict": false,
  "reason": "..."
}
```

---

## 11. Runtime immutability

All constants are defined in `oracle_constants.py` via the `OraclePolicy` class.
The `_FrozenPolicyMeta` metaclass prevents any attribute reassignment at runtime.
Any attempt to modify a frozen constant raises `AttributeError`.

---

## 12. Version History

| Version | Date       | Changes                        |
|---------|------------|--------------------------------|
| v1.0    | 2026-03-03 | Initial frozen policy release  |
