"""
Oracle Policy Constants — v1.0

All settlement thresholds and policy parameters are defined here.
These constants are FROZEN and must NOT be modified at runtime.

Any change to these values constitutes a new policy version and
requires updating ORACLE_POLICY_VERSION.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Policy Version
# ---------------------------------------------------------------------------

ORACLE_POLICY_VERSION = "v1.0"


# ---------------------------------------------------------------------------
# Trusted Domain Whitelist
# ---------------------------------------------------------------------------

TRUSTED_SOURCES = (
    "reuters.com",
    "bbc.com",
    "apnews.com",
    "cnn.com",
    "nytimes.com",
)


# ---------------------------------------------------------------------------
# Source Tier Weights
# ---------------------------------------------------------------------------

TIER_WEIGHTS: dict[str, float] = {
    "reuters.com": 1.0,   # Tier 1
    "apnews.com": 1.0,    # Tier 1
    "bbc.com": 0.8,       # Tier 2
    "nytimes.com": 0.6,   # Tier 3
    "cnn.com": 0.4,       # Tier 4
}

DEFAULT_TIER_WEIGHT = 0.4  # Unknown but trusted sources default to Tier 4


# ---------------------------------------------------------------------------
# Classification Categories
# ---------------------------------------------------------------------------

VALID_CLASSIFICATIONS = (
    "CONFIRM",
    "DENY",
    "CONDITIONAL",
    "FUTURE_INTENT",
    "OPINION",
    "IRRELEVANT",
)


# ---------------------------------------------------------------------------
# Aggregation Thresholds
# ---------------------------------------------------------------------------

MIN_CONFIRMATIONS = 2          # Minimum CONFIRM count to return result=1
MIN_DENIALS = 2                # Minimum DENY count to return result=0 (denial path)
MIN_UNIQUE_SOURCES = 2         # Minimum unique sources before analysis proceeds


# ---------------------------------------------------------------------------
# Confidence Formula Parameters
# ---------------------------------------------------------------------------

CONFLICT_PENALTY = 0.6         # Multiplier when conflict is detected
AMBIGUITY_PENALTY = 0.7        # Multiplier when ambiguity_total > confirm_count
DIVERSITY_DIVISOR = 4           # diversity_multiplier = min(1.0, unique_sources / DIVERSITY_DIVISOR)

MAX_CONFIDENCE = 0.95          # Upper clamp
MIN_CONFIDENCE = 0.05          # Lower clamp

FALLBACK_BASE_CONFIDENCE = 0.3 # Base confidence when no clear signal
SAFETY_FALLBACK_CONFIDENCE = 0.3  # Confidence when fewer than MIN_UNIQUE_SOURCES
DEFAULT_FALLBACK_CONFIDENCE = 0.2 # Confidence for empty/error states


# ---------------------------------------------------------------------------
# Freeze Guard — Prevent Runtime Modification
# ---------------------------------------------------------------------------

class _FrozenPolicyMeta(type):
    """Metaclass that prevents setting attributes on the module after import."""

    def __setattr__(cls, name: str, value: object) -> None:
        if name in cls.__dict__:
            raise AttributeError(
                f"Oracle policy constant '{name}' is FROZEN under policy {ORACLE_POLICY_VERSION}. "
                "Modification at runtime is not allowed."
            )
        super().__setattr__(name, value)


class OraclePolicy(metaclass=_FrozenPolicyMeta):
    """
    Immutable policy container.
    All constants are class-level attributes that cannot be reassigned at runtime.
    """

    VERSION = ORACLE_POLICY_VERSION

    TRUSTED_SOURCES = TRUSTED_SOURCES
    TIER_WEIGHTS = TIER_WEIGHTS
    DEFAULT_TIER_WEIGHT = DEFAULT_TIER_WEIGHT

    VALID_CLASSIFICATIONS = VALID_CLASSIFICATIONS

    MIN_CONFIRMATIONS = MIN_CONFIRMATIONS
    MIN_DENIALS = MIN_DENIALS
    MIN_UNIQUE_SOURCES = MIN_UNIQUE_SOURCES

    CONFLICT_PENALTY = CONFLICT_PENALTY
    AMBIGUITY_PENALTY = AMBIGUITY_PENALTY
    DIVERSITY_DIVISOR = DIVERSITY_DIVISOR

    MAX_CONFIDENCE = MAX_CONFIDENCE
    MIN_CONFIDENCE = MIN_CONFIDENCE
    FALLBACK_BASE_CONFIDENCE = FALLBACK_BASE_CONFIDENCE
    SAFETY_FALLBACK_CONFIDENCE = SAFETY_FALLBACK_CONFIDENCE
    DEFAULT_FALLBACK_CONFIDENCE = DEFAULT_FALLBACK_CONFIDENCE
