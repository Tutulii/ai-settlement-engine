"""Pydantic models for the settlement API."""

from datetime import datetime
from pydantic import BaseModel, Field


class SettleRequest(BaseModel):
    """Incoming settlement query."""

    market_id: str = Field(..., description="Unique ID for the market to ensure settlement idempotency")
    subject: str = Field(..., description="The subject of the prediction (e.g. a person or entity)")
    event: str = Field(..., description="The event to verify (e.g. 'officially declared deceased')")
    deadline: datetime = Field(..., description="ISO-8601 deadline; only articles before this date count")
    trusted_sources: list[str] = Field(
        ...,
        min_length=1,
        description="List of trusted news domains (e.g. ['reuters.com', 'bbc.com'])",
    )


class SettleResponse(BaseModel):
    """Settlement verdict returned to the caller."""

    policy_version: str = Field(..., description="Oracle policy version (e.g. v1.0)")
    result: int = Field(..., description="1 = YES (event occurred), 0 = NO")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    evidence_count: int = Field(..., ge=0, description="Number of supporting articles found")
    confirm_count: int = Field(..., ge=0, description="Number of articles explicitly confirming the event")
    deny_count: int = Field(..., ge=0, description="Number of articles explicitly denying the event")
    weighted_confirm_score: float = Field(..., ge=0, description="Sum of confirm tier weights")
    weighted_deny_score: float = Field(..., ge=0, description="Sum of deny tier weights")
    conditional_count: int = Field(..., ge=0, description="Number of conditional articles ('if', 'may', etc)")
    future_intent_count: int = Field(..., ge=0, description="Number of future intent articles ('expected', 'will')")
    opinion_count: int = Field(..., ge=0, description="Number of opinion/analysis articles")
    unique_source_count: int = Field(..., ge=0, description="Number of unique trusted domains providing evidence")
    conflict: bool = Field(..., description="True if conflicting information was detected")
    reason: str = Field(..., description="The reason for the final result (e.g. multi_source_confirmation)")
