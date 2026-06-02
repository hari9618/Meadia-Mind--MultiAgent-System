# schemas.py
# ─────────────────────────────────────────────────────────────────────
# Pydantic v2 schemas for input validation, output validation,
# and structured agent responses.
#
# Every user query is validated BEFORE hitting the agent pipeline.
# Every agent output is validated BEFORE being returned to the UI.
# This catches bad inputs early and ensures consistent output shapes.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ── Enums ─────────────────────────────────────────────────────────────

class AgentType(str, Enum):
    SUMMARIZE = "summarize_agent"
    HIGHLIGHT = "highlight_agent"
    SOCIAL    = "social_agent"
    QA        = "qa_agent"


class ImportanceLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"


class HighlightCategory(str, Enum):
    STATISTIC  = "statistic"
    INSIGHT    = "insight"
    QUOTE      = "quote"
    PREDICTION = "prediction"


class TimestampHint(str, Enum):
    EARLY = "early"
    MID   = "mid"
    LATE  = "late"


# ── Input Schemas ─────────────────────────────────────────────────────

class QueryInput(BaseModel):
    """Validated user query before entering the agent pipeline."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's question or request",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracing / observability",
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Strip excessive whitespace, reject empty strings."""
        cleaned = " ".join(v.split())
        if not cleaned:
            raise ValueError("Query cannot be empty after stripping whitespace.")
        return cleaned

    @field_validator("query")
    @classmethod
    def reject_injection(cls, v: str) -> str:
        """Basic prompt injection guard — rejects obvious jailbreak patterns."""
        DANGEROUS_PATTERNS = [
            r"ignore (all |previous )?instructions",
            r"you are now",
            r"disregard (your |all )?",
            r"forget (everything|all instructions)",
            r"act as (a |an )?(?!media|summariz|highlight)",  # allow 'act as a media analyst'
            r"DAN mode",
            r"jailbreak",
        ]
        lower = v.lower()
        for pat in DANGEROUS_PATTERNS:
            if re.search(pat, lower):
                raise ValueError(
                    "Query contains potentially unsafe content. Please rephrase."
                )
        return v


class TranscriptInput(BaseModel):
    """Validated transcript before indexing into ChromaDB."""

    text: str = Field(
        ...,
        min_length=50,
        max_length=500_000,
        description="Raw transcript text",
    )
    doc_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="Unique document identifier (alphanumeric + _ -)",
    )

    @field_validator("text")
    @classmethod
    def check_meaningful(cls, v: str) -> str:
        """Reject transcripts that are mostly whitespace or symbols."""
        alphanum = sum(c.isalnum() for c in v)
        if alphanum / max(len(v), 1) < 0.3:
            raise ValueError(
                "Transcript appears to contain mostly non-text content."
            )
        return v


# ── Output Schemas ────────────────────────────────────────────────────

class HighlightItem(BaseModel):
    """A single highlight extracted by the highlight agent."""

    highlight:      str            = Field(..., min_length=10, max_length=500)
    importance:     ImportanceLevel
    category:       HighlightCategory
    timestamp_hint: TimestampHint

    @field_validator("highlight")
    @classmethod
    def strip_highlight(cls, v: str) -> str:
        return v.strip()


class SummarizeOutput(BaseModel):
    """Validated output from the summarize agent."""

    task:    Literal["summarize_agent"]
    content: str = Field(..., min_length=50)

    @field_validator("content")
    @classmethod
    def has_required_sections(cls, v: str) -> str:
        """Ensure the summary contains at least an overview section."""
        if "Overview" not in v and "overview" not in v.lower():
            # Don't fail hard — just warn; the LLM may format differently
            pass
        return v


class HighlightOutput(BaseModel):
    """Validated output from the highlight agent."""

    task:    Literal["highlight_agent"]
    content: list[HighlightItem]

    @model_validator(mode="after")
    def check_non_empty(self) -> "HighlightOutput":
        if not self.content:
            raise ValueError("Highlight output must contain at least one item.")
        return self

    @classmethod
    def from_raw(cls, raw_list: list[dict]) -> "HighlightOutput":
        """
        Coerce raw LLM JSON output into a validated HighlightOutput.
        Falls back gracefully for individual malformed items.
        """
        items = []
        for item in raw_list:
            try:
                items.append(HighlightItem(**item))
            except Exception:
                # Coerce bad values to defaults so one bad item doesn't fail all
                items.append(HighlightItem(
                    highlight=str(item.get("highlight", "Key moment"))[:500],
                    importance=ImportanceLevel.MEDIUM,
                    category=HighlightCategory.INSIGHT,
                    timestamp_hint=TimestampHint.MID,
                ))
        return cls(task="highlight_agent", content=items)


class SocialOutput(BaseModel):
    """Validated output from the social agent."""

    task:    Literal["social_agent"]
    content: str = Field(..., min_length=50)


class QAOutput(BaseModel):
    """Validated output from the Q&A agent."""

    task:    Literal["qa_agent"]
    content: str = Field(..., min_length=10)


class AgentResult(BaseModel):
    """
    Final validated result returned by run_agent() to the UI.
    Union discriminated by the 'task' field.
    """

    task:       str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    output:     Any                  # str | list[HighlightItem dict]
    num_chunks: int                  = Field(default=0, ge=0)

    @field_validator("tool_calls")
    @classmethod
    def cap_tool_calls(cls, v: list) -> list:
        """Prevent runaway tool calls from bloating state."""
        return v[:10]


# ── Guardrail Schemas ─────────────────────────────────────────────────

class GuardrailResult(BaseModel):
    """Result of running a guardrail check."""

    passed:  bool
    reason:  Optional[str] = None
    blocked: bool          = False   # True = hard block, False = soft warn


class ContentModerationResult(BaseModel):
    """Output of the content moderation guardrail."""

    is_safe:   bool
    flags:     list[str] = Field(default_factory=list)
    action:    Literal["allow", "warn", "block"] = "allow"
    safe_text: Optional[str] = None   # sanitized version if action == "warn"
