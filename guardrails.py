# guardrails/guardrails.py
# ─────────────────────────────────────────────────────────────────────
# Production guardrails for MediaMind.
#
# Three layers of protection:
#   1. INPUT GUARDRAILS  — validate & sanitize before the agent pipeline
#   2. OUTPUT GUARDRAILS — validate & sanitize before returning to the UI
#   3. CONTENT GUARDRAILS — detect off-topic, toxic, or malicious queries
#
# Design principle: FAIL SAFE.
#   If a guardrail itself errors, we LOG the error but let the request
#   through (with a warning) rather than blocking legitimate users.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import re
from typing import Any

from schemas import (
    QueryInput,
    TranscriptInput,
    GuardrailResult,
    ContentModerationResult,
    HighlightOutput,
    AgentResult,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# INPUT GUARDRAILS
# ═════════════════════════════════════════════════════════════════════

def validate_query(raw_query: str, session_id: str | None = None) -> GuardrailResult:
    """
    Validate and sanitize a user query using the QueryInput Pydantic schema.

    Returns:
        GuardrailResult with passed=True and the cleaned query in `reason`,
        or passed=False with the error message.
    """
    try:
        validated = QueryInput(query=raw_query, session_id=session_id)
        return GuardrailResult(passed=True, reason=validated.query, blocked=False)
    except Exception as e:
        logger.warning(f"[GUARDRAIL] Query validation failed: {e}")
        return GuardrailResult(
            passed=False,
            reason=str(e),
            blocked=True,
        )


def validate_transcript(text: str, doc_id: str) -> GuardrailResult:
    """
    Validate a transcript before indexing.
    Catches empty transcripts, non-text content, and invalid doc IDs.
    """
    try:
        TranscriptInput(text=text, doc_id=doc_id)
        return GuardrailResult(passed=True)
    except Exception as e:
        logger.warning(f"[GUARDRAIL] Transcript validation failed: {e}")
        return GuardrailResult(passed=False, reason=str(e), blocked=True)


# ═════════════════════════════════════════════════════════════════════
# CONTENT MODERATION GUARDRAIL
# ═════════════════════════════════════════════════════════════════════

# These are rule-based checks — no API call required.
# For production at scale, swap the `_rule_based_check` with a call to
# a moderation API (OpenAI Moderation, Perspective API, etc.)

_OFF_TOPIC_PATTERNS = [
    r"\b(hack|exploit|bypass|jailbreak)\b",
    r"\b(weapon|bomb|drug|illegal)\b",
    r"\b(password|credit.?card|ssn|social.?security)\b",
    r"(porn|nsfw|adult.?content|explicit)",
]

_MEDIA_TOPIC_HINTS = [
    "podcast", "transcript", "video", "episode", "interview",
    "media", "content", "audio", "summarize", "highlight",
    "social", "post", "caption", "summary", "youtube",
    "what", "how", "who", "when", "why", "explain", "tell me",
]


def check_content(query: str) -> ContentModerationResult:
    """
    Rule-based content moderation.

    - Blocks clearly off-topic or harmful queries.
    - Passes media-related queries directly.
    - Ambiguous queries get a soft warn (allowed but flagged).
    """
    try:
        lower = query.lower()
        flags = []

        # Check for harmful patterns
        for pat in _OFF_TOPIC_PATTERNS:
            if re.search(pat, lower):
                flags.append(pat)

        if flags:
            logger.warning(f"[GUARDRAIL] Content blocked. Flags: {flags}")
            return ContentModerationResult(
                is_safe=False,
                flags=flags,
                action="block",
            )

        # Check if the query is media-related
        is_media = any(hint in lower for hint in _MEDIA_TOPIC_HINTS)

        if is_media:
            return ContentModerationResult(is_safe=True, action="allow")

        # Ambiguous — let it through with a soft flag
        logger.info(f"[GUARDRAIL] Ambiguous query (soft warn): '{query[:60]}'")
        return ContentModerationResult(
            is_safe=True,
            action="warn",
            flags=["ambiguous_topic"],
        )

    except Exception as e:
        # Fail open — don't block on guardrail errors
        logger.error(f"[GUARDRAIL] Content check error (failing open): {e}")
        return ContentModerationResult(is_safe=True, action="allow")


# ═════════════════════════════════════════════════════════════════════
# OUTPUT GUARDRAILS
# ═════════════════════════════════════════════════════════════════════

def validate_agent_output(result: dict) -> GuardrailResult:
    """
    Validate the agent's output dict before returning it to the UI.
    Uses the AgentResult Pydantic schema.

    Also post-processes highlight outputs through HighlightOutput.from_raw()
    to coerce any malformed highlight items.
    """
    try:
        task = result.get("task", "unknown")

        # Special handling for highlight output — coerce malformed items
        if task == "highlight_agent":
            raw_content = result.get("output", [])
            if isinstance(raw_content, list):
                validated_highlights = HighlightOutput.from_raw(raw_content)
                result["output"] = [h.model_dump() for h in validated_highlights.content]

        # Validate the full result
        AgentResult(**result)
        return GuardrailResult(passed=True)

    except Exception as e:
        logger.warning(f"[GUARDRAIL] Output validation failed: {e}")
        # Don't block the output — just warn and return as-is
        return GuardrailResult(passed=False, reason=str(e), blocked=False)


def sanitize_output(content: Any) -> Any:
    """
    Light sanitization on output text before rendering in the UI.
    Removes any accidentally leaked system-prompt artefacts or control chars.
    """
    if not isinstance(content, str):
        return content  # lists (highlights) pass through unchanged

    # Remove common prompt-leak patterns
    LEAK_PATTERNS = [
        r"<\|im_start\|>.*?<\|im_end\|>",  # ChatML tokens
        r"<<SYS>>.*?<</SYS>>",              # Llama system tags
        r"\[INST\].*?\[/INST\]",            # Llama inst tags
        r"Human:.*?Assistant:",             # conversational prefixes
    ]
    cleaned = content
    for pat in LEAK_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.DOTALL)

    # Strip null bytes and control characters (keep newlines/tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned.strip()


# ═════════════════════════════════════════════════════════════════════
# COMBINED PIPELINE GUARD
# ═════════════════════════════════════════════════════════════════════

def run_input_pipeline(
    query: str,
    session_id: str | None = None,
) -> tuple[bool, str, str | None]:
    """
    Run all input guardrails in sequence.

    Returns:
        (ok, clean_query, error_message)
        ok=True  → clean_query is the sanitized query, error_message=None
        ok=False → clean_query is the original, error_message explains why
    """
    # 1. Pydantic validation
    vr = validate_query(query, session_id=session_id)
    if not vr.passed:
        return False, query, f"Invalid query: {vr.reason}"

    clean_query = vr.reason  # reason holds the cleaned query text

    # 2. Content moderation
    cm = check_content(clean_query)
    if cm.action == "block":
        return False, query, (
            "Your query was flagged as potentially harmful or off-topic. "
            "Please ask something about your media content."
        )

    return True, clean_query, None


def run_output_pipeline(result: dict) -> dict:
    """
    Run all output guardrails on the agent result.

    Always returns a dict (the original or a sanitized version).
    Never raises — errors are logged and the original is returned.
    """
    try:
        # Validate structure
        validate_agent_output(result)

        # Sanitize text content
        content = result.get("output")
        result["output"] = sanitize_output(content)
        return result

    except Exception as e:
        logger.error(f"[GUARDRAIL] Output pipeline error: {e}")
        return result  # fail open
