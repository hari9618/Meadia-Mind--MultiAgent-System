# llm.py
# ─────────────────────────────────────────────────────────────────────
# Groq LLM client.
# Groq runs Llama 3.3 70B on custom LPU hardware — much faster than
# GPU-hosted APIs. We use three pre-built instances (one per temperature
# mode) so we don't rebuild the client on every call.
# ─────────────────────────────────────────────────────────────────────

import json
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import (
    GROQ_API_KEY,
    GROQ_MODEL_FAST,
    TEMP_PRECISE,
    TEMP_BALANCED,
    TEMP_CREATIVE,
    LLM_PARAMS,
    RETRY_ATTEMPTS,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
)

logger = logging.getLogger(__name__)


def _build_llm(temperature: float) -> ChatGroq:
    """
    Build a Groq LLM instance with a specific temperature.
    Called once at module load — not on every API call.
    """
    return ChatGroq(
        model=GROQ_MODEL_FAST,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=LLM_PARAMS["max_tokens"],
        # Note: Groq does not support top_p in the same way, skip it
    )


# Three pre-built LLM instances — one per task type
# Building them once at startup avoids repeated client initialization
_llm_precise  = _build_llm(TEMP_PRECISE)    # tool selection, JSON output
_llm_balanced = _build_llm(TEMP_BALANCED)   # summaries, highlights
_llm_creative = _build_llm(TEMP_CREATIVE)   # social media content

# Output parser: strips the LangChain wrapper and returns a plain string
_parser = StrOutputParser()

# Map mode strings to LLM instances
_LLM_MAP = {
    "precise":  _llm_precise,
    "balanced": _llm_balanced,
    "creative": _llm_creative,
}


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,   # if all retries fail, raise the original exception
)
def call_llm(prompt: str, system: str = None, mode: str = "balanced") -> str:
    """
    Call Groq via LangChain with automatic retry + exponential backoff.

    Args:
        prompt : The user-facing content to send to the model.
        system : Optional system instruction (sets model role/behavior).
        mode   : "precise"  → temp 0.0  (routing, JSON)
                 "balanced" → temp 0.3  (summaries, highlights)
                 "creative" → temp 0.75 (social media)

    Returns:
        Raw text string from the model (stripped of whitespace).
    """
    llm = _LLM_MAP.get(mode, _llm_balanced)

    # Build the message list LangChain expects
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    # chain: llm → parser gives us a plain string back
    chain  = llm | _parser
    result = chain.invoke(messages)
    return result.strip()


def call_llm_for_json(prompt: str, system: str = None) -> dict | list:
    """
    Call Groq in precise mode and parse the response as JSON.
    Strips markdown code fences if the model adds them (it sometimes does).

    Used for: highlight extraction (returns a JSON array).

    Returns:
        Parsed Python dict or list from the model's JSON response.

    Raises:
        ValueError: if the model returns something that isn't valid JSON.
    """
    raw = call_llm(prompt, system=system, mode="precise")

    # Strip markdown fences like ```json ... ``` that some models add
    clean = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"[LLM] JSON parse failed. Raw response:\n{raw}")
        raise ValueError(f"Model returned invalid JSON: {e}") from e


def get_llm(mode: str = "balanced") -> ChatGroq:
    """
    Return the raw ChatGroq instance (used by LangChain chains in agents).
    Agents need the raw LLM object, not the call_llm() wrapper.
    """
    return _LLM_MAP.get(mode, _llm_balanced)
