# tracer.py
# ─────────────────────────────────────────────────────────────────────
# Langfuse Python SDK v4 observability wrapper for MediaMind.
#
# SDK v4 (current) key facts:
#   - Client:    from langfuse import get_client  -> get_client()
#   - Tracing:   langfuse.start_as_current_observation(as_type="span", ...)
#                NOT langfuse.trace()  -- that method no longer exists in v4
#   - LangChain: from langfuse.langchain import CallbackHandler
#                NOT from langfuse.callback import CallbackHandler
#   - Env var:   LANGFUSE_BASE_URL  (NOT LANGFUSE_HOST)
#
# Zero-impact when LANGFUSE keys are not set --
# all methods become no-ops so the app runs normally in dev.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Try to import Langfuse v4
try:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    logger.info("[TRACER] langfuse not installed -- observability disabled.")


# ── Null implementations (used when Langfuse is unavailable/disabled) ─

class _NullSpan:
    def __init__(self): self.id = str(uuid.uuid4())
    def end(self, **kwargs): pass
    def update(self, **kwargs): pass
    def event(self, **kwargs): pass
    def score(self, **kwargs): pass
    def generation(self, **kwargs): return _NullSpan()
    def span(self, **kwargs): return _NullSpan()
    def __enter__(self): return self
    def __exit__(self, *a): pass


class _NullTracer:
    @contextmanager
    def trace(self, name: str, **kwargs):
        yield _NullSpan()
    def langchain_handler(self, **kwargs): return None
    def log_guardrail_event(self, *args, **kwargs): pass
    def log_routing_decision(self, *args, **kwargs): pass
    def log_rag_retrieval(self, *args, **kwargs): pass
    def score_response(self, *args, **kwargs): pass
    def flush(self): pass


# ── Real Langfuse v4 tracer ───────────────────────────────────────────

class MediaMindTracer:
    """
    Wrapper around the Langfuse Python SDK v4.

    v4 tracing API (langfuse.trace() no longer exists):
        langfuse = get_client()
        with langfuse.start_as_current_observation(as_type="span", name="...") as span:
            span.update(output="...")
        langfuse.flush()

    LangChain/LangGraph callback:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()
        graph.invoke(state, config={"callbacks": [handler]})
    """

    def __init__(self):
        self._client = get_client()
        logger.info("[TRACER] Langfuse v4 observability initialized.")

    @contextmanager
    def trace(
        self,
        name: str,
        query: str = "",
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> Generator[Any, None, None]:
        """
        Context manager that wraps the full pipeline in a root Langfuse span.
        Yields the span so callers can attach nested child observations.

        Usage:
            with tracer.trace("run_agent", query=q, session_id=sid) as span:
                handler = tracer.langchain_handler()
                graph.invoke(state, config={"callbacks": [handler]})
        """
        start = time.perf_counter()
        try:
            self._client.update_current_trace(
                name=name,
                input={"query": query},
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {},
                tags=["mediamind", "production"],
            )
        except Exception:
            pass

        with self._client.start_as_current_observation(
            as_type="span",
            name=name,
            input={"query": query},
            metadata={**(metadata or {}), "session_id": session_id},
        ) as root_span:
            try:
                yield root_span
            except Exception as e:
                try:
                    root_span.update(output={"error": str(e)}, level="ERROR")
                except Exception:
                    pass
                raise
            finally:
                elapsed = time.perf_counter() - start
                try:
                    root_span.update(metadata={"latency_seconds": round(elapsed, 3)})
                except Exception:
                    pass

        try:
            self._client.flush()
        except Exception:
            pass

    def langchain_handler(
        self,
        trace_id: str | None = None,
        session_id: str | None = None,
    ):
        """
        Returns a LangChain CallbackHandler for automatic LLM/tool tracing.
        Pass to graph.invoke(state, config={"callbacks": [handler]}).
        """
        try:
            return LangfuseCallbackHandler()
        except Exception as e:
            logger.warning(f"[TRACER] Could not create CallbackHandler: {e}")
            return None

    def log_guardrail_event(self, trace: Any, stage: str, passed: bool, reason: str = "") -> None:
        try:
            with self._client.start_as_current_observation(
                as_type="span",
                name=f"guardrail_{stage}",
                input={"stage": stage},
                level="DEFAULT" if passed else "WARNING",
            ) as span:
                span.update(output={"passed": passed, "reason": reason})
        except Exception as e:
            logger.debug(f"[TRACER] guardrail event failed: {e}")

    def log_routing_decision(self, trace: Any, query: str, decision: str, latency_ms: float) -> None:
        try:
            with self._client.start_as_current_observation(
                as_type="span",
                name="supervisor_routing",
                input={"query": query},
                metadata={"latency_ms": round(latency_ms, 2)},
            ) as span:
                span.update(output={"routed_to": decision})
        except Exception as e:
            logger.debug(f"[TRACER] routing span failed: {e}")

    def log_rag_retrieval(self, trace: Any, query: str, chunks: list[str], latency_ms: float) -> None:
        try:
            with self._client.start_as_current_observation(
                as_type="span",
                name="hybrid_rag_retrieval",
                input={"query": query},
                metadata={"latency_ms": round(latency_ms, 2)},
            ) as span:
                span.update(output={
                    "num_chunks": len(chunks),
                    "chunk_preview": [c[:100] for c in chunks[:2]],
                })
        except Exception as e:
            logger.debug(f"[TRACER] RAG span failed: {e}")

    def score_response(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        # Langfuse v4: .score() was renamed to .create_score()
        # .score() silently did nothing — scores appeared in terminal but never in dashboard
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type="NUMERIC",
                comment=comment,
            )
        except Exception as e:
            logger.debug(f"[TRACER] score failed: {e}")

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as e:
            logger.debug(f"[TRACER] flush failed: {e}")


# ── Singleton factory ──────────────────────────────────────────────────

_tracer: "MediaMindTracer | _NullTracer | None" = None


def get_tracer() -> "MediaMindTracer | _NullTracer":
    """
    Return the singleton tracer instance.

    Required env vars:
        LANGFUSE_PUBLIC_KEY = pk-lf-...
        LANGFUSE_SECRET_KEY = sk-lf-...
        LANGFUSE_BASE_URL   = https://cloud.langfuse.com   <- NOT LANGFUSE_HOST
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    langfuse_configured = (
        _LANGFUSE_AVAILABLE
        and os.environ.get("LANGFUSE_SECRET_KEY")
        and os.environ.get("LANGFUSE_PUBLIC_KEY")
    )

    if langfuse_configured:
        try:
            _tracer = MediaMindTracer()
        except Exception as e:
            logger.warning(f"[TRACER] Langfuse init failed, using null tracer: {e}")
            _tracer = _NullTracer()
    else:
        logger.info("[TRACER] Langfuse not configured -- using null tracer.")
        _tracer = _NullTracer()

    return _tracer