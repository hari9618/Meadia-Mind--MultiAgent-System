# evaluation.py
# ─────────────────────────────────────────────────────────────────────
# Production-safe RAG Evaluation — RAGAS 0.4.x compatible
#
# RAGAS 0.4.3 API facts (verified from source):
#   Input columns : user_input, response, retrieved_contexts  (NOT question/answer/contexts)
#   LLM wiring    : pass llm= and embeddings= to evaluate() directly, NOT per-metric
#   Result object : EvaluationResult dataclass
#     .scores       → List[Dict[str, float]]   one dict per dataset row
#     ._repr_dict   → Dict[str, float]         averaged float per metric  ← USE THIS
#     .to_pandas()  → DataFrame                fallback
#   allow_nest_asyncio=False → required in production threads (no Jupyter)
#
# PRODUCTION METRIC TABLE (no ground truth needed):
#   ┌──────────────────────────────────────┬───────────────┐
#   │ Metric class                         │ Needs ref?    │
#   ├──────────────────────────────────────┼───────────────┤
#   │ Faithfulness                         │ NO  ✅        │
#   │ ResponseRelevancy                    │ NO  ✅        │
#   │ LLMContextPrecisionWithoutReference  │ NO  ✅        │
#   │ LLMContextPrecision                  │ YES ❌ skip   │
#   │ LLMContextRecall                     │ YES ❌ skip   │
#   └──────────────────────────────────────┴───────────────┘
#   Hallucination = 1 - faithfulness  (derived, free)
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from tracer import get_tracer

logger = logging.getLogger(__name__)

# ── Feature flags ─────────────────────────────────────────────────────
EVAL_ENABLED         = os.getenv("EVAL_ENABLED",         "true").lower()  == "true"
EVAL_LOG_TO_LANGFUSE = os.getenv("EVAL_LOG_TO_LANGFUSE", "true").lower()  == "true"
EVAL_COLLECT_DATASET = os.getenv("EVAL_COLLECT_DATASET", "false").lower() == "true"
EVAL_DATASET_PATH    = os.getenv("EVAL_DATASET_PATH",    "eval_dataset.jsonl")

# ── Thresholds ────────────────────────────────────────────────────────
THRESH_FAITHFULNESS  = float(os.getenv("EVAL_THRESH_FAITHFULNESS",  "0.70"))
THRESH_PRECISION     = float(os.getenv("EVAL_THRESH_PRECISION",     "0.60"))
THRESH_RELEVANCY     = float(os.getenv("EVAL_THRESH_RELEVANCY",     "0.65"))
THRESH_HALLUCINATION = float(os.getenv("EVAL_THRESH_HALLUCINATION", "0.30"))


# ── Result container ──────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Structured RAGAS scores for one production query.
    None means the metric could not be computed (never means 0.0).
    """
    trace_id:            str
    query:               str
    faithfulness:        Optional[float] = None
    answer_relevancy:    Optional[float] = None
    context_precision:   Optional[float] = None
    hallucination_score: Optional[float] = None   # derived: 1 - faithfulness
    latency_ms:          float           = 0.0
    error:               Optional[str]   = None
    alerts:              list[str]       = field(default_factory=list)
    skipped:             bool            = False

    def to_dict(self) -> dict:
        return {
            "trace_id":            self.trace_id,
            "query":               self.query,
            "faithfulness":        self.faithfulness,
            "answer_relevancy":    self.answer_relevancy,
            "context_precision":   self.context_precision,
            "hallucination_score": self.hallucination_score,
            "latency_ms":          round(self.latency_ms, 2),
            "error":               self.error,
            "alerts":              self.alerts,
            "skipped":             self.skipped,
        }

    def has_alerts(self) -> bool:
        return bool(self.alerts)


# ── LLM / Embeddings builders ─────────────────────────────────────────

def _build_ragas_llm():
    """Wrap our Groq LLM for RAGAS. Called lazily — never at import time."""
    from ragas.llms import LangchainLLMWrapper
    from llm import get_llm
    return LangchainLLMWrapper(get_llm(mode="balanced"))


def _build_ragas_embeddings():
    """Wrap our local sentence-transformer embeddings for RAGAS."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from config import EMBED_MODEL
    return LangchainEmbeddingsWrapper(
        SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    )


# ── Score extractor for RAGAS 0.4.x EvaluationResult ─────────────────

def _extract_scores(ragas_result) -> dict:
    """
    Extract metric scores from a RAGAS 0.4.x EvaluationResult object.

    RAGAS 0.4.3 EvaluationResult is a dataclass with:
      .scores      → List[Dict[str, float]]   per-row scores
      ._repr_dict  → Dict[str, float]         averaged scores  ← primary path
      .to_pandas() → DataFrame                fallback

    Returns plain Dict[str, float] or raises if nothing works.
    """
    # ── Primary: _repr_dict has already-averaged floats ──────────────
    if hasattr(ragas_result, "_repr_dict") and ragas_result._repr_dict:
        scores = {}
        for k, v in ragas_result._repr_dict.items():
            if v is not None:
                try:
                    scores[k] = float(v)
                except (TypeError, ValueError):
                    pass
        if scores:
            logger.debug(f"[EVAL] Extracted via _repr_dict: {scores}")
            return scores

    # ── Fallback 1: .scores[0] — first row dict ───────────────────────
    if hasattr(ragas_result, "scores") and ragas_result.scores:
        row = ragas_result.scores[0]
        if isinstance(row, dict):
            scores = {}
            for k, v in row.items():
                if v is not None:
                    try:
                        import math
                        if not math.isnan(float(v)):
                            scores[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            if scores:
                logger.debug(f"[EVAL] Extracted via scores[0]: {scores}")
                return scores

    # ── Fallback 2: pandas DataFrame ─────────────────────────────────
    if hasattr(ragas_result, "to_pandas"):
        try:
            df = ragas_result.to_pandas()
            scores = {}
            for col in df.columns:
                val = df[col].iloc[0]
                try:
                    import math
                    fval = float(val)
                    if not math.isnan(fval):
                        scores[col] = fval
                except (TypeError, ValueError):
                    pass
            if scores:
                logger.debug(f"[EVAL] Extracted via to_pandas: {scores}")
                return scores
        except Exception as e:
            logger.debug(f"[EVAL] to_pandas extraction failed: {e}")

    raise ValueError(
        f"Cannot extract scores from RAGAS result. "
        f"Type={type(ragas_result)}, attrs={[a for a in dir(ragas_result) if not a.startswith('__')]}"
    )


# ── Core RAGAS call ───────────────────────────────────────────────────

def _run_ragas_sync(
    question: str,
    answer:   str,
    contexts: list[str],
) -> dict:
    """
    Run RAGAS 0.4.x evaluation with 3 reference-free metrics.

    Returns Dict[str, float] with metric scores.
    Raises on failure — caller handles exceptions.

    RAGAS 0.4.3 API:
      - Pass llm= and embeddings= directly to evaluate() (NOT per metric)
      - Input dataset uses: user_input, response, retrieved_contexts
      - allow_nest_asyncio=False in production threads
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
    )

    ragas_llm        = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]

    # RAGAS 0.4.x column names
    data = {
        "user_input":         [question],
        "response":           [answer],
        "retrieved_contexts": [contexts],
    }
    dataset = Dataset.from_dict(data)

    logger.info(f"[EVAL] Running RAGAS on dataset: {dataset}")

    ragas_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,       # return NaN rows instead of crashing
        show_progress=True,
        allow_nest_asyncio=False,     # production threads — no Jupyter compat needed
    )

    logger.info(f"[EVAL] RAGAS raw result: {ragas_result}")

    scores = _extract_scores(ragas_result)
    logger.info(f"[EVAL] Extracted scores: {scores}")
    return scores


# ── Threshold monitor ─────────────────────────────────────────────────

def _check_thresholds(result: EvaluationResult) -> None:
    """Fire WARNING logs and append UI alerts for out-of-threshold scores."""

    if result.faithfulness is not None and result.faithfulness < THRESH_FAITHFULNESS:
        logger.warning(
            f"[EVAL][ALERT] Low faithfulness={result.faithfulness:.2f} "
            f"threshold={THRESH_FAITHFULNESS} | trace={result.trace_id}"
        )
        result.alerts.append(
            f"⚠️ Low faithfulness ({result.faithfulness:.0%}) — answer not fully grounded in context"
        )

    if result.hallucination_score is not None and result.hallucination_score > THRESH_HALLUCINATION:
        logger.warning(
            f"[EVAL][ALERT] High hallucination={result.hallucination_score:.2f} "
            f"threshold={THRESH_HALLUCINATION} | trace={result.trace_id}"
        )
        result.alerts.append(
            f"🚨 High hallucination risk ({result.hallucination_score:.0%})"
        )

    if result.context_precision is not None and result.context_precision < THRESH_PRECISION:
        logger.warning(
            f"[EVAL][ALERT] Low context_precision={result.context_precision:.2f} "
            f"threshold={THRESH_PRECISION} | trace={result.trace_id}"
        )
        result.alerts.append(
            f"⚠️ Low context precision ({result.context_precision:.0%}) — retrieval pulling noisy chunks"
        )

    if result.answer_relevancy is not None and result.answer_relevancy < THRESH_RELEVANCY:
        logger.warning(
            f"[EVAL][ALERT] Low answer_relevancy={result.answer_relevancy:.2f} "
            f"threshold={THRESH_RELEVANCY} | trace={result.trace_id}"
        )
        result.alerts.append(
            f"⚠️ Low answer relevancy ({result.answer_relevancy:.0%}) — answer may be off-topic"
        )


# ── Langfuse score logger ─────────────────────────────────────────────

def _log_scores_to_langfuse(result: EvaluationResult) -> None:
    """Post each RAGAS score as a named score on the Langfuse trace."""
    if not EVAL_LOG_TO_LANGFUSE:
        return
    tracer = get_tracer()
    score_map = {
        "ragas_faithfulness":      result.faithfulness,
        "ragas_answer_relevancy":  result.answer_relevancy,
        "ragas_context_precision": result.context_precision,
        "ragas_hallucination":     result.hallucination_score,
    }
    for name, value in score_map.items():
        if value is not None:
            tracer.score_response(
                trace_id=result.trace_id,
                name=name,
                value=round(value, 4),
                comment=f"RAGAS auto-eval | latency={result.latency_ms:.0f}ms",
            )
    if result.alerts:
        tracer.score_response(
            trace_id=result.trace_id,
            name="ragas_alert_count",
            value=float(len(result.alerts)),
            comment=" | ".join(result.alerts),
        )


# ── EvaluationService ─────────────────────────────────────────────────

class EvaluationService:
    """
    Reusable evaluation service. Only reference-free metrics — safe for
    every production query. No ground truth required.

    Call patterns:
        evaluate()                  → sync, blocking  (CLI / tests)
        evaluate_async()            → async           (FastAPI)
        evaluate_fire_and_forget()  → daemon thread   (Streamlit ← use this)
    """

    def __init__(self):
        if not EVAL_ENABLED:
            logger.info("[EVAL] Disabled via EVAL_ENABLED=false")

    def evaluate(
        self,
        trace_id: str,
        query:    str,
        answer:   str,
        contexts: list[str],
    ) -> EvaluationResult:
        """
        Run RAGAS evaluation synchronously and return structured result.
        Logs scores to Langfuse automatically.
        Never raises — always returns EvaluationResult.
        """
        if not EVAL_ENABLED:
            return EvaluationResult(trace_id=trace_id, query=query, skipped=True)

        # Normalise answer — highlight agent returns list[dict]
        if isinstance(answer, list):
            answer = " ".join(
                item.get("highlight", "") if isinstance(item, dict) else str(item)
                for item in answer
            )

        answer = str(answer).strip()

        if not answer or not contexts:
            logger.warning("[EVAL] Skipping — empty answer or no contexts")
            return EvaluationResult(
                trace_id=trace_id, query=query,
                skipped=True, error="Empty answer or context"
            )

        t0     = time.perf_counter()
        result = EvaluationResult(trace_id=trace_id, query=query)

        try:
            scores = _run_ragas_sync(
                question=query,
                answer=answer,
                contexts=contexts,
            )

            # Map RAGAS metric names → our result fields.
            # Terminal log confirmed RAGAS 0.4.3 returns these exact keys:
            #   "faithfulness"
            #   "answer_relevancy"                          ← NOT response_relevancy
            #   "llm_context_precision_without_reference"
            result.faithfulness = scores.get("faithfulness")

            result.answer_relevancy = (
                scores.get("answer_relevancy")                       # confirmed 0.4.3 key
                or scores.get("response_relevancy")                  # fallback
            )
            result.context_precision = (
                scores.get("llm_context_precision_without_reference") # confirmed 0.4.3 key
                or scores.get("context_precision")                    # fallback
            )

            # Derive hallucination from faithfulness
            if result.faithfulness is not None:
                result.hallucination_score = round(1.0 - result.faithfulness, 4)

        except Exception as e:
            result.error = str(e)
            logger.error(f"[EVAL] RAGAS failed for trace_id={trace_id}: {e}", exc_info=True)

        finally:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[EVAL] Completed {result.latency_ms:.0f}ms | "
                f"faithfulness={result.faithfulness} | "
                f"relevancy={result.answer_relevancy} | "
                f"precision={result.context_precision} | "
                f"hallucination={result.hallucination_score} | "
                f"trace_id={trace_id}"
            )

        _check_thresholds(result)

        try:
            _log_scores_to_langfuse(result)
        except Exception as e:
            logger.warning(f"[EVAL] Langfuse logging failed (non-critical): {e}")

        return result

    async def evaluate_async(
        self,
        trace_id: str,
        query:    str,
        answer:   str,
        contexts: list[str],
    ) -> EvaluationResult:
        """Async wrapper — RAGAS runs in thread pool, event loop stays unblocked."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.evaluate(
                trace_id=trace_id,
                query=query,
                answer=answer,
                contexts=contexts,
            )
        )

    def evaluate_fire_and_forget(
        self,
        trace_id: str,
        query:    str,
        answer:   str,
        contexts: list[str],
    ) -> None:
        """
        Launch evaluation in a background daemon thread.
        Returns immediately — user sees their answer without waiting.
        Scores appear in Langfuse ~10-20s later.

        Use this in Streamlit (no asyncio event loop available).
        """
        def _run():
            try:
                self.evaluate(
                    trace_id=trace_id,
                    query=query,
                    answer=answer,
                    contexts=contexts,
                )
            except Exception as e:
                logger.error(f"[EVAL] Background thread error: {e}", exc_info=True)

        t = threading.Thread(
            target=_run,
            daemon=True,
            name=f"ragas-eval-{trace_id[:8]}"
        )
        t.start()
        logger.info(f"[EVAL] Background thread started | trace_id={trace_id}")


# ── Singleton ─────────────────────────────────────────────────────────

_eval_service: Optional[EvaluationService] = None


def get_eval_service() -> EvaluationService:
    global _eval_service
    if _eval_service is None:
        _eval_service = EvaluationService()
    return _eval_service


# ── Offline dataset collection ────────────────────────────────────────

@dataclass
class EvalDatasetCollector:
    """
    Collect production Q/A/context samples for offline regression testing.
    Enable with EVAL_COLLECT_DATASET=true in .env
    """
    rows: list[dict] = field(default_factory=list)

    def add(
        self,
        question:  str,
        contexts:  list[str],
        answer:    str,
        metadata:  Optional[dict] = None,
    ) -> None:
        self.rows.append({
            "question":  question,
            "contexts":  contexts,
            "answer":    answer,
            "metadata":  metadata or {},
            "timestamp": time.time(),
        })

    def save(self, filepath: str) -> int:
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"[EVAL] Saved {len(self.rows)} rows → {filepath}")
        return len(self.rows)

    @classmethod
    def load(cls, filepath: str) -> "EvalDatasetCollector":
        import json
        c = cls()
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                c.rows.append(json.loads(line.strip()))
        return c

    def run_offline_evaluation(self) -> list[EvaluationResult]:
        """Batch-evaluate all rows — use for CI/CD regression tests."""
        svc = get_eval_service()
        results = []
        for i, row in enumerate(self.rows):
            logger.info(f"[EVAL] Offline {i+1}/{len(self.rows)}")
            results.append(svc.evaluate(
                trace_id=f"offline_{i}",
                query=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
            ))
        return results