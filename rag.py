# rag.py
# ─────────────────────────────────────────────────────────────────────
# RAG (Retrieval-Augmented Generation) pipeline.
#
# IMPROVEMENTS in this version (for better RAGAS scores):
#   ✦ _is_specific_question()       — detects direct Q&A vs broad tasks
#   ✦ Query-aware TOP_K             — specific questions use TOP_K=2
#                                     broad tasks use TOP_K=4
#                                     → directly improves Context Precision
#   ✦ _hybrid_retrieve_chunks()     — shared internal logic (DRY)
#   ✦ retrieve_context_with_chunks()— returns (str, List[str]) for RAGAS
#   ✦ retrieve_context()            — preserved for backward compatibility
#   ✦ Retrieval latency logging     — for Langfuse spans
#
# WHY QUERY-AWARE TOP_K IMPROVES CONTEXT PRECISION:
#   Before: "What was the error rate?" → 4 chunks → 1 relevant, 3 noise
#           RAGAS precision = 1/4 = 0.25
#   After:  "What was the error rate?" → 2 chunks → 1 relevant, 1 noise
#           RAGAS precision = 1/2 = 0.50+
#
# WHY HYBRID SEARCH:
#   BM25 catches exact keyword matches ("error rate", "34%")
#   ChromaDB catches semantic/meaning matches ("how much did it drop")
#   Together significantly more accurate than either alone.
# ─────────────────────────────────────────────────────────────────────

import logging
import time
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from config import (
    EMBED_MODEL,
    COLLECTION_NAME,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
)

logger = logging.getLogger(__name__)

# ── ChromaDB client (persistent) ─────────────────────────────────────
_chroma_client = chromadb.PersistentClient(path="./chroma_store")

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)

# ── Module-level BM25 storage ─────────────────────────────────────────
_bm25_index:  BM25Okapi | None = None
_bm25_chunks: List[str]        = []


def _get_or_create_collection() -> chromadb.Collection:
    try:
        return _chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedding_fn,
        )
    except Exception:
        return _chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedding_fn,
        )


def clear_collection() -> None:
    global _bm25_index, _bm25_chunks
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
        logger.info("[RAG] Cleared existing ChromaDB collection.")
    except Exception:
        pass
    _bm25_index  = None
    _bm25_chunks = []


def store_transcript(transcript: str, doc_id: str = "doc_001") -> int:
    global _bm25_index, _bm25_chunks

    clear_collection()
    collection = _get_or_create_collection()

    chunks    = _splitter.split_text(transcript)
    ids       = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info(f"[RAG] Stored {len(chunks)} chunks for '{doc_id}' in ChromaDB")

    _bm25_chunks = chunks
    _bm25_index  = _build_bm25(chunks)
    logger.info(f"[RAG] BM25 index built with {len(chunks)} chunks")

    return len(chunks)


def _build_bm25(chunks: List[str]) -> BM25Okapi:
    tokenized = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def _bm25_retrieve(query: str, top_k: int) -> List[str]:
    if _bm25_index is None or not _bm25_chunks:
        logger.warning("[RAG] BM25 index not built yet.")
        return []
    query_tokens = query.lower().split()
    scores       = _bm25_index.get_scores(query_tokens)
    top_indices  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [_bm25_chunks[i] for i in top_indices]


# ── Query-aware retrieval depth ───────────────────────────────────────

def _is_specific_question(query: str) -> bool:
    """
    Detect if the query is a specific direct question vs a broad task.

    WHY THIS IMPROVES CONTEXT PRECISION:
      Specific questions need 2 tight chunks — not 4 broad ones.
      Retrieving 4 chunks for "what was the error rate?" pulls 3 irrelevant
      chunks about unrelated parts of the transcript → precision tanks.

    Examples:
      "What was the error rate?"        → specific → TOP_K=2
      "Who is Dr. Sarah Chen?"          → specific → TOP_K=2
      "Summarize the entire episode"    → broad    → TOP_K=4
      "What are the key highlights?"    → broad    → TOP_K=4
    """
    question_starters = [
        "what", "who", "when", "where", "why", "how", "which",
        "did", "does", "is", "are", "was", "were", "define", "explain",
        "tell me", "can you tell",
    ]
    lower = query.lower().strip()
    has_question_word = any(
        lower.startswith(w) or f" {w} " in lower
        for w in question_starters
    )
    is_short = len(query.split()) < 15
    return has_question_word and is_short


def _hybrid_retrieve_chunks(query: str) -> List[str]:
    """
    Shared internal retrieval logic used by both public functions.
    Performs hybrid BM25 + vector search with query-aware TOP_K.

    Returns deduplicated list of chunks, vector results prioritised.
    """
    t0         = time.perf_counter()
    collection = _get_or_create_collection()

    # Query-aware retrieval depth:
    #   Specific Q&A → TOP_K=2  (tight, precise)
    #   Broad tasks  → TOP_K=4  (wide coverage)
    effective_top_k = 2 if _is_specific_question(query) else TOP_K
    query_type      = "specific" if effective_top_k == 2 else "broad"
    logger.info(f"[RAG] Query type={query_type} → effective_top_k={effective_top_k}")

    # ── Vector search (semantic similarity) ──────────────────────────
    vector_results = collection.query(
        query_texts=[query], n_results=effective_top_k
    )
    vector_chunks = (
        vector_results["documents"][0]
        if vector_results["documents"] else []
    )

    # ── BM25 search (keyword matching) ───────────────────────────────
    bm25_chunks = _bm25_retrieve(query, top_k=effective_top_k)

    # ── Merge + deduplicate (vector chunks have priority) ─────────────
    seen   = {}
    merged = []

    for chunk in vector_chunks:
        key = chunk[:100]
        if key not in seen:
            seen[key] = True
            merged.append(chunk)

    for chunk in bm25_chunks:
        key = chunk[:100]
        if key not in seen:
            seen[key] = True
            merged.append(chunk)

    final_chunks = merged[:effective_top_k]
    latency_ms   = (time.perf_counter() - t0) * 1000

    if not final_chunks:
        logger.warning("[RAG] No chunks retrieved.")
        return []

    logger.info(
        f"[RAG] Retrieved {len(final_chunks)} chunks "
        f"(hybrid: vector + BM25) in {latency_ms:.0f}ms"
    )
    return final_chunks


# ── Public API ────────────────────────────────────────────────────────

def retrieve_context(query: str) -> str:
    """
    Original API — preserved for backward compatibility.
    Returns the joined context string only.
    """
    final_chunks = _hybrid_retrieve_chunks(query)
    if not final_chunks:
        return "No relevant context found."
    return "\n\n---\n\n".join(final_chunks)


def retrieve_context_with_chunks(query: str) -> Tuple[str, List[str]]:
    """
    Returns BOTH the joined context string AND the raw chunk list.

    WHY TWO RETURN VALUES:
      - LLM prompt needs a single concatenated string  → context_str
      - RAGAS evaluation needs List[str]               → context_chunks
      One retrieval call, two uses — no double-fetching.

    Usage:
        context, chunks = retrieve_context_with_chunks(query)
        result = run_agent(query, context, num_chunks, context_chunks=chunks)
    """
    final_chunks = _hybrid_retrieve_chunks(query)
    if not final_chunks:
        return "No relevant context found.", []
    return "\n\n---\n\n".join(final_chunks), final_chunks