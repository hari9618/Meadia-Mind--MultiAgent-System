# rag.py
# ─────────────────────────────────────────────────────────────────────
# RAG (Retrieval-Augmented Generation) pipeline.
#
# Changes vs v1:
#   ✦ retrieve_context_with_chunks() — new function returns BOTH the
#     joined string (for LLM prompt) AND the raw list (for RAGAS eval)
#   ✦ retrieve_context() preserved unchanged — no breaking changes
#   ✦ Retrieval latency logging added for Langfuse spans
#
# What RAG does:
#   1. Takes a long transcript and splits it into small chunks
#   2. Stores those chunks in a vector database (ChromaDB)
#   3. When a query comes in, finds the most relevant chunks
#   4. Returns those chunks as "context" for the LLM to use
#
# Why HYBRID search:
#   BM25 catches exact keyword matches.
#   ChromaDB catches semantic/meaning matches.
#   Together they're significantly more accurate than either alone.
#
# Production note on reranking:
#   For even better retrieval quality, add a cross-encoder reranker
#   between the merge step and the final TOP_K slice.
#   See the recommendation section in evaluation.py.
#   Example: pip install sentence-transformers
#   Then: from sentence_transformers import CrossEncoder
#         reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#         scores = reranker.predict([(query, chunk) for chunk in merged])
#         final_chunks = [merged[i] for i in argsort(scores)[-TOP_K:]]
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

# ── Module-level storage for BM25 ─────────────────────────────────────
_bm25_index: BM25Okapi | None = None
_bm25_chunks: List[str] = []


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
        logger.warning("[RAG] BM25 index not built yet. Returning empty list.")
        return []

    query_tokens = query.lower().split()
    scores       = _bm25_index.get_scores(query_tokens)
    top_indices  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [_bm25_chunks[i] for i in top_indices]


def _hybrid_retrieve_chunks(query: str) -> List[str]:
    """
    Internal: perform hybrid retrieval and return deduplicated list of chunks.
    Extracted so both retrieve_context() and retrieve_context_with_chunks()
    call the same logic — DRY principle.
    """
    t0 = time.perf_counter()
    collection = _get_or_create_collection()

    # Vector search (semantic similarity)
    vector_results = collection.query(query_texts=[query], n_results=TOP_K)
    vector_chunks  = vector_results["documents"][0] if vector_results["documents"] else []

    # BM25 search (keyword matching)
    bm25_chunks = _bm25_retrieve(query, top_k=TOP_K)

    # Merge and deduplicate — vector chunks have priority
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

    final_chunks = merged[:TOP_K]
    latency_ms   = (time.perf_counter() - t0) * 1000

    if not final_chunks:
        logger.warning("[RAG] No chunks retrieved. Returning empty list.")
        return []

    logger.info(
        f"[RAG] Retrieved {len(final_chunks)} chunks (hybrid: vector + BM25) "
        f"in {latency_ms:.0f}ms"
    )
    return final_chunks


def retrieve_context(query: str) -> str:
    """
    Original API — preserved for backward compatibility.
    Returns the joined context string only.

    Use retrieve_context_with_chunks() when you also need the raw chunk
    list for RAGAS evaluation.
    """
    final_chunks = _hybrid_retrieve_chunks(query)

    if not final_chunks:
        return "No relevant context found."

    return "\n\n---\n\n".join(final_chunks)


def retrieve_context_with_chunks(query: str) -> Tuple[str, List[str]]:
    """
    NEW: Returns both the joined context string AND the raw chunk list.

    Why two return values?
        - LLM prompts need a single concatenated string (context)
        - RAGAS evaluation needs list[str] (one string per chunk)
        Returning both from one retrieval call avoids doing it twice.

    Usage in app.py / agent.py:
        context, chunks = retrieve_context_with_chunks(query)
        result = run_agent(query, context, num_chunks, context_chunks=chunks)

    Args:
        query: The user's question or task description.

    Returns:
        Tuple of:
            - context_str:    Top chunks joined by separator (for LLM prompt)
            - context_chunks: Raw list of chunk strings (for RAGAS)
    """
    final_chunks = _hybrid_retrieve_chunks(query)

    if not final_chunks:
        return "No relevant context found.", []

    context_str = "\n\n---\n\n".join(final_chunks)
    return context_str, final_chunks