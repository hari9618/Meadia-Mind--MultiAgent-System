# rag.py
# ─────────────────────────────────────────────────────────────────────
# RAG (Retrieval-Augmented Generation) pipeline.
#
# What RAG does:
#   1. Takes a long transcript and splits it into small chunks
#   2. Stores those chunks in a vector database (ChromaDB)
#   3. When a query comes in, finds the most relevant chunks
#   4. Returns those chunks as "context" for the LLM to use
#
# Why HYBRID search (new vs your old code):
#   Old: pure semantic/vector search — misses exact keyword matches
#   New: BM25 keyword search + vector search merged together
#        BM25 catches exact terms, vectors catch meaning/synonyms
#        Together they are significantly more accurate
# ─────────────────────────────────────────────────────────────────────

import logging
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi  # BM25 keyword search algorithm

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
# PersistentClient saves all vectors to ./chroma_store on disk.
# Data survives server restarts — no re-indexing needed on startup.
_chroma_client = chromadb.PersistentClient(path="./chroma_store")

# Embedding function: converts text → numbers (vectors) for semantic search
# all-MiniLM-L6-v2 is free, runs locally, no API key needed
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# Text splitter: breaks a long transcript into overlapping chunks
# RecursiveCharacterTextSplitter tries to split on paragraphs first,
# then sentences, then words — respecting natural language boundaries
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)

# ── Module-level storage for BM25 ─────────────────────────────────────
# BM25 is an in-memory index, so we keep the chunks and index here
_bm25_index: BM25Okapi | None = None
_bm25_chunks: List[str] = []


def _get_or_create_collection() -> chromadb.Collection:
    """Return existing ChromaDB collection or create a fresh one."""
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
    """
    Wipe the entire ChromaDB collection and reset BM25.
    Called before indexing new content so old chunks never
    contaminate retrieval for a different document.
    """
    global _bm25_index, _bm25_chunks
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
        logger.info("[RAG] Cleared existing ChromaDB collection.")
    except Exception:
        pass  # collection didn't exist yet — that's fine
    _bm25_index  = None
    _bm25_chunks = []


def store_transcript(transcript: str, doc_id: str = "doc_001") -> int:
    """
    Wipe the collection, then chunk and store the new transcript.
    Always starts fresh — no stale chunks from previous documents.

    Args:
        transcript : The full text of the transcript.
        doc_id     : A unique ID label for this document.

    Returns:
        Number of chunks stored.
    """
    global _bm25_index, _bm25_chunks

    # ── Always wipe first so old content never bleeds into new queries ──
    clear_collection()
    collection = _get_or_create_collection()

    # Split the transcript into chunks
    chunks    = _splitter.split_text(transcript)
    ids       = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk_index": i} for i in range(len(chunks))]

    # Store in ChromaDB (handles vector embeddings automatically)
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info(f"[RAG] Stored {len(chunks)} chunks for '{doc_id}' in ChromaDB")

    # Build BM25 index from the same chunks
    _bm25_chunks = chunks
    _bm25_index  = _build_bm25(chunks)
    logger.info(f"[RAG] BM25 index built with {len(chunks)} chunks")

    return len(chunks)


def _build_bm25(chunks: List[str]) -> BM25Okapi:
    """
    Build a BM25 index from a list of text chunks.
    BM25 tokenizes each chunk into words for keyword matching.
    """
    tokenized = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def _bm25_retrieve(query: str, top_k: int) -> List[str]:
    """
    Retrieve top-K chunks using BM25 keyword search.
    Returns the most keyword-relevant chunks for the query.
    """
    if _bm25_index is None or not _bm25_chunks:
        logger.warning("[RAG] BM25 index not built yet. Returning empty list.")
        return []

    # Tokenize query the same way we tokenized chunks
    query_tokens = query.lower().split()
    scores       = _bm25_index.get_scores(query_tokens)

    # Get indices of top-K highest-scoring chunks
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [_bm25_chunks[i] for i in top_indices]


def retrieve_context(query: str) -> str:
    """
    HYBRID retrieval: merge BM25 keyword results + ChromaDB vector results.

    How it works:
      1. BM25 finds chunks with exact keyword matches
      2. ChromaDB finds chunks with similar meaning (semantic)
      3. We merge both lists, deduplicate, and return top chunks
      4. This catches both "exact term" and "similar meaning" matches

    Args:
        query : The user's question or task description.

    Returns:
        Top chunks joined as a single context string, ready for the LLM.
    """
    collection = _get_or_create_collection()

    # ── Vector search (semantic) ──────────────────────────────────────
    vector_results = collection.query(query_texts=[query], n_results=TOP_K)
    vector_chunks  = vector_results["documents"][0] if vector_results["documents"] else []

    # ── BM25 search (keyword) ─────────────────────────────────────────
    bm25_chunks = _bm25_retrieve(query, top_k=TOP_K)

    # ── Merge and deduplicate ─────────────────────────────────────────
    # Use a dict to deduplicate while preserving insertion order
    # Vector chunks get priority (inserted first), BM25 fills in the gaps
    seen   = {}
    merged = []

    for chunk in vector_chunks:
        key = chunk[:100]  # use first 100 chars as dedup key
        if key not in seen:
            seen[key] = True
            merged.append(chunk)

    for chunk in bm25_chunks:
        key = chunk[:100]
        if key not in seen:
            seen[key] = True
            merged.append(chunk)

    # Take the top TOP_K chunks from the merged list
    final_chunks = merged[:TOP_K]

    if not final_chunks:
        logger.warning("[RAG] No chunks retrieved. Returning empty context.")
        return "No relevant context found."

    logger.info(f"[RAG] Retrieved {len(final_chunks)} chunks (hybrid: vector + BM25)")
    return "\n\n---\n\n".join(final_chunks)
