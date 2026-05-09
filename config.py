# config.py
# ─────────────────────────────────────────────────────────────────────
# Central configuration — change model names or tune parameters here.
# Every other file imports from this file, so you only edit ONE place.
# ─────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

load_dotenv()  # reads your .env file automatically

# ── Groq LLM ──────────────────────────────────────────────────────────
# Groq runs Llama on custom LPU hardware — very fast inference.
# Get your free key at: https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Primary model: fast, great for summarization & social content
GROQ_MODEL_FAST = "llama-3.3-70b-versatile"

# Reasoning model: slower but better for fact-checking & analysis
GROQ_MODEL_REASON = "deepseek-r1-distill-llama-70b"

# ── Temperature per task type ─────────────────────────────────────────
# Temperature controls creativity. 0.0 = fully deterministic, 1.0 = very random.
TEMP_PRECISE  = 0.0    # used for: tool selection, JSON output, routing
TEMP_BALANCED = 0.3    # used for: summaries, highlights — reliable + clear
TEMP_CREATIVE = 0.75   # used for: social media posts — engaging & varied

# ── LLM Parameters ────────────────────────────────────────────────────
LLM_PARAMS = {
    "max_tokens": 1024,   # max output length
    "top_p": 0.85,        # nucleus sampling — controls output diversity
}

# ── Retry configuration ───────────────────────────────────────────────
RETRY_ATTEMPTS = 3     # retry up to 3 times on API failure
RETRY_WAIT_MIN = 2     # wait at least 2 seconds between retries
RETRY_WAIT_MAX = 10    # wait at most 10 seconds (exponential backoff)

# ── RAG (Retrieval-Augmented Generation) configuration ───────────────
EMBED_MODEL      = "all-MiniLM-L6-v2"       # free local embedding model
COLLECTION_NAME  = "mediamind_transcripts"  # ChromaDB collection name
TOP_K            = 4     # number of chunks to retrieve per query
CHUNK_SIZE       = 400   # characters per chunk
CHUNK_OVERLAP    = 80    # overlap between chunks (prevents context cutoff)
BM25_WEIGHT      = 0.4   # weight for keyword search in hybrid retrieval
VECTOR_WEIGHT    = 0.6   # weight for semantic search in hybrid retrieval
