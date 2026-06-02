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

# ── RAGAS Evaluation configuration ───────────────────────────────────
# Toggle evaluation on/off without code changes.
# Set EVAL_ENABLED=false in .env to disable in resource-constrained envs.
EVAL_ENABLED = os.getenv("EVAL_ENABLED", "true").lower() == "true"

# Metric thresholds — scores below these trigger WARNING logs and
# Langfuse alerts. Tune based on your quality requirements.
#
#   Faithfulness    : < 0.70 → hallucination risk (LLM not using context)
#   Context Precision: < 0.60 → retrieval noise (chunks not relevant)
#   Answer Relevancy : < 0.65 → off-topic answers (router misfire?)
#   Context Recall   : < 0.60 → retrieval gaps (missing key info)
#   Hallucination    : > 0.30 → high fabrication risk
EVAL_THRESH_FAITHFULNESS  = float(os.getenv("EVAL_THRESH_FAITHFULNESS",  "0.70"))
EVAL_THRESH_PRECISION     = float(os.getenv("EVAL_THRESH_PRECISION",     "0.60"))
EVAL_THRESH_RELEVANCY     = float(os.getenv("EVAL_THRESH_RELEVANCY",     "0.65"))
EVAL_THRESH_RECALL        = float(os.getenv("EVAL_THRESH_RECALL",        "0.60"))
EVAL_THRESH_HALLUCINATION = float(os.getenv("EVAL_THRESH_HALLUCINATION", "0.30"))

# Langfuse evaluation toggle — set false to skip score posting even
# if evaluation runs (useful for local testing without Langfuse keys)
EVAL_LOG_TO_LANGFUSE = os.getenv("EVAL_LOG_TO_LANGFUSE", "true").lower() == "true"

# Dataset collection — set true to save every production Q/A/context
# to a JSONL file for offline regression testing
EVAL_COLLECT_DATASET = os.getenv("EVAL_COLLECT_DATASET", "false").lower() == "true"
EVAL_DATASET_PATH    = os.getenv("EVAL_DATASET_PATH", "eval_dataset.jsonl")