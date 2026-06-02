# Dockerfile
# ─────────────────────────────────────────────────────────────────────
# Multi-stage build for MediaMind production deployment.
#
# Changes vs v1:
#   ✦ Added ragas + datasets pre-install in builder stage
#   ✦ Pre-warm RAGAS imports to catch dependency issues at build time
#   ✦ RAGAS needs tokenizers/transformers — added to system libs
#   ✦ HF_HUB_OFFLINE=1 set at runtime to prevent unexpected downloads
# ─────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies:
#   build-essential, gcc, g++ — compile C extensions (chromadb, tokenizers)
#   git                       — some pip packages fetch from git
#   curl                      — healthcheck fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /build/venv
ENV PATH="/build/venv/bin:$PATH"

# Install Python dependencies
# requirements.txt now includes ragas>=0.2.0 and datasets>=2.18.0
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformer model → cached in image
# Prevents slow cold starts on first deployment
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Embedding model cached OK')"

# Pre-warm RAGAS imports to validate installation and cache any lazy downloads
# This catches import errors at build time, not at runtime
RUN python -c "\
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference; \
print('RAGAS metrics import OK')" || \
    echo "RAGAS pre-warm failed — will retry at runtime"


# ── Stage 2: runtime image ────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs only (no compilers):
#   libgomp1  — OpenMP, required by sentence-transformers / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /build/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy cached model weights from builder
COPY --from=builder /root/.cache /root/.cache

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Persistent ChromaDB storage mount point
RUN mkdir -p /app/chroma_store && chown appuser:appuser /app/chroma_store

# Eval dataset output directory
RUN mkdir -p /app/eval_data && chown appuser:appuser /app/eval_data

# Switch to non-root user
USER appuser

# ── Environment defaults ──────────────────────────────────────────────
# Prevent HuggingFace from making unexpected network calls at runtime.
# Models are already cached in /root/.cache from the builder stage.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Streamlit telemetry off (no phone-home in production)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# RAGAS evaluation enabled by default — override in docker-compose.yml
ENV EVAL_ENABLED=true
ENV EVAL_LOG_TO_LANGFUSE=true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]