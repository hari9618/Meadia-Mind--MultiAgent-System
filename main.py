# main.py
# ─────────────────────────────────────────────────────────────────────
# CLI entry point — run the agent pipeline without Streamlit.
# Use this to test everything works before running the full UI.
#
# Usage:
#   python main.py
# ─────────────────────────────────────────────────────────────────────

import json
import logging
from dotenv import load_dotenv

from rag import store_transcript, retrieve_context
from agent import run_agent

load_dotenv()  # load GROQ_API_KEY from .env

# Configure logging so we can see what each module is doing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ── Sample transcript ─────────────────────────────────────────────────
# This is the same sample transcript from your original project.
# Replace this with real podcast content or a YouTube URL later.

SAMPLE_TRANSCRIPT = """
Welcome to TechPulse Weekly, your go-to podcast for everything in technology and innovation.

Today we're diving deep into the world of artificial intelligence and its transformative impact
on the media industry. Our guest is Dr. Sarah Chen, Chief AI Officer at MediaFuture Inc.

Host: Sarah, it's wonderful to have you. Let's start — how is AI changing content creation?

Sarah: The change is dramatic. Two years ago, content teams spent 60% of their time on
repetitive tasks — transcription, tagging, clipping. Today, AI handles all of that in
real-time. Our editors focus entirely on creative decisions now.

Host: Can you give a concrete example?

Sarah: We run an AI pipeline that watches every live broadcast. The moment a newsworthy
segment ends, the system automatically generates a 60-second clip, writes three social
captions, and pushes them to our queue. What took 45 minutes now happens in under 90 seconds.

Host: What about accuracy?

Sarah: We run two-stage verification. AI does the first pass, then a human editor reviews
in about 30 seconds. Error rate dropped from 8% with manual workflows to under 1.2%.

Host: How does AI factor into recommendation systems?

Sarah: We use collaborative filtering plus large language models. The LLM reads every
transcript and generates a semantic fingerprint — themes, tone, topics. We match those
against viewer engagement history. Session duration went up 34% after rollout.

Host: Any concerns about AI replacing journalists?

Sarah: AI is eliminating mundane work, not creative work. We've actually hired more editorial
staff since deploying AI — content output tripled and we needed more voices, not fewer.

Host: Where do you see media AI in five years?

Sarah: Real-time, multilingual, personalized content at scale. A broadcast that adapts its
depth based on who's watching — three minutes for casual viewers, twenty minutes for
enthusiasts — all from the same source feed. We're 18 months from that being standard.

Host: Dr. Sarah Chen, thank you so much.

Sarah: My pleasure. Exciting times ahead.
"""


def run(query: str):
    """Run the full pipeline for a single query and print results."""
    print("\n" + "═" * 65)
    print(f"  QUERY: {query}")
    print("═" * 65)

    print("\n[1] Ingesting transcript into ChromaDB + BM25...")
    num_chunks = store_transcript(SAMPLE_TRANSCRIPT, doc_id="techpulse_ep01")
    print(f"    ✓ {num_chunks} chunks indexed")

    print("\n[2] Retrieving relevant context via hybrid RAG...")
    context = retrieve_context(query)
    print(f"    ✓ Retrieved top chunks")

    print("\n[3] Running multi-agent pipeline (supervisor → specialist)...")
    result = run_agent(query, context, num_chunks)

    print(f"\n    Agent used   : {result['task']}")
    print(f"    Tools called : {[t['tool'] for t in result['tool_calls']]}")

    print("\n" + "─" * 65)
    print(f"  OUTPUT — {result['task'].upper()}")
    print("─" * 65 + "\n")

    if isinstance(result["output"], list):
        print(json.dumps(result["output"], indent=2))
    else:
        print(result["output"])

    print("\n" + "═" * 65)


if __name__ == "__main__":
    # Test all three task types
    queries = [
        "Give me a structured summary of this podcast episode",
        "What are the most important highlights from this interview?",
        "Create social media posts promoting this podcast content",
    ]
    for q in queries:
        run(q)
        print()
