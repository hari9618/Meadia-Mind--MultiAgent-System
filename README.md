# 🎙️ MediaMind — Autonomous Media Intelligence Platform

<p align="center">
  <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="700" />
</p>

<p align="center">
  ⚡ Multi-Agent AI that turns any podcast, video or transcript into summaries, highlights, social content & direct answers
</p>

---

## 🧩 Tech Badges

<p align="center">
  <img src="https://img.shields.io/badge/AI-Multi--Agent-blueviolet"/>
  <img src="https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-orange"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-red"/>
  <img src="https://img.shields.io/badge/LangGraph-Agent%20Orchestration-green"/>
  <img src="https://img.shields.io/badge/LangChain-Framework-yellow"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-darkgreen"/>
  <img src="https://img.shields.io/badge/BM25-Hybrid%20RAG-purple"/>
  <img src="https://img.shields.io/badge/RAGAS-RAG%20Evaluation-crimson"/>
  <img src="https://img.shields.io/badge/Langfuse-Observability-teal"/>
  <img src="https://img.shields.io/badge/Render-Deployed-brightgreen"/>
</p>

---

## 🚀 Live Demo

👉 **Try the App Here**

🔗 [https://mediamind-ai.onrender.com/](https://mediamind-ai.onrender.com/)

---

## 📌 Project Overview

**MediaMind** is a production-grade **Autonomous Media Intelligence Platform** powered by a multi-agent AI pipeline.

Instead of a single LLM call, it routes every user request through a **Supervisor → Specialist** agent system — intelligently deciding whether to summarize, extract highlights, generate social content, or answer a direct question.

It combines **Groq's ultra-fast inference** with **Hybrid RAG** (ChromaDB + BM25), **MCP-style tool calling**, **Langfuse v4 observability**, **RAGAS auto-evaluation**, and **real-time YouTube transcript ingestion** — all behind a clean, session-aware Streamlit chat UI.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| ⚡ **Ultra-Fast Inference** | Groq LPU running Llama 3.3 70B — sub-2s responses |
| 🧠 **Multi-Agent Pipeline** | Supervisor routes to Summarize / Highlight / Social / Q&A agent |
| 📺 **YouTube Ingestion** | Paste any YouTube URL — transcript fetched, indexed, answered |
| 🔍 **Hybrid RAG** | ChromaDB vector search (60%) + BM25 keyword search (40%) merged |
| 🔧 **MCP Tool Registry** | Wikipedia, DuckDuckGo, YouTube Transcript, File Reader — per-agent access control |
| 💬 **Multi-Session Chat** | Full session history, auto-titles, session switching, export to markdown |
| 💬 **Direct Q&A Mode** | Ask any question — Q&A Agent answers in 2–5 sentences, grounded in context |
| 🛡️ **Production Guardrails** | Input validation, content moderation, output sanitization (Pydantic v2) |
| 📊 **RAGAS Auto-Evaluation** | Faithfulness, Answer Relevancy, Context Precision + Hallucination detection |
| 🔭 **Langfuse Observability** | Every trace, span, LLM call and RAGAS score logged to Langfuse dashboard |
| 🚀 **Deployed on Render** | Persistent ChromaDB storage — data survives server restarts |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python 3.11+ | Core programming |
| ⚡ Groq API | Fast LLM inference (Llama 3.3 70B) |
| 🧠 LangGraph | Agent orchestration (StateGraph) |
| 🔗 LangChain | LLM integration + tool binding |
| 🎨 Streamlit | Frontend UI + multi-session chat |
| 📦 ChromaDB | Persistent vector store |
| 📊 BM25 (rank_bm25) | Keyword search for hybrid RAG |
| 🤗 all-MiniLM-L6-v2 | Local embeddings — zero API cost |
| 🌐 DuckDuckGo DDGS | Live web search tool |
| 📖 Wikipedia API | Factual enrichment tool |
| 📊 RAGAS 0.4.x | Production RAG evaluation (reference-free) |
| 🔭 Langfuse v4 | LLM observability + score tracking |
| 🚀 Render | Cloud deployment |
| 🐳 Docker | Containerised production build |

---

## 🏗️ System Architecture

```
MediaMind
│
├── app.py              # Streamlit UI — multi-session chat, source management
├── agent.py            # LangGraph multi-agent pipeline + evaluation trigger
├── rag.py              # Hybrid RAG (ChromaDB + BM25) — retrieve_context_with_chunks()
├── evaluation.py       # RAGAS evaluation service (fire-and-forget, non-blocking)
├── mcp_tools.py        # MCP tool registry (4 tools, per-agent access control)
├── llm.py              # Groq LLM client (3 temperature modes)
├── prompts.py          # All LLM prompts — ChatPromptTemplate + Few-Shot + CoT
├── guardrails.py       # Input validation, content moderation, output sanitization
├── tracer.py           # Langfuse v4 observability wrapper
├── schemas.py          # Pydantic v2 schemas for I/O validation
├── config.py           # Central config — models, RAG params, eval thresholds
├── tracer.py           # Langfuse v4 SDK wrapper (create_score, spans, traces)
├── Dockerfile          # Multi-stage production build
├── docker-compose.yml  # Local dev + persistent volumes
├── requirements.txt
├── .env                # API keys (NOT pushed to GitHub)
└── README.md
```

---

## 🤖 Agent Pipeline

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                    Supervisor Node                    │
│          Reads query, decides routing (temp=0.0)      │
└────┬──────────┬──────────────┬────────────────────────┘
     │          │              │              │
     ▼          ▼              ▼              ▼
┌─────────┐ ┌─────────┐ ┌─────────┐  ┌────────────┐
│Summarize│ │Highlight│ │ Social  │  │  Q&A Agent │
│  Agent  │ │  Agent  │ │  Agent  │  │            │
└────┬────┘ └────┬────┘ └────┬────┘  └─────┬──────┘
     │           │           │             │
     ▼           ▼           ▼             ▼
 Wikipedia   Wikipedia   Web Search    Wikipedia
 Web Search  Web Search    (only)      Web Search
     │           │           │             │
     ▼           ▼           ▼             ▼
  Groq 0.3   Groq 0.0    Groq 0.75    Groq 0.3
 (balanced) (precise)   (creative)   (balanced)
     │           │           │             │
     └───────────┴───────────┴─────────────┘
                         │
                         ▼
              Output Guardrails
              (validation + sanitization)
                         │
                         ▼
                 Final Response → UI
                         │
                         ▼ (background daemon thread)
              RAGAS Evaluation (non-blocking)
                         │
                         ▼
              Langfuse Scores Dashboard
```

### 🤖 Agent Routing Logic

| Query type | Example | Routes to |
|---|---|---|
| Wants a summary / overview | "summarize this video" | `summarize_agent` |
| Wants highlights / key moments | "what are the key points?" | `highlight_agent` |
| Wants social media content | "write a LinkedIn post" | `social_agent` |
| Asks a direct question | "what does X mean?" / "who is Y?" | `qa_agent` |

> **Routing rule:** If the query contains question words — *what, why, how, who, when, explain, define, tell me* — the supervisor always routes to `qa_agent`. The Q&A Agent answers in 2–5 sentences grounded in transcript content, no structured reports.

---

## 🔍 Hybrid RAG Pipeline

```
User Query
    │
    ├──────────────────────────────┐
    ▼                              ▼
ChromaDB Vector Search         BM25 Keyword Search
(semantic similarity)          (exact term matching)
all-MiniLM-L6-v2 embeddings    rank_bm25 BM25Okapi
Top-4 chunks (60% weight)      Top-4 chunks (40% weight)
    │                              │
    └──────────┬───────────────────┘
               ▼
       Merge + Deduplicate
    (vector results get priority)
               │
               ▼
  retrieve_context_with_chunks()
  returns: (joined_str, List[str])
       │              │
       ▼              ▼
   LLM Prompt    RAGAS Evaluation
```

> `retrieve_context_with_chunks()` returns both the joined context string (for the LLM prompt) and the raw chunk list (for RAGAS evaluation) in a single retrieval call — zero double-fetching.

---

## 📊 RAGAS Evaluation Pipeline

MediaMind automatically evaluates every production query using **RAGAS 0.4.x** — no ground truth or human labels required.

```
run_agent() completes
       │
       ▼ (fire-and-forget daemon thread — user never waits)
EvaluationService.evaluate_fire_and_forget()
       │
       ▼
_run_ragas_sync()
  ├── Faithfulness                          (is answer grounded in context?)
  ├── Answer Relevancy                      (does answer address the question?)
  └── LLMContextPrecisionWithoutReference   (is retrieval pulling clean chunks?)
       │
       ▼
_check_thresholds()  →  WARNING logs + UI alerts if below threshold
       │
       ▼
tracer.score_response()  →  client.create_score()  →  Langfuse dashboard
```

### Evaluation Metrics

| Metric | What it measures | Threshold |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? Low = hallucination risk | `>= 0.70` |
| **Answer Relevancy** | Does the answer actually address the query? | `>= 0.65` |
| **Context Precision** | Are retrieved chunks mostly relevant? Low = noisy retrieval | `>= 0.60` |
| **Hallucination Score** | Derived: `1 - faithfulness`. High = fabrication risk | `<= 0.30` |

> All metrics are **reference-free** — no ground truth needed. Scores appear in Langfuse within ~10–20s of each response.

---

## 🔭 Langfuse Observability

Every pipeline run is fully traced in **Langfuse v4**:

| What is traced | Langfuse object |
|---|---|
| Full pipeline run | Root span (`mediamind_pipeline`) |
| Supervisor routing decision | Child span |
| RAG retrieval | Child span (chunk count + latency) |
| Every LLM call (LangGraph) | Auto-traced via `CallbackHandler` |
| Guardrail checks | Child span (pass/fail + reason) |
| RAGAS scores | `create_score()` per metric |
| User thumbs up / down | `create_score("user_feedback")` |

> **Langfuse v4 note:** Uses `get_client()` + `start_as_current_observation()` + `create_score()`. The old `.score()` method silently did nothing — fixed to `create_score()` with `data_type="NUMERIC"`.

---

## 🛡️ Production Guardrails

Three-layer protection on every request:

```
Input Query
    │
    ▼
1. Pydantic QueryInput validation
   (min/max length, whitespace, injection patterns)
    │
    ▼
2. Content moderation
   (off-topic, harmful, ambiguous — rule-based, no API cost)
    │
    ▼
3. Output guardrails
   (HighlightOutput schema validation, prompt-leak sanitization)
```

---

## 🔧 MCP Tool Registry

| Tool | Description | Agent Access |
|---|---|---|
| `youtube_transcript` | Fetches full transcript from YouTube URL | Research agent |
| `web_search` | Live DuckDuckGo search for news & trends | All agents |
| `wikipedia_search` | Factual background on people & topics | Summarize, Highlight, Q&A |
| `read_file` | Reads local .txt / .srt / .md transcript | Research agent |

> Each agent gets **only the tools it needs** — deliberate access control, not default behaviour.

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/hari9618/mediamind
cd mediamind
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

Create a `.env` file (see `.env.example`):

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Langfuse observability (v4 SDK)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # or https://cloud.langfuse.com for EU

# RAGAS evaluation
EVAL_ENABLED=true
EVAL_LOG_TO_LANGFUSE=true
EVAL_THRESH_FAITHFULNESS=0.70
EVAL_THRESH_PRECISION=0.60
EVAL_THRESH_RELEVANCY=0.65
EVAL_THRESH_HALLUCINATION=0.30
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)  
Get your Langfuse keys at [cloud.langfuse.com](https://cloud.langfuse.com)

### 5️⃣ Run the App

```bash
streamlit run app.py
```

### 🐳 Run with Docker

```bash
docker compose up --build    # first run
docker compose up            # subsequent runs
docker compose down -v       # stop + wipe volumes
```

---

## 🧠 How It Works

```
1️⃣  User sends a query (or pastes a YouTube URL)
        │
2️⃣  Input guardrails → Pydantic validation + content moderation
        │
3️⃣  YouTube URL detected? → Fetch transcript → Clear ChromaDB → Re-index
        │
4️⃣  retrieve_context_with_chunks() → ChromaDB semantic + BM25 keyword → Top-4 chunks
        │                              returns both joined string AND raw list
        │
5️⃣  Supervisor reads query → Routes to Summarize / Highlight / Social / Q&A agent
        │        (question words detected? → qa_agent for direct concise answer)
        │
6️⃣  Agent calls MCP tools (Wikipedia, DuckDuckGo) via ReAct reasoning
        │
7️⃣  Agent formats prompt → Groq LLM → Output guardrails → Response to UI
        │
8️⃣  Background: RAGAS evaluates faithfulness + relevancy + precision
        │         Scores posted to Langfuse via create_score() (~10–20s delay)
        │
9️⃣  Response rendered in chat — markdown or styled highlight cards
       User can submit 👍/👎 feedback → score_response("user_feedback") → Langfuse
```

---

## 📷 Application Preview

<img width="951" height="446" alt="MediaMind Screenshot" src="https://github.com/user-attachments/assets/978fbee0-d71f-4b39-9519-98e0de61ecab" />

---

## 📚 What I Learned

✔ **LangGraph StateGraph** — building real state machines with typed state and conditional edges  
✔ **Hybrid RAG Engineering** — combining vector + keyword search with weighted merging  
✔ **MCP Tool Architecture** — per-agent access control, tool binding, ToolMessage conversations  
✔ **Multi-Session State Management** — Streamlit session_state design for complex apps  
✔ **Production RAG Deployment** — PersistentClient ChromaDB, real-time re-indexing  
✔ **LLM Temperature Strategy** — precise / balanced / creative modes for different task types  
✔ **YouTube API Integration** — youtube-transcript-api v1.x, URL parsing, live ingestion  
✔ **Intelligent Task Routing** — keyword-based intent detection to separate Q&A from generation tasks  
✔ **RAGAS Evaluation** — reference-free production RAG evaluation with Faithfulness, Relevancy, Context Precision  
✔ **Langfuse v4 Observability** — tracing, spans, scores, `create_score()` API, US vs EU region config  
✔ **Production Guardrails** — Pydantic v2 input/output validation, prompt injection detection, content moderation  
✔ **SDK Version Debugging** — diagnosed breaking changes across RAGAS 0.1→0.4 and Langfuse v2→v4 APIs  

---

## 🎯 Future Improvements

🔹 Cross-encoder reranking — add `cross-encoder/ms-marco-MiniLM-L-6-v2` between retrieval and TOP-K slice  
🔹 Semantic chunking — replace character splitter with `SemanticChunker` for boundary-aware chunks  
🔹 Offline CI evaluation — RAGAS batch evaluation against collected JSONL dataset in GitHub Actions  
🔹 Speaker diarization — identify who said what in transcripts  
🔹 Multi-turn Q&A — follow-up questions that remember previous answers in session  
🔹 Multi-document RAG — index multiple videos/files simultaneously  
🔹 Audio file support — direct .mp3/.wav upload with Whisper transcription  
🔹 Synthetic dataset generation — RAGAS testset generator for automated benchmark creation  

---

## 👨‍💻 Author

**Hari Krishna T**  
AI Engineer | Multi-Agent Systems Builder | Gen AI Developer

🔗 GitHub: [github.com/hari9618](https://github.com/hari9618)  
🔗 LinkedIn: [linkedin.com/in/hari-krishna-thota-06b850275](https://linkedin.com/in/hari-krishna-thota-06b850275)

---

## ⭐ Support

If you like this project:

⭐ **Star the repository**  
📢 **Share with others**  
🍴 **Fork and build on top of it**

---

## 📢 Tags

`AI` `Multi-Agent` `LangGraph` `LangChain` `Groq` `RAG` `ChromaDB` `BM25` `Streamlit` `YouTube` `MCP` `RAGAS` `Langfuse` `Evaluation` `Observability` `Guardrails` `Python` `Generative AI` `LLM` `Render` `Docker`
