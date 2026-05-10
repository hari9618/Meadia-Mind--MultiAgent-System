# 🎙️ MediaMind — Autonomous Media Intelligence Platform

<p align="center">
  <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="700" />
</p>

<p align="center">
  ⚡ Multi-Agent AI that turns any podcast, video or transcript into summaries, highlights & social content
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
  <img src="https://img.shields.io/badge/Render-Deployed-brightgreen"/>
  <img src="https://img.shields.io/badge/QA-Direct%20Q%26A%20Agent-blue"/>
</p>

---

## 🚀 Live Demo

👉 **Try the App Here**

🔗 **Frontend (Streamlit):**
[https://mediamind-ai.onrender.com/](https://mediamind-ai.onrender.com/)

---

## 📌 Project Overview

**MediaMind** is a production-grade **Autonomous Media Intelligence Platform** powered by a multi-agent AI pipeline.

Instead of a single LLM call, it routes every user request through a **Supervisor → Specialist** agent system — intelligently deciding whether to summarize, extract highlights, or generate social content.

It combines **Groq's ultra-fast inference** with **Hybrid RAG** (ChromaDB + BM25), **MCP-style tool calling**, and **real-time YouTube transcript ingestion** — all behind a clean, session-aware Streamlit chat UI.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| ⚡ **Ultra-Fast Inference** | Groq LPU running Llama 3.3 70B — sub-2s responses |
| 🧠 **Multi-Agent Pipeline** | Supervisor routes to Summarize / Highlight / Social agent |
| 📺 **YouTube Ingestion** | Paste any YouTube URL — transcript fetched, indexed, answered |
| 🔍 **Hybrid RAG** | ChromaDB vector search (60%) + BM25 keyword search (40%) merged |
| 🔧 **MCP Tool Registry** | Wikipedia, DuckDuckGo, YouTube Transcript, File Reader — per-agent access control |
| 💬 **Multi-Session Chat** | Full session history, auto-titles, session switching, export to markdown |
| 💬 **Direct Q&A Mode** | Ask any question — Q&A Agent answers concisely, no structured reports |
| 🚀 **Deployed on Render** | Persistent ChromaDB storage — data survives server restarts |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming |
| ⚡ Groq API | Fast LLM inference (Llama 3.3 70B) |
| 🧠 LangGraph | Agent orchestration (StateGraph) |
| 🔗 LangChain | LLM integration + tool binding |
| 🎨 Streamlit | Frontend UI + multi-session chat |
| 📦 ChromaDB | Vector store (persistent) |
| 📊 BM25 (rank_bm25) | Keyword search for hybrid RAG |
| 🤗 all-MiniLM-L6-v2 | Local embeddings — zero API cost |
| 🌐 DuckDuckGo DDGS | Live web search tool |
| 📖 Wikipedia API | Factual enrichment tool |
| 🚀 Render | Cloud deployment |

---

## 🏗️ System Architecture

```
MediaMind
│
├── app.py              # Streamlit UI — multi-session chat, source management
├── agent.py            # LangGraph multi-agent pipeline
├── rag.py              # Hybrid RAG (ChromaDB + BM25)
├── mcp_tools.py        # MCP tool registry (4 tools, per-agent access control)
├── llm.py              # Groq LLM client (3 temperature modes)
├── prompts.py          # All LLM prompts — clean separation of concerns
├── config.py           # Central config — models, RAG params, retry settings
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
┌─────────────────────────────────────────────────────┐
│                   Supervisor Node                    │
│          Reads query, decides routing (temp=0.0)     │
└──────┬──────────┬──────────────┬────────────────────┘
       │          │              │                │
       ▼          ▼              ▼                ▼
┌──────────┐ ┌──────────┐ ┌──────────┐  ┌─────────────┐
│Summarize │ │Highlight │ │  Social  │  │   Q&A Agent │
│  Agent   │ │  Agent   │ │  Agent   │  │  (NEW ✨)   │
└────┬─────┘ └────┬─────┘ └────┬─────┘  └──────┬──────┘
     │            │            │               │
     ▼            ▼            ▼               ▼
 Wikipedia    Wikipedia    Web Search      Wikipedia
 Web Search   Web Search    (only)        Web Search
     │            │            │               │
     ▼            ▼            ▼               ▼
  Groq 0.3    Groq 0.0     Groq 0.75       Groq 0.3
  (balanced)  (precise)   (creative)      (balanced)
     │            │            │               │
     └────────────┴────────────┴───────────────┘
                              │
                              ▼
                    Final Response → Chat UI
```

### 🤖 Agent Routing Logic

| Query type | Example | Routes to |
|---|---|---|
| Wants a summary / overview | "summarize this video" | `summarize_agent` |
| Wants highlights / key moments | "what are the key points?" | `highlight_agent` |
| Wants social media content | "write a LinkedIn post" | `social_agent` |
| Asks a direct question | "what does X mean?" / "who is Y?" | `qa_agent` ✨ |

> **How the supervisor decides:** If the query contains question words — *what, why, how, who, when, explain, define* — it always routes to `qa_agent`. The Q&A Agent answers in 2–5 sentences, grounded in the transcript, with no structured reports or bullet points.

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
       Top-4 chunks → context string → Agent
```

---

## 🔧 MCP Tool Registry

| Tool | Description | Agent Access |
|---|---|---|
| `youtube_transcript` | Fetches full transcript from YouTube URL | Research agent |
| `web_search` | Live DuckDuckGo search for news & trends | All agents |
| `wikipedia_search` | Factual background on people & topics | Summarize, Highlight |
| `read_file` | Reads local .txt / .srt / .md transcript | Research agent |

> Each agent gets **only the tools it needs** — social agent gets web search only, summarize and highlight agents get Wikipedia + web search. This is deliberate architecture, not default behaviour.

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

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

```
1️⃣  User sends a query (or pastes a YouTube URL)
        │
2️⃣  YouTube URL detected? → Fetch transcript → Clear ChromaDB → Re-index
        │
3️⃣  Hybrid RAG retrieval → ChromaDB semantic + BM25 keyword → Top-4 chunks
        │
4️⃣  Supervisor reads query → Routes to Summarize / Highlight / Social / Q&A agent
        │        (question words detected? → qa_agent for direct concise answer)
        │
5️⃣  Agent calls MCP tools (Wikipedia, DuckDuckGo) for real-world enrichment
        │
6️⃣  Agent formats prompt: RAG context + tool results + user query → Groq LLM
        │
7️⃣  Response rendered in chat — markdown or styled highlight cards
```

---

## 📷 Application Preview

> *(Add your screenshot here)*

```
<img width="951" height="446" alt="Screenshot 2026-05-09 170729" src="https://github.com/user-attachments/assets/978fbee0-d71f-4b39-9519-98e0de61ecab" />

```

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

---

## 🎯 Future Improvements

🔹 Speaker diarization — identify who said what in transcripts  
🔹 Multi-turn Q&A — follow-up questions that remember previous answers in session  
🔹 Multi-document RAG — index multiple videos/files simultaneously  
🔹 Audio file support — direct .mp3/.wav upload with Whisper transcription  
🔹 Scheduled indexing — auto-index new episodes from RSS feeds  
🔹 Shareable sessions — export and share full conversation threads  

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

`AI` `Multi-Agent` `LangGraph` `LangChain` `Groq` `RAG` `ChromaDB` `BM25` `Streamlit` `YouTube` `MCP` `Python` `Generative AI` `LLM` `Render`
