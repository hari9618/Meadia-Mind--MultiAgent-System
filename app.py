# app.py  (PRODUCTION UPGRADE — v3 with RAGAS evaluation)
# ─────────────────────────────────────────────────────────────────────
# Changes vs v2:
#   ✦ retrieve_context_with_chunks() replaces retrieve_context()
#     → raw chunks passed to run_agent() for RAGAS evaluation
#   ✦ eval_result shown in sidebar when EVAL_ENABLED=true
#   ✦ All existing logic preserved — UI behaviour unchanged
# ─────────────────────────────────────────────────────────────────────

import re
import uuid
import logging
import datetime
import streamlit as st
from dotenv import load_dotenv

# Updated import: retrieve_context_with_chunks instead of retrieve_context
from rag import store_transcript, retrieve_context_with_chunks
from agent import run_agent
from mcp_tools import TOOL_NAMES
from guardrails import run_input_pipeline, validate_transcript
from tracer import get_tracer
from config import EVAL_ENABLED

load_dotenv()
logging.basicConfig(level=logging.INFO)

tracer = get_tracer()

# ── Helpers ───────────────────────────────────────────────────────────
_YT = re.compile(r'((?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-?=&]+)')

def _yt_url(text):
    m = _YT.search(text)
    return m.group(1) if m else None

def _fetch_yt(url):
    tool = TOOL_NAMES.get("youtube_transcript")
    if not tool:
        raise RuntimeError("youtube_transcript tool not available.")
    t = tool.invoke(url)
    if t.startswith("YouTube transcript failed"):
        raise RuntimeError(t)
    vid = url.split("youtu.be/")[-1].split("v=")[-1].split("&")[0].split("?")[0]
    doc_id = f"yt_{vid}"
    vr = validate_transcript(t, doc_id)
    if not vr.passed:
        raise RuntimeError(f"Transcript validation failed: {vr.reason}")
    n = store_transcript(t, doc_id=doc_id)
    return t, n

def _uid():
    import random, string
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

def _ts():
    return datetime.datetime.now().strftime("%b %d, %H:%M")

def _short(q, n=40):
    q = q.strip()
    return q[:n] + "…" if len(q) > n else q

def _export(sess):
    lines = [f"# MediaMind — {sess['title']}", f"*{sess['created_at']}*",
             f"**Source:** {sess.get('source_label','—')}", "---"]
    for m in sess["history"]:
        if m["role"] == "user":
            lines += [f"\n**You** _{m.get('ts','')}_", m["content"]]
        else:
            c = m["content"]
            lines.append(f"\n**MediaMind · {m.get('task','agent')}** _{m.get('ts','')}_")
            if isinstance(c, list):
                for i in c:
                    lines.append(f"- {i.get('highlight','') if isinstance(i,dict) else i}")
            else:
                lines.append(str(c))
    return "\n".join(lines)

# ── Sample transcript ─────────────────────────────────────────────────
SAMPLE = """
Welcome to TechPulse Weekly. Our guest is Dr. Sarah Chen, Chief AI Officer at MediaFuture Inc.

Host: How is AI changing content creation?
Sarah: Two years ago, teams spent 60% of their time on repetitive tasks. Today AI handles all of that in real-time. What took 45 minutes now happens in under 90 seconds.

Host: How does AI factor into recommendations?
Sarah: We use collaborative filtering plus LLMs. Session duration went up 34% after rollout.

Host: Concerns about AI replacing journalists?
Sarah: AI eliminates mundane work, not creative work. We hired more editorial staff since deploying AI — content output tripled.

Host: Where do you see media AI in five years?
Sarah: Real-time, multilingual, personalized content at scale. We are 18 months from that being standard.
"""

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediaMind",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (unchanged) ───────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.mm-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 55%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1.15;
}
.mm-sub { font-size: 0.9rem; color: #64748b; margin-top: 4px; }
.chip-row { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin: 10px 0 4px; }
.chip { font-size: 0.7rem; padding: 3px 10px; border-radius: 99px;
        border: 1px solid rgba(148,163,184,0.25); color: #64748b; }
.src-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem;
            color: #94a3b8; background: rgba(99,102,241,0.08);
            border: 1px solid rgba(99,102,241,0.2); border-radius: 99px; padding: 4px 14px; }
.agent-tag { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
             text-transform: uppercase; color: #7c3aed; margin-bottom: 4px; }
.hl { border-left: 3px solid #7c3aed; padding: 6px 12px;
      border-radius: 0 8px 8px 0; background: rgba(124,58,237,0.05); margin-bottom: 6px; }
.hl b { font-size: 0.87rem; }
.hl small { font-size: 0.68rem; color: #94a3b8; display: block; margin-top: 2px; }
.bdg { display: inline-block; font-size: 0.62rem; font-weight: 700;
       padding: 1px 6px; border-radius: 99px; text-transform: uppercase;
       letter-spacing: 0.05em; margin-right: 3px; }
.bdg-statistic { background:rgba(59,130,246,.15); color:#60a5fa; }
.bdg-insight   { background:rgba(34,197,94,.15);  color:#4ade80; }
.bdg-quote     { background:rgba(249,115,22,.15); color:#fb923c; }
.bdg-high { background:rgba(239,68,68,.15);   color:#f87171; }
.bdg-mid  { background:rgba(245,158,11,.15);  color:#fbbf24; }
.bdg-low  { background:rgba(148,163,184,.15); color:#94a3b8; }
.ts { font-size: 0.63rem; color: #475569; margin-top: 3px; }
.empty { text-align: center; padding: 60px 20px; color: #475569; }
.empty .ei { font-size: 2.5rem; margin-bottom: 12px; }
.empty .et { font-size: 1rem; font-weight: 600; }
.empty .es { font-size: 0.82rem; color: #64748b; margin-top: 4px; }
[data-testid="stSidebar"] > div:first-child {
    background: #0d0d11; border-right: 1px solid rgba(255,255,255,0.07); }
.sb-brand { display: flex; align-items: center; gap: 10px;
            padding: 20px 16px 12px; border-bottom: 1px solid rgba(255,255,255,0.06);
            margin-bottom: 8px; }
.sb-brand-icon { font-size: 1.5rem; }
.sb-brand-text { font-size: 1.15rem; font-weight: 800;
                 background: linear-gradient(135deg, #a78bfa, #38bdf8);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sb-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: #334155; padding: 10px 16px 4px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────
def _new_session():
    sid = _uid()
    return {
        "id": sid,
        "title": "New chat",
        "created_at": _ts(),
        "history": [],
        "transcript_indexed": False,
        "transcript_source": "sample",
        "source_label": "TechPulse Weekly (sample)",
        "num_chunks": 0,
        "session_id": str(uuid.uuid4()),
    }

if "sessions" not in st.session_state:
    st.session_state.sessions = [_new_session()]
if "active_sid" not in st.session_state:
    st.session_state.active_sid = st.session_state.sessions[0]["id"]
if "show_src" not in st.session_state:
    st.session_state.show_src = False

def cur():
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.active_sid:
            return s
    return st.session_state.sessions[0]

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-icon">🎙️</span>
        <span class="sb-brand-text">MediaMind</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Sessions</div>', unsafe_allow_html=True)

    for sess in st.session_state.sessions:
        label = sess["title"]
        is_active = sess["id"] == st.session_state.active_sid
        if st.button(f"{'▶ ' if is_active else ''}{label}", key=f"sess_{sess['id']}",
                     use_container_width=True):
            st.session_state.active_sid = sess["id"]
            st.rerun()

    if st.button("＋ New chat", use_container_width=True, key="new_sess"):
        ns = _new_session()
        st.session_state.sessions.append(ns)
        st.session_state.active_sid = ns["id"]
        st.rerun()

    st.markdown("---")

    # ── Evaluation status in sidebar ──────────────────────────────────
    if EVAL_ENABLED:
        st.markdown('<div class="sb-label">🧪 RAGAS Evaluation</div>', unsafe_allow_html=True)
        st.markdown(
            "<small style='color:#64748b'>Auto-evaluation runs after each response. "
            "Scores appear in Langfuse dashboard within ~10s.</small>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<small style='color:#94a3b8'>Metrics: Faithfulness · Relevancy · "
            "Context Precision · Hallucination</small>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<small style='color:#475569'>Evaluation disabled (EVAL_ENABLED=false)</small>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Export ────────────────────────────────────────────────────────
    if cur()["history"]:
        md = _export(cur())
        st.download_button(
            "⬇ Export chat",
            md,
            file_name=f"mediamind_{cur()['id']}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ── Main content ──────────────────────────────────────────────────────
st.markdown('<div class="mm-title">MediaMind</div>', unsafe_allow_html=True)
st.markdown('<div class="mm-sub">Autonomous Media Intelligence · Hybrid RAG · Multi-Agent</div>',
            unsafe_allow_html=True)

# Source management (unchanged from v2)
with st.expander("📁 Manage content source", expanded=not cur()["transcript_indexed"]):
    tab_yt, tab_up, tab_sample = st.tabs(["YouTube URL", "Upload file", "Sample podcast"])

    with tab_yt:
        yt_inp = st.text_input("YouTube URL", key="yt_inp", placeholder="https://youtube.com/watch?v=...")
        if st.button("Load YouTube", key="load_yt") and yt_inp:
            with st.spinner("Fetching transcript…"):
                try:
                    txt, n = _fetch_yt(yt_inp)
                    vid_id = yt_inp.split("youtu.be/")[-1].split("v=")[-1][:11]
                    st.session_state.rag_doc = f"yt_{vid_id}"
                    st.session_state.active_transcript = txt
                    cur().update({"num_chunks": n, "transcript_indexed": True,
                                  "transcript_source": "youtube",
                                  "source_label": f"YouTube — {yt_inp[:42]}"})
                    st.session_state.show_src = False
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))

    with tab_up:
        uf = st.file_uploader("Upload transcript (.txt, .md, .srt)", type=["txt","md","srt"])
        if uf and st.button("Index file", key="idx_file"):
            with st.spinner("Indexing…"):
                text = uf.read().decode("utf-8", errors="ignore")
                doc_id = f"upload_{uf.name[:20].replace(' ','_')}"
                vr = validate_transcript(text, doc_id)
                if not vr.passed:
                    st.error(f"Validation failed: {vr.reason}")
                else:
                    n = store_transcript(text, doc_id=doc_id)
                    st.session_state.rag_doc = doc_id
                    st.session_state.active_transcript = text
                    cur().update({"num_chunks": n, "transcript_indexed": True,
                                  "transcript_source": "upload", "source_label": uf.name})
                    st.session_state.show_src = False
                    st.rerun()

    with tab_sample:
        if cur()["transcript_source"] != "sample":
            if st.button("Use sample", key="use_s"):
                with st.spinner("Loading…"):
                    n = store_transcript(SAMPLE, doc_id="techpulse_ep01")
                st.session_state.rag_doc = "techpulse_ep01"
                st.session_state.active_transcript = SAMPLE
                cur().update({"num_chunks": n, "transcript_indexed": True,
                              "transcript_source": "sample",
                              "source_label": "TechPulse Weekly (sample)"})
                st.session_state.show_src = False
                st.rerun()
        else:
            st.success("Sample podcast is already active.")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

q1, q2, q3 = st.columns(3)
quick = None
with q1:
    if st.button("📝  Summarize", use_container_width=True, key="qs"):
        quick = "Give me a structured summary of this podcast episode"
with q2:
    if st.button("⭐  Highlights", use_container_width=True, key="qh"):
        quick = "What are the most important highlights from this interview?"
with q3:
    if st.button("📱  Social posts", use_container_width=True, key="qp"):
        quick = "Create social media posts promoting this podcast content"

user_in = st.chat_input("Ask anything — or paste a YouTube URL…")
final_q = quick or user_in

# ── Process query ─────────────────────────────────────────────────────
if final_q:
    session_id = cur().get("session_id", str(uuid.uuid4()))

    ok, clean_query, error_msg = run_input_pipeline(final_q, session_id=session_id)

    if not ok:
        cur()["history"].append({
            "role": "assistant",
            "content": f"⚠️ {error_msg}",
            "task": "error",
            "tools": [],
            "ts": _ts(),
            "trace_id": None,
        })
        st.rerun()

    cur()["history"].append({"role": "user", "content": final_q, "ts": _ts()})
    if cur()["title"] == "New chat":
        cur()["title"] = _short(final_q)

    yt = _yt_url(clean_query)
    if yt:
        with st.spinner("Fetching YouTube transcript…"):
            try:
                txt, n = _fetch_yt(yt)
                short_yt = yt[:42] + ("…" if len(yt) > 42 else "")
                vid_id = yt.split("youtu.be/")[-1].split("v=")[-1][:11]
                st.session_state.rag_doc = f"yt_{vid_id}"
                st.session_state.active_transcript = txt
                cur().update({"num_chunks": n, "transcript_indexed": True,
                              "transcript_source": "youtube",
                              "source_label": f"YouTube — {short_yt}"})
            except RuntimeError as e:
                cur()["history"].append({"role":"assistant","content":f"❌ {e}",
                                         "task":"error","tools":[],"ts":_ts(),"trace_id":None})
                st.rerun()
        cq = clean_query.replace(yt,"").strip(" :.,-") or "Give me a structured summary of this video"
    else:
        cq = clean_query

    with st.spinner("Thinking…"):
        try:
            # ── KEY CHANGE: use retrieve_context_with_chunks ──────────
            # This returns BOTH the joined string (for LLM) AND the raw
            # chunk list (for RAGAS evaluation) in a single retrieval call.
            ctx, ctx_chunks = retrieve_context_with_chunks(cq)

            res = run_agent(
                query=cq,
                context=ctx,
                num_chunks=cur()["num_chunks"],
                session_id=session_id,
                context_chunks=ctx_chunks,   # NEW: passed to RAGAS eval
            )
            cur()["history"].append({
                "role":     "assistant",
                "content":  res["output"],
                "task":     res["task"],
                "tools":    res["tool_calls"],
                "ts":       _ts(),
                "trace_id": res.get("trace_id"),
            })
        except Exception as e:
            cur()["history"].append({
                "role":     "assistant",
                "content":  f"❌ Error: {e}\n\nCheck GROQ_API_KEY in .env",
                "task":     "error",
                "tools":    [],
                "ts":       _ts(),
                "trace_id": None,
            })
    st.rerun()

# ── Conversation render (unchanged from v2) ───────────────────────────
history = cur()["history"]

if not history:
    st.markdown("""
    <div class="empty">
        <div class="ei">💬</div>
        <div class="et">Ask me anything about the content</div>
        <div class="es">Use quick actions, type a question, or paste a YouTube URL</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for idx, msg in enumerate(history):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
                st.markdown(f'<div class="ts">{msg.get("ts","")}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                task = msg.get("task","")
                amap = {"summarize_agent":("📝","Summarize Agent"),
                        "highlight_agent":("⭐","Highlight Agent"),
                        "social_agent":("📱","Social Agent"),
                        "qa_agent":("💬","Q&A Agent"),
                        "error":("❌","Error")}
                ico2, nm = amap.get(task, ("🤖","MediaMind"))
                st.markdown(f'<div class="agent-tag">{ico2} {nm}</div>', unsafe_allow_html=True)

                content = msg["content"]
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            imp = item.get("importance","mid")
                            cat = item.get("category","")
                            dot = "🔴" if imp=="high" else "🟡"
                            bc2 = f'<span class="bdg bdg-{cat}">{cat}</span>' if cat else ""
                            bi  = f'<span class="bdg bdg-{imp}">{imp}</span>'
                            st.markdown(
                                f'<div class="hl"><b>{dot} {item.get("highlight","")}</b>'
                                f'<small>{bc2} {bi} · {item.get("timestamp_hint","")}</small></div>',
                                unsafe_allow_html=True)
                        else:
                            st.write(item)
                else:
                    st.markdown(content)

                st.markdown(f'<div class="ts">{msg.get("ts","")}</div>', unsafe_allow_html=True)

                # ── Feedback buttons ──────────────────────────────────
                trace_id = msg.get("trace_id")
                if trace_id and task not in ("error",):
                    fb_col1, fb_col2, _ = st.columns([1, 1, 8])
                    with fb_col1:
                        if st.button("👍", key=f"up_{idx}_{task}", help="Good response"):
                            tracer.score_response(trace_id, "user_feedback", 1.0, "thumbs_up")
                            st.toast("Thanks for the feedback!")
                    with fb_col2:
                        if st.button("👎", key=f"dn_{idx}_{task}", help="Bad response"):
                            tracer.score_response(trace_id, "user_feedback", 0.0, "thumbs_down")
                            st.toast("Feedback recorded.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _, cc, _ = st.columns([3,2,3])
    with cc:
        if st.button("🗑️  Clear this chat", use_container_width=True, key="clr"):
            cur()["history"] = []
            cur()["title"] = "New chat"
            st.rerun()