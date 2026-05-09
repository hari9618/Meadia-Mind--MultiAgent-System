# app.py — MediaMind (clean rebuild)
# Run: streamlit run app.py

import re
import logging
import datetime
import streamlit as st
from dotenv import load_dotenv

from rag import store_transcript, retrieve_context
from agent import run_agent
from mcp_tools import TOOL_NAMES

load_dotenv()
logging.basicConfig(level=logging.INFO)

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
    n = store_transcript(t, doc_id=f"yt_{vid}")
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

# ── Minimal CSS (only what Streamlit won't do natively) ───────────────
st.markdown("""
<style>
/* Hide default chrome */
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Gradient title */
.mm-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 55%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.15;
}
.mm-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 4px;
}
.chip-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin: 10px 0 4px;
}
.chip {
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 99px;
    border: 1px solid rgba(148,163,184,0.25);
    color: #64748b;
}
.src-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: #94a3b8;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 99px;
    padding: 4px 14px;
}
.agent-tag {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 4px;
}
.hl {
    border-left: 3px solid #7c3aed;
    padding: 6px 12px;
    border-radius: 0 8px 8px 0;
    background: rgba(124,58,237,0.05);
    margin-bottom: 6px;
}
.hl b { font-size: 0.87rem; }
.hl small { font-size: 0.68rem; color: #94a3b8; display: block; margin-top: 2px; }
.bdg {
    display: inline-block; font-size: 0.62rem; font-weight: 700;
    padding: 1px 6px; border-radius: 99px; text-transform: uppercase;
    letter-spacing: 0.05em; margin-right: 3px;
}
.bdg-statistic { background:rgba(59,130,246,.15); color:#60a5fa; }
.bdg-insight   { background:rgba(34,197,94,.15);  color:#4ade80; }
.bdg-quote     { background:rgba(249,115,22,.15); color:#fb923c; }
.bdg-high { background:rgba(239,68,68,.15);   color:#f87171; }
.bdg-mid  { background:rgba(245,158,11,.15);  color:#fbbf24; }
.bdg-low  { background:rgba(148,163,184,.15); color:#94a3b8; }
.ts { font-size: 0.63rem; color: #475569; margin-top: 3px; }
.empty {
    text-align: center; padding: 60px 20px; color: #475569;
}
.empty .ei { font-size: 2.5rem; margin-bottom: 12px; }
.empty .et { font-size: 1rem; font-weight: 600; }
.empty .es { font-size: 0.82rem; color: #64748b; margin-top: 4px; }

/* Sidebar */
[data-testid="stSidebar"] > div:first-child {
    background: #0d0d11;
    border-right: 1px solid rgba(255,255,255,0.07);
}
.sb-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 20px 16px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 8px;
}
.sb-brand-icon { font-size: 1.5rem; }
.sb-brand-text {
    font-size: 1.15rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-label {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #334155; padding: 10px 16px 4px;
}
.sess-btn {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-radius: 8px; margin: 2px 8px;
    cursor: pointer; transition: background .15s;
}
.sess-active { background: rgba(99,102,241,0.15) !important; }
.sess-t { font-size: 0.8rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 160px; }
.sess-m { font-size: 0.65rem; color: #475569; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
if "sessions" not in st.session_state:
    fid = _uid()
    st.session_state.sessions = {
        fid: {
            "title": "New chat",
            "history": [],
            "created_at": _ts(),
            "source_label": "TechPulse Weekly (sample)",
            "transcript_source": "sample",
            "num_chunks": 0,
            "transcript_indexed": False,
        }
    }
    st.session_state.active_id = fid

if "active_id" not in st.session_state:
    st.session_state.active_id = list(st.session_state.sessions.keys())[0]

if "active_transcript" not in st.session_state:
    st.session_state.active_transcript = SAMPLE

if "rag_doc" not in st.session_state:
    st.session_state.rag_doc = None

if "show_src" not in st.session_state:
    st.session_state.show_src = False

def cur():
    return st.session_state.sessions[st.session_state.active_id]

# Auto-index sample ONLY on very first load (rag_doc is None)
# Never run this if user has uploaded a file or indexed a YouTube video
if st.session_state.rag_doc is None:
    with st.spinner("Loading…"):
        n = store_transcript(SAMPLE, doc_id="techpulse_ep01")
    st.session_state.rag_doc = "techpulse_ep01"
    cur()["num_chunks"] = n
    cur()["transcript_indexed"] = True

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-icon">🎙️</span>
        <span class="sb-brand-text">MediaMind</span>
    </div>
    """, unsafe_allow_html=True)

    # New chat
    if st.button("＋  New chat", use_container_width=True, type="primary"):
        nid = _uid()
        st.session_state.sessions[nid] = {
            "title": "New chat",
            "history": [],
            "created_at": _ts(),
            "source_label": cur()["source_label"],
            "transcript_source": cur()["transcript_source"],
            "num_chunks": cur()["num_chunks"],
            "transcript_indexed": cur()["transcript_indexed"],
        }
        st.session_state.active_id = nid
        st.session_state.show_src = False
        st.rerun()

    st.markdown('<div class="sb-label">Chat history</div>', unsafe_allow_html=True)

    # Session list
    for sid in reversed(list(st.session_state.sessions.keys())):
        sess = st.session_state.sessions[sid]
        is_active = (sid == st.session_state.active_id)
        ico = "📺" if sess["transcript_source"] == "youtube" else ("📄" if sess["transcript_source"] == "upload" else "🎙️")
        n_msg = len([m for m in sess["history"] if m["role"] == "user"])

        c1, c2 = st.columns([5, 1])
        with c1:
            label = f"{ico}  {sess['title']}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"s_{sid}", use_container_width=True,
                         help=f"{sess['created_at']} · {n_msg} message{'s' if n_msg!=1 else ''}",
                         type=btn_type):
                st.session_state.active_id = sid
                st.session_state.show_src = False
                st.rerun()
        with c2:
            if len(st.session_state.sessions) > 1:
                if st.button("✕", key=f"d_{sid}", help="Delete"):
                    del st.session_state.sessions[sid]
                    if st.session_state.active_id == sid:
                        st.session_state.active_id = list(st.session_state.sessions.keys())[-1]
                    st.rerun()

    st.divider()

    # Export
    st.download_button(
        "⬇  Export chat",
        data=_export(cur()),
        file_name=f"mediamind_{st.session_state.active_id}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()
    st.markdown('<div class="sb-label">Powered by</div>', unsafe_allow_html=True)
    for c in ["LangGraph multi-agent", "Groq Llama 3.3 70B", "ChromaDB + BM25 RAG",
              "MCP tool calling", "YouTube ingestion"]:
        st.caption(f"✦  {c}")

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
# Centre column
_, main, _ = st.columns([0.5, 9, 0.5])

with main:

    # ── Hero ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 24px 0 8px;">
        <div style="font-size:3rem; margin-bottom:6px;">🎙️</div>
        <p class="mm-title">MediaMind</p>
        <p class="mm-sub">Multi-agent AI for podcasts, videos &amp; media transcripts</p>
    </div>
    <div class="chip-row">
        <span class="chip">✦ Summarize</span>
        <span class="chip">✦ Highlights</span>
        <span class="chip">✦ Social posts</span>
        <span class="chip">✦ YouTube</span>
        <span class="chip">✦ Q&amp;A</span>
        <span class="chip">✦ Multi-agent RAG</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Source badge + change source ──────────────────────────────────
    sess = cur()
    sic = "📺" if sess["transcript_source"]=="youtube" else ("📄" if sess["transcript_source"]=="upload" else "🎙️")

    ba, bb, bc = st.columns([2, 1, 2])
    with ba:
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; align-items:center; height:38px;">
            <span class="src-pill">{sic}&nbsp; Active: <strong>&nbsp;{sess['source_label']}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    with bb:
        lbl = "▲ Close" if st.session_state.show_src else "⊕ Change source"
        if st.button(lbl, use_container_width=True, key="src_toggle"):
            st.session_state.show_src = not st.session_state.show_src
            st.rerun()

    # ── Source panel ──────────────────────────────────────────────────
    if st.session_state.show_src:
        with st.container(border=True):
            opt = st.radio("", ["Sample podcast (TechPulse Weekly)", "Upload .txt transcript"],
                           horizontal=True, label_visibility="collapsed", key="src_radio")
            if opt == "Upload .txt transcript":
                uf = st.file_uploader("Transcript file", type=["txt"],
                                      label_visibility="collapsed", key="uf")
                if uf:
                    text = uf.read().decode("utf-8")
                    with st.spinner("Indexing…"):
                        n = store_transcript(text, doc_id=f"upload_{uf.name}")
                    st.session_state.rag_doc = f"upload_{uf.name}"
                    st.session_state.active_transcript = text
                    cur().update({"num_chunks": n, "transcript_indexed": True,
                                  "transcript_source": "upload", "source_label": uf.name})
                    st.session_state.show_src = False
                    st.rerun()
            else:
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

    # ── Quick actions ─────────────────────────────────────────────────
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

    # ── Chat input ────────────────────────────────────────────────────
    user_in = st.chat_input("Ask anything — or paste a YouTube URL…")
    final_q = quick or user_in

    # ── Process ───────────────────────────────────────────────────────
    if final_q:
        cur()["history"].append({"role": "user", "content": final_q, "ts": _ts()})
        if cur()["title"] == "New chat":
            cur()["title"] = _short(final_q)

        yt = _yt_url(final_q)
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
                                             "task":"error","tools":[],"ts":_ts()})
                    st.rerun()
            cq = final_q.replace(yt,"").strip(" :.,-") or "Give me a structured summary of this video"
        else:
            cq = final_q

        with st.spinner("Thinking…"):
            try:
                ctx = retrieve_context(cq)
                res = run_agent(query=cq, context=ctx, num_chunks=cur()["num_chunks"])
                cur()["history"].append({"role":"assistant","content":res["output"],
                                         "task":res["task"],"tools":res["tool_calls"],
                                         "ts":_ts()})
            except Exception as e:
                cur()["history"].append({"role":"assistant",
                                         "content":f"❌ Error: {e}\n\nCheck GROQ_API_KEY in .env",
                                         "task":"error","tools":[],"ts":_ts()})
        st.rerun()

    # ── Conversation ──────────────────────────────────────────────────
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
        for msg in history:
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

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _, cc, _ = st.columns([3,2,3])
        with cc:
            if st.button("🗑️  Clear this chat", use_container_width=True, key="clr"):
                cur()["history"] = []
                cur()["title"] = "New chat"
                st.rerun()
