"""
app.py — Streamlit frontend for the RAG system.
Apple-inspired: light, airy, typographic, precise.
"""

import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st

from config import settings
from document_processor import DocumentProcessor
from rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Intellect — RAG System",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Apple-inspired: SF Pro feel, pure whites, precise spacing, hairlines
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #1d1d1f;
    background-color: #f5f5f7;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { background: #f5f5f7; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e0e0e5;
    padding-top: 0;
}
[data-testid="stSidebar"] > div:first-child { padding: 0; }
[data-testid="stSidebar"] * { color: #1d1d1f !important; }

.sidebar-logo {
    padding: 32px 28px 24px;
    border-bottom: 1px solid #e0e0e5;
    margin-bottom: 24px;
}
.sidebar-logo .mark {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1d1d1f;
    letter-spacing: -0.02em;
    display: block;
}
.sidebar-logo .sub {
    font-size: 0.72rem;
    color: #86868b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
    display: block;
}

.sidebar-section {
    padding: 0 20px 20px;
}
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 500;
    color: #86868b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #f5f5f7;
    border: 1.5px dashed #c7c7cc;
    border-radius: 12px;
    padding: 4px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #1d1d1f; }
[data-testid="stFileUploader"] label { display: none; }

/* ── Buttons ── */
.stButton > button {
    background: #1d1d1f !important;
    color: #f5f5f7 !important;
    border: none !important;
    border-radius: 980px !important;
    padding: 10px 22px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
    transition: background 0.2s, transform 0.15s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #3a3a3c !important;
    transform: scale(1.01) !important;
}
.stButton > button:active { transform: scale(0.99) !important; }

/* ── Stats pill ── */
.stats-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #f5f5f7;
    border: 1px solid #e0e0e5;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 12px 0 0;
}
.stats-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #34c759;
    flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(52,199,89,0.15);
}
.stats-text {
    font-size: 0.78rem;
    color: #3a3a3c;
    line-height: 1.5;
}
.stats-text strong { font-weight: 500; color: #1d1d1f; }

/* ── Divider ── */
.hairline {
    height: 1px;
    background: #e0e0e5;
    margin: 20px 0;
}

/* ── Main area ── */
.main-wrap {
    max-width: 780px;
    margin: 0 auto;
    padding: 56px 32px 140px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    margin-bottom: 56px;
    animation: fadeUp 0.6s ease both;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 500;
    color: #86868b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    font-weight: 400;
    color: #1d1d1f;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 16px;
}
.hero-title em {
    font-style: italic;
    color: #86868b;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6e6e73;
    font-weight: 300;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
}

/* ── Chat messages ── */
.chat-area {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.msg-wrap {
    animation: fadeUp 0.35s ease both;
    margin-bottom: 2px;
}

.msg-user-wrap {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 16px;
}
.msg-user {
    background: #1d1d1f;
    color: #f5f5f7;
    border-radius: 20px 20px 5px 20px;
    padding: 13px 18px;
    font-size: 0.9rem;
    font-weight: 300;
    line-height: 1.55;
    max-width: 75%;
}

.msg-assistant-wrap {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 4px;
}
.msg-assistant {
    background: #ffffff;
    border: 1px solid #e0e0e5;
    border-radius: 5px 20px 20px 20px;
    padding: 16px 20px;
    font-size: 0.9rem;
    font-weight: 300;
    line-height: 1.65;
    max-width: 82%;
    color: #1d1d1f;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

/* ── Source chips ── */
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0 20px 4px;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #f5f5f7;
    border: 1px solid #e0e0e5;
    border-radius: 980px;
    padding: 4px 12px 4px 9px;
    font-size: 0.72rem;
    color: #6e6e73;
    font-weight: 400;
    letter-spacing: 0.01em;
}
.source-chip-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #86868b;
    display: inline-block;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 72px 20px;
    animation: fadeUp 0.5s ease both;
}
.empty-glyph {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #c7c7cc;
    margin-bottom: 16px;
    line-height: 1;
}
.empty-title {
    font-size: 1.1rem;
    font-weight: 400;
    color: #3a3a3c;
    margin-bottom: 8px;
}
.empty-sub {
    font-size: 0.83rem;
    color: #86868b;
    font-weight: 300;
    line-height: 1.6;
}

/* ── Chat input override ── */
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    border: 1px solid #e0e0e5 !important;
    border-radius: 980px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 300 !important;
    color: #1d1d1f !important;
    padding: 14px 24px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #1d1d1f !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.1) !important;
    outline: none !important;
}

/* ── Indexed file list ── */
.file-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid #f0f0f5;
    font-size: 0.78rem;
    color: #3a3a3c;
}
.file-item:last-child { border-bottom: none; }
.file-icon { color: #86868b; font-size: 0.85rem; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c7c7cc; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session() -> None:
    defaults = {
        "chat_history": [],
        "indexed_files": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_engine() -> RAGEngine:
    engine = RAGEngine()
    if engine.index_exists:
        try:
            engine.load_existing_index()
        except Exception as exc:
            logger.warning("Could not load existing index: %s", exc)
    return engine

@st.cache_resource(show_spinner=False)
def _get_processor() -> DocumentProcessor:
    return DocumentProcessor()

engine = _get_engine()
processor = _get_processor()
stats = engine.get_index_stats()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">
            <span class="mark">Intellect</span>
            <span class="sub">RAG · Document Intelligence</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("Index Documents"):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            new_files = [f for f in uploaded_files
                         if f.name not in st.session_state.indexed_files]
            if not new_files:
                st.info("Already indexed.")
            else:
                progress = st.progress(0, text="Preparing…")
                all_chunks = []
                for i, uf in enumerate(new_files):
                    progress.progress(
                        (i + 1) / len(new_files),
                        text=f"Processing {uf.name}…"
                    )
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.read())
                        tmp_path = Path(tmp.name)
                    try:
                        chunks = processor.load_and_split(tmp_path)
                        for chunk in chunks:
                            chunk.metadata["file_name"] = uf.name
                        all_chunks.extend(chunks)
                        st.session_state.indexed_files.append(uf.name)
                    except Exception as exc:
                        st.error(f"Failed: {uf.name} — {exc}")
                        logger.error("Processing error %s: %s", uf.name, exc)
                    finally:
                        tmp_path.unlink(missing_ok=True)

                if all_chunks:
                    progress.progress(1.0, text="Building index…")
                    try:
                        engine.add_documents(all_chunks)
                        st.success(f"{len(all_chunks):,} chunks indexed.")
                    except Exception as exc:
                        st.error(f"Indexing failed: {exc}")
                progress.empty()

    # Stats
    stats = engine.get_index_stats()
    if stats["indexed"]:
        st.markdown(f"""
            <div class="stats-pill">
                <div class="stats-dot"></div>
                <div class="stats-text">
                    <strong>{stats['total_vectors']:,}</strong> vectors &nbsp;·&nbsp;
                    <strong>{len(st.session_state.indexed_files)}</strong> file(s)
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.indexed_files:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sidebar-label">Indexed files</div>',
                        unsafe_allow_html=True)
            files_html = "".join(
                f'<div class="file-item"><span class="file-icon">↳</span>{name}</div>'
                for name in st.session_state.indexed_files
            )
            st.markdown(files_html, unsafe_allow_html=True)

    st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)

    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        engine.clear_memory()
        st.rerun()

    st.markdown("""
        <div style='padding: 8px 0 0; font-size:0.68rem; color:#c7c7cc; letter-spacing:0.04em;'>
            LLAMA 3.1 · GROQ · FAISS<br>LANGCHAIN · HUGGINGFACE
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

# Hero
st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Document Intelligence</div>
        <h1 class="hero-title">Ask anything.<br><em>Know everything.</em></h1>
        <p class="hero-sub">
            Upload your documents. Ask questions in plain language.
            Get precise answers with exact source references.
        </p>
    </div>
""", unsafe_allow_html=True)

# Chat history
if st.session_state.chat_history:
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
                <div class="msg-wrap msg-user-wrap">
                    <div class="msg-user">{msg["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="msg-wrap msg-assistant-wrap">
                    <div class="msg-assistant">{msg["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
            if msg.get("sources"):
                chips = "".join(
                    f'<span class="source-chip">'
                    f'<span class="source-chip-dot"></span>{s}</span>'
                    for s in msg["sources"]
                )
                st.markdown(
                    f'<div class="sources-row">{chips}</div>',
                    unsafe_allow_html=True
                )
    st.markdown('</div>', unsafe_allow_html=True)

elif stats["indexed"]:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-glyph">◎</div>
            <div class="empty-title">Ready to answer</div>
            <div class="empty-sub">
                Your documents are indexed.<br>
                Type a question below to begin.
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-glyph">◎</div>
            <div class="empty-title">No documents yet</div>
            <div class="empty-sub">
                Upload PDFs in the sidebar and click<br>
                <strong>Index Documents</strong> to get started.
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if user_input := st.chat_input(
    "Ask a question about your documents…",
    disabled=not stats["indexed"],
):
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input, "sources": []}
    )

    with st.spinner(""):
        try:
            response = engine.query(user_input)
            answer = response["answer"]
            sources = response["sources"]
        except Exception as exc:
            answer = f"Something went wrong: {exc}"
            sources = []
            logger.error("Query error: %s", exc, exc_info=True)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()
