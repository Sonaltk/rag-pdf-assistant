import streamlit as st
from pathlib import Path
from auth.auth import init_auth_state, is_logged_in, render_auth_sidebar

init_auth_state()
render_auth_sidebar()

if not is_logged_in():
    st.warning("👈 Please login first using the Login page in the sidebar.")
    st.stop()


# ─────────────────────────────────────────
# Page Configuration
# Must be the FIRST streamlit call
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "RAG PDF Assistant",
    page_icon  = "🔍",
    layout     = "wide",          # full width layout
    initial_sidebar_state = "expanded"
)


# ─────────────────────────────────────────
# Custom CSS
# Why custom CSS?
#   Streamlit default looks basic
#   Custom CSS makes it look professional
#   Interviewers notice polished UI
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }

    /* Card style */
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }

    .metric-card h2 {
        color: #7c83fd;
        font-size: 2rem;
        margin: 0;
    }

    .metric-card p {
        color: #9ca3af;
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
    }

    /* Confidence badges */
    .badge-high {
        background: #065f46;
        color: #6ee7b7;
        padding: 2px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-medium {
        background: #78350f;
        color: #fcd34d;
        padding: 2px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-low {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 2px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Source cards */
    .source-card {
        background: #1e2130;
        border-left: 3px solid #7c83fd;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }

    .source-card-figure {
        border-left-color: #f59e0b;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #161b2e;
    }

    /* Chat message styling */
    .stChatMessage {
        background: #1e2130;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Session State Initialization
# Why session state?
#   Streamlit reruns entire script on every interaction
#   Session state persists data across reruns
#   Like global variables that survive reruns
#
# What we store:
#   ingested_pdf    → which PDF is currently loaded
#   chunk_stats     → ingestion summary (counts)
#   chat_history    → list of (question, answer) pairs
#   query_logs      → for analytics page
# ─────────────────────────────────────────
def init_session_state():
    defaults = {
        "ingested_pdf":   None,     # currently loaded PDF name
        "chunk_stats":    None,     # {text_chunks, figure_chunks, total}
        "chat_history":   [],       # [{question, answer, sources, confidence, time}]
        "query_logs":     [],       # [{question, response_ms, confidence, pipeline}]
        "pipeline":       "phase2", # which pipeline to use
        "active_page":    "home",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/search.png",
        width=60
    )
    st.title("RAG PDF Assistant")
    st.caption("Multimodal Document Q&A System")

    st.divider()

    # Current document status
    st.markdown("### 📄 Current Document")
    if st.session_state.ingested_pdf:
        st.success(f"✅ {st.session_state.ingested_pdf}")
        if st.session_state.chunk_stats:
            stats = st.session_state.chunk_stats
            st.caption(
                f"📝 {stats.get('text_chunks', 0)} text chunks  \n"
                f"🖼️ {stats.get('figure_chunks', 0)} figure chunks  \n"
                f"🔢 {stats.get('total_chunks', 0)} total vectors"
            )
    else:
        st.warning("⚠️ No document loaded")
        st.caption("Go to Upload page to ingest a PDF")

    st.divider()

    # Pipeline selector
    st.markdown("### ⚙️ Pipeline")
    pipeline = st.radio(
        "Select pipeline:",
        ["phase2", "phase1"],
        format_func=lambda x: (
            "🚀 Phase 2 (Hybrid + Rerank)"
            if x == "phase2"
            else "📦 Phase 1 (Vector only)"
        ),
        index=0
    )
    st.session_state.pipeline = pipeline

    st.divider()

    # Quick stats
    if st.session_state.query_logs:
        logs = st.session_state.query_logs
        avg_time = round(
            sum(l["response_ms"] for l in logs) / len(logs)
        )
        high_conf = sum(
            1 for l in logs if l.get("confidence") == "HIGH"
        )
        st.markdown("### 📊 Session Stats")
        st.caption(f"Queries: {len(logs)}")
        st.caption(f"Avg response: {avg_time}ms")
        st.caption(
            f"High confidence: {high_conf}/{len(logs)}"
        )

    st.divider()
    st.caption("Built with LangGraph · ChromaDB · Cohere · LLaVA")
    st.caption("Phase 1 → 2 → 3 → 4 (Multimodal)")


# ─────────────────────────────────────────
# Home / Welcome Screen
# ─────────────────────────────────────────
st.title("🔍 RAG PDF Assistant")
st.markdown(
    "**Multimodal Document Q&A** — ask questions about any PDF, "
    "get cited answers with page references and figure citations."
)

st.divider()

# Feature cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>📄</h2>
        <p><strong>Upload PDF</strong></p>
        <p>Ingest any research paper or document</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>💬</h2>
        <p><strong>Ask Questions</strong></p>
        <p>Hybrid BM25 + vector retrieval</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>🖼️</h2>
        <p><strong>Figure Citations</strong></p>
        <p>LLaVA vision model indexes diagrams</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h2>📊</h2>
        <p><strong>Analytics</strong></p>
        <p>Response time · Confidence · RAGAS</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Architecture overview
st.markdown("### 🏗️ System Architecture")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    **Phase 2 Pipeline (active):**
    ```
    Question
       ↓
    BM25 Keywords ──┐
    Vector Semantic─┤→ RRF Fusion (33 candidates)
                    ↓
             Cohere Re-ranker (top 5)
                    ↓
             LangGraph routing
             ├── HIGH → confident prompt
             └── LOW  → careful prompt
                    ↓
             Groq LLM (llama-3.1-8b)
                    ↓
             Citation Enforcer
                    ↓
             Final Answer
    ```
    """)

with col2:
    st.markdown("""
    **Phase 4 Additions (multimodal):**
    ```
    PDF Ingestion
       ↓
    PyMuPDF → extract figures
       ↓
    LLaVA → describe figures
       ↓
    Figure descriptions → ChromaDB
    (alongside text chunks)
       ↓
    Retrieval finds TEXT + FIGURES
       ↓
    Answer cites:
      [Page 3, Para 0] text source
      [Page 5, Figure 1] image source
    ```
    """)

st.divider()

# RAGAS results
st.markdown("### 📈 Evaluation Results (RAGAS)")

metrics = {
    "Faithfulness":      (0.9543, 0.8472),
    "Answer Relevancy":  (0.7756, 0.9800),
    "Context Precision": (0.7339, 0.9554),
    "Context Recall":    (0.8333, 0.9500),
}

cols = st.columns(4)
for col, (metric, (p1, p2)) in zip(cols, metrics.items()):
    delta  = round((p2 - p1) * 100, 1)
    with col:
        st.metric(
            label = metric,
            value = f"{p2:.4f}",
            delta = f"{delta:+.1f}% vs Phase 1"
        )

st.divider()

# Navigation guide
st.markdown("### 🧭 Getting Started")
st.markdown("""
1. **Upload page** → drag and drop your PDF and click Ingest
2. **Chat page** → ask questions and see cited answers
3. **Analytics page** → view response times and confidence scores
""")