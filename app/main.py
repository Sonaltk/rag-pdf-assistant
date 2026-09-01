import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────
st.set_page_config(
    page_title            = "RAG PDF Assistant",
    page_icon             = "🔍",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────
# Auth
# ─────────────────────────────────────────
from auth.auth import (
    init_auth_state,
    is_logged_in,
    get_current_username,
    get_current_user_id,
    render_auth_sidebar,
    render_login_page
)

init_auth_state()
render_auth_sidebar()

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Not logged in → show login page
# ─────────────────────────────────────────
if not is_logged_in():
    render_login_page()
    st.stop()

# ─────────────────────────────────────────
# Logged in → show dashboard
# ─────────────────────────────────────────
username = get_current_username()
user_id  = get_current_user_id()

st.title(f"🔍 Welcome back, {username}!")
st.markdown(
    "**Multimodal Document Q&A** — "
    "ask questions about any PDF with cited answers and figure citations."
)
st.divider()

# ─────────────────────────────────────────
# User stats from database
# ─────────────────────────────────────────
from database.connection import get_streamlit_db
from database.queries import DocumentQueries, ConversationQueries, MessageQueries

db    = get_streamlit_db()
docs  = DocumentQueries.get_user_documents(db, user_id)
ready = [d for d in docs if d.status == "ready"]

# Count total conversations and messages
total_convs = 0
total_msgs  = 0
for doc in ready:
    convs        = ConversationQueries.get_document_conversations(
        db, doc.id, user_id
    )
    total_convs += len(convs)
    for conv in convs:
        msgs        = MessageQueries.get_conversation_messages(db, conv.id)
        total_msgs += len(msgs)

db.close()

# Stats row
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{len(ready)}</h2>
        <p><strong>Documents</strong></p>
        <p>PDFs ingested</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{total_convs}</h2>
        <p><strong>Conversations</strong></p>
        <p>Chat sessions</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{total_msgs // 2}</h2>
        <p><strong>Questions asked</strong></p>
        <p>Total Q&A pairs</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    # Total chunks across all documents
    total_chunks = sum(d.total_chunks for d in ready)
    st.markdown(f"""
    <div class="metric-card">
        <h2>{total_chunks}</h2>
        <p><strong>Indexed chunks</strong></p>
        <p>Text + figure vectors</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────
# Quick actions
# ─────────────────────────────────────────
st.markdown("### 🚀 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(
        "📚 My Documents",
        use_container_width = True,
        type = "primary"
    ):
        st.switch_page("pages/1_documents.py")

with col2:
    if st.button(
        "💬 Continue Chatting",
        use_container_width = True,
        disabled = not st.session_state.get("ingested_pdf")
    ):
        st.switch_page("pages/2_chat.py")

with col3:
    if st.button(
        "📊 View Analytics",
        use_container_width = True
    ):
        st.switch_page("pages/3_analytics.py")

st.divider()

# ─────────────────────────────────────────
# Recent documents
# ─────────────────────────────────────────
if ready:
    st.markdown("### 📄 Recent Documents")

    for doc in ready[:3]:   # show last 3
        col_doc, col_action = st.columns([4, 1])

        with col_doc:
            st.markdown(
                f"""
                <div style='background:#1e2130;border-radius:8px;
                            padding:0.8rem 1rem;margin:0.3rem 0'>
                    <p style='margin:0;color:#e5e7eb;font-weight:500'>
                        📄 {doc.filename}
                    </p>
                    <p style='margin:2px 0 0 0;color:#6b7280;
                              font-size:0.8rem'>
                        📝 {doc.text_chunks} text &nbsp;|&nbsp;
                        🖼️ {doc.figure_chunks} figures &nbsp;|&nbsp;
                        🔢 {doc.total_chunks} total &nbsp;|&nbsp;
                        🕐 {doc.upload_time.strftime('%b %d') if doc.upload_time else ''}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_action:
            if st.button(
                "Open",
                key  = f"open_{doc.id}",
                use_container_width = True
            ):
                # Load document into session
                st.session_state.ingested_pdf    = doc.filename
                st.session_state.chunk_stats     = {
                    "pdf_name":      doc.filename,
                    "pages":         doc.page_count,
                    "text_chunks":   doc.text_chunks,
                    "figure_chunks": doc.figure_chunks,
                    "total_chunks":  doc.total_chunks,
                }
                st.session_state.collection_name = doc.collection_name
                st.session_state.current_doc_id  = doc.id
                st.session_state.chat_history    = []
                st.switch_page("pages/2_chat.py")

    st.divider()

# ─────────────────────────────────────────
# RAGAS scores
# ─────────────────────────────────────────
st.markdown("### 📈 RAGAS Evaluation Results")

import json
scores_path = Path("phase3_evaluation/eval/scores.json")

if scores_path.exists():
    with open(scores_path) as f:
        scores = json.load(f)

    p1 = scores.get("phase1", {})
    p2 = scores.get("phase2", {})

    metrics = [
        ("Faithfulness",      "faithfulness"),
        ("Answer Relevancy",  "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall",    "context_recall"),
    ]

    cols = st.columns(4)
    for col, (label, key) in zip(cols, metrics):
        p1s   = p1.get(key, 0)
        p2s   = p2.get(key, 0)
        delta = round((p2s - p1s) * 100, 1)
        with col:
            st.metric(
                label = label,
                value = f"{p2s:.4f}",
                delta = f"{delta:+.1f}% vs Phase 1"
            )
else:
    st.info("Run RAGAS evaluation to see scores here.")

st.divider()

# ─────────────────────────────────────────
# Architecture overview
# ─────────────────────────────────────────
st.markdown("### 🏗️ System Architecture")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Phase 2 Pipeline:**
    ```
    Question
       ↓
    BM25 Keywords ──┐
    Vector Semantic─┤→ RRF Fusion
                    ↓
             Cohere Re-ranker
                    ↓
             LangGraph routing
                    ↓
             Groq LLM → Answer
    ```
    """)

with col2:
    st.markdown("""
    **Phase 4 Additions:**
    ```
    PDF → PyMuPDF → figures
       → LLaVA → descriptions
       → ChromaDB (per user)
       → BM25 (per document)
    
    User system:
    Login → Documents → Chat
    History saved to SQLite
    ```
    """)

st.caption(
    "Built with LangGraph · ChromaDB · Cohere · LLaVA · "
    "SQLite · Streamlit"
)