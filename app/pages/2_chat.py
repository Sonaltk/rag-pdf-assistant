import streamlit as st
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auth.auth import require_login, get_current_user_id, render_auth_sidebar
from database.connection import get_streamlit_db
from database.queries import (
    ConversationQueries,
    MessageQueries,
    DocumentQueries
)

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Chat — RAG Assistant",
    page_icon  = "💬",
    layout     = "wide"
)

# ─────────────────────────────────────────
# Auth + sidebar
# ─────────────────────────────────────────
render_auth_sidebar()
require_login()

user_id = get_current_user_id()

# ─────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────
defaults = {
    "ingested_pdf":    None,
    "chunk_stats":     None,
    "chat_history":    [],
    "query_logs":      [],
    "pipeline":        "phase2",
    "current_doc_id":  None,
    "current_conv_id": None,
    "collection_name": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────
# Check document is loaded
# ─────────────────────────────────────────
if not st.session_state.ingested_pdf:
    st.warning("⚠️ No document loaded.")
    st.markdown("👈 Go to **My Documents** to select a PDF.")
    if st.button("📚 Go to Documents"):
        st.switch_page("pages/1_documents.py")
    st.stop()

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
stats = st.session_state.chunk_stats or {}
col_info, col_actions = st.columns([4, 1])

with col_info:
    st.title("💬 Chat")
    st.success(
        f"📄 **{st.session_state.ingested_pdf}** — "
        f"{stats.get('text_chunks', 0)} text | "
        f"{stats.get('figure_chunks', 0)} figures | "
        f"Pipeline: **{st.session_state.pipeline.upper()}**"
    )

with col_actions:
    st.markdown("###")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history    = []
        st.session_state.current_conv_id = None
        st.rerun()

    if st.button("📚 Switch doc", use_container_width=True):
        st.switch_page("pages/1_documents.py")

st.divider()

# ─────────────────────────────────────────
# Conversation selector (sidebar)
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Conversations")

    if st.session_state.current_doc_id:
        db    = get_streamlit_db()
        convs = ConversationQueries.get_document_conversations(
            db,
            st.session_state.current_doc_id,
            user_id
        )
        db.close()

        if st.button(
            "➕ New conversation",
            use_container_width=True
        ):
            st.session_state.chat_history    = []
            st.session_state.current_conv_id = None
            st.rerun()

        for conv in convs:
            is_active = st.session_state.current_conv_id == conv.id
            label     = f"{'▶ ' if is_active else ''}{conv.title[:30]}..."

            if st.button(
                label,
                key  = f"conv_{conv.id}",
                use_container_width = True,
                type = "primary" if is_active else "secondary"
            ):
                # Load this conversation's messages
                db2  = get_streamlit_db()
                msgs = MessageQueries.get_messages_as_dicts(
                    db2, conv.id
                )
                db2.close()

                st.session_state.current_conv_id = conv.id
                st.session_state.chat_history    = []

                for i in range(0, len(msgs) - 1, 2):
                    if i + 1 < len(msgs):
                        st.session_state.chat_history.append({
                            "question":    msgs[i]["content"],
                            "answer":      msgs[i+1]["content"],
                            "confidence":  msgs[i+1].get("confidence", ""),
                            "sources":     msgs[i+1].get("sources", []),
                            "pipeline":    msgs[i+1].get("pipeline", "phase2"),
                            "response_ms": msgs[i+1].get("response_ms", 0)
                        })
                st.rerun()

    st.divider()
    st.markdown("### ⚙️ Pipeline")
    pipeline = st.radio(
        "Select:",
        ["phase2", "phase1"],
        format_func=lambda x: (
            "🚀 Phase 2 (Hybrid)"
            if x == "phase2"
            else "📦 Phase 1 (Vector)"
        )
    )
    st.session_state.pipeline = pipeline


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def confidence_badge(confidence: str) -> str:
    colors = {
        "HIGH":   ("🟢", "#065f46", "#6ee7b7"),
        "MEDIUM": ("🟡", "#78350f", "#fcd34d"),
        "LOW":    ("🔴", "#7f1d1d", "#fca5a5"),
    }
    conf  = (confidence or "").upper()
    icon, bg, fg = colors.get(conf, ("⚪", "#1f2937", "#9ca3af"))
    return (
        f"<span style='background:{bg};color:{fg};"
        f"padding:2px 10px;border-radius:99px;"
        f"font-size:0.75rem;font-weight:600'>"
        f"{icon} {conf or 'UNKNOWN'}</span>"
    )


def display_sources(sources, col):
    with col:
        if not sources:
            return
        st.markdown("**📚 Sources**")
        for src in sources:
            is_figure  = src.get("is_figure", False) \
                if isinstance(src, dict) \
                else getattr(src, "is_figure", False)
            page       = src.get("page_number", 0) \
                if isinstance(src, dict) \
                else getattr(src, "page_number", 0)
            image_path = src.get("image_path", "") \
                if isinstance(src, dict) \
                else getattr(src, "image_path", "")
            figure_id  = src.get("figure_id", "") \
                if isinstance(src, dict) \
                else getattr(src, "figure_id", "")
            text       = src.get("text", "") \
                if isinstance(src, dict) \
                else getattr(src, "text", "")
            para       = src.get("paragraph_index", 0) \
                if isinstance(src, dict) \
                else getattr(src, "paragraph_index", 0)

            if is_figure:
                st.markdown(
                    f"<div style='background:#1e2130;"
                    f"border-left:3px solid #f59e0b;"
                    f"border-radius:0 8px 8px 0;"
                    f"padding:0.6rem 0.8rem;margin:0.3rem 0'>"
                    f"<span style='color:#f59e0b;font-size:0.8rem'>"
                    f"🖼️ Figure | Page {page}</span></div>",
                    unsafe_allow_html=True
                )
                if image_path and Path(image_path).exists():
                    with st.expander(
                        f"View Figure — Page {page}",
                        expanded=False
                    ):
                        st.image(
                            image_path,
                            caption=f"{figure_id} | Page {page}",
                            use_container_width=True
                        )
                        if text:
                            st.caption(text[:200] + "...")
            else:
                st.markdown(
                    f"<div style='background:#1e2130;"
                    f"border-left:3px solid #7c83fd;"
                    f"border-radius:0 8px 8px 0;"
                    f"padding:0.6rem 0.8rem;margin:0.3rem 0'>"
                    f"<span style='color:#7c83fd;font-size:0.8rem'>"
                    f"📝 Page {page} | Para {para}</span></div>",
                    unsafe_allow_html=True
                )
                if text:
                    with st.expander("View chunk", expanded=False):
                        st.caption(text[:300] + "...")


# ─────────────────────────────────────────
# Display chat history
# ─────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])

    with st.chat_message("assistant"):
        col_answer, col_sources = st.columns([3, 1], gap="large")
        with col_answer:
            st.markdown(
                confidence_badge(entry.get("confidence", "")),
                unsafe_allow_html=True
            )
            st.markdown("")
            st.markdown(entry["answer"])
            from_cache = entry.get("from_cache", False)
            st.caption(
                f"{'⚡ cached' if from_cache else '🔍 live'} | "
                f"⏱️ {entry.get('response_ms', 0)}ms | "
                f"Pipeline: {entry.get('pipeline', 'phase2').upper()}"
            )
        display_sources(entry.get("sources", []), col_sources)


# ─────────────────────────────────────────
# Handle pending question from suggestions
# ─────────────────────────────────────────
if st.session_state.get("pending_question"):
    question = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    question = st.chat_input("Ask a question about your PDF...")


# ─────────────────────────────────────────
# Process question
# ─────────────────────────────────────────
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        col_answer, col_sources = st.columns([3, 1], gap="large")

        with col_answer:
            with st.spinner("🔍 Searching..."):
                start_time = time.time()
                sources    = []

                try:
                    pipeline         = st.session_state.pipeline
                    collection_name  = st.session_state.get(
                        "collection_name"
                    )

                    if pipeline == "phase2":
                        if stats.get("figure_chunks", 0) > 0:
                            from phase4_production.multimodal.multimodal_pipeline \
                                import MultimodalRetriever
                            pdf_path  = (
                                f"data/raw/"
                                f"{st.session_state.ingested_pdf}"
                            )
                            #retriever = MultimodalRetriever(pdf_path)
                            retriever = MultimodalRetriever(
                                pdf_path        = pdf_path,
                                collection_name = collection_name
                            )
                            response  = retriever.ask(question)
                            st.write(f"DEBUG sources count: {len(sources)}")
                            for s in sources:
                                st.write(f"  is_figure: {getattr(s, 'is_figure', 'N/A')} | para: {getattr(s, 'paragraph_index', 'N/A')}")
                        else:
                            from phase2_production.generation.graph \
                                import RAGGraph
                            graph    = RAGGraph(
                                collection_name=collection_name
                            )
                            response = graph.ask(question)
                    else:
                        from phase1_fundamentals.generation.chain \
                            import RAGChain
                        chain    = RAGChain(
                            collection_name=collection_name
                        )
                        response = chain.ask(question)

                    elapsed_ms = int(
                        (time.time() - start_time) * 1000
                    )
                    answer     = response.get("answer", "")
                    confidence = response.get("confidence", "")
                    sources    = response.get("sources", [])
                    from_cache = response.get("from_cache", False)

                    # Display
                    st.markdown(
                        confidence_badge(confidence),
                        unsafe_allow_html=True
                    )
                    st.markdown("")
                    st.markdown(answer)
                    st.caption(
                        f"{'⚡ cached' if from_cache else '🔍 live'} | "
                        f"⏱️ {elapsed_ms}ms | "
                        f"Pipeline: {pipeline.upper()}"
                    )

                    # ── Save to database ──
                    db = get_streamlit_db()

                    # Create conversation if first message
                    if not st.session_state.current_conv_id:
                        conv = ConversationQueries.create_conversation(
                            db,
                            user_id,
                            st.session_state.current_doc_id,
                            question[:60]
                        )
                        db.commit()
                        st.session_state.current_conv_id = conv.id
                    else:
                        # Update conversation timestamp
                        ConversationQueries.update_title(
                            db,
                            st.session_state.current_conv_id,
                            question[:60]
                            if not st.session_state.chat_history
                            else st.session_state.chat_history[0]["question"]
                        )

                    # Save Q&A pair
                    MessageQueries.save_qa_pair(
                        db              = db,
                        conversation_id = st.session_state.current_conv_id,
                        question        = question,
                        answer          = answer,
                        confidence      = confidence,
                        sources         = sources,
                        pipeline        = pipeline,
                        response_ms     = elapsed_ms
                    )
                    db.commit()
                    db.close()

                    # Add to session history
                    st.session_state.chat_history.append({
                        "question":    question,
                        "answer":      answer,
                        "confidence":  confidence,
                        "sources":     sources,
                        "pipeline":    pipeline,
                        "response_ms": elapsed_ms,
                        "from_cache":  from_cache
                    })

                    # Add to query logs for analytics
                    st.session_state.query_logs.append({
                        "question":    question,
                        "confidence":  confidence,
                        "pipeline":    pipeline,
                        "response_ms": elapsed_ms,
                        "num_sources": len(sources)
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        display_sources(sources, col_sources)


# ─────────────────────────────────────────
# Suggested questions (empty chat only)
# ─────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("### 💡 Try asking...")
    suggestions = [
        "What is the main method proposed?",
        "What are the key contributions?",
        "What datasets were used for evaluation?",
        "What does the pipeline diagram show?",
        "How does this compare to existing methods?",
        "What are the limitations?",
    ]
    col1, col2 = st.columns(2)
    for i, s in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_question = s
                st.rerun()