import streamlit as st
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Chat — RAG Assistant",
    page_icon  = "💬",
    layout     = "wide"
)

# ─────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────
if "ingested_pdf"  not in st.session_state:
    st.session_state.ingested_pdf  = None
if "chunk_stats"   not in st.session_state:
    st.session_state.chunk_stats   = None
if "chat_history"  not in st.session_state:
    st.session_state.chat_history  = []
if "query_logs"    not in st.session_state:
    st.session_state.query_logs    = []
if "pipeline"      not in st.session_state:
    st.session_state.pipeline      = "phase2"

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("💬 Chat with your PDF")

# Check if PDF is ingested
if not st.session_state.ingested_pdf:
    st.warning(
        "⚠️ No document loaded. "
        "Please go to the **Upload** page first."
    )
    st.stop()

# Show current document
col_info, col_clear = st.columns([4, 1])
with col_info:
    stats = st.session_state.chunk_stats or {}
    st.success(
        f"📄 **{st.session_state.ingested_pdf}** — "
        f"{stats.get('text_chunks', 0)} text chunks, "
        f"{stats.get('figure_chunks', 0)} figure chunks | "
        f"Pipeline: **{st.session_state.pipeline.upper()}**"
    )
with col_clear:
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.divider()

# ─────────────────────────────────────────
# Helper — confidence badge
# ─────────────────────────────────────────
def confidence_badge(confidence: str) -> str:
    """Return colored HTML badge for confidence level."""
    colors = {
        "HIGH":   ("🟢", "#065f46", "#6ee7b7"),
        "MEDIUM": ("🟡", "#78350f", "#fcd34d"),
        "LOW":    ("🔴", "#7f1d1d", "#fca5a5"),
    }
    conf_upper = confidence.upper() if confidence else "UNKNOWN"
    icon, bg, fg = colors.get(conf_upper, ("⚪", "#1f2937", "#9ca3af"))
    return (
        f"<span style='background:{bg};color:{fg};"
        f"padding:2px 10px;border-radius:99px;"
        f"font-size:0.75rem;font-weight:600'>"
        f"{icon} {conf_upper}</span>"
    )


# ─────────────────────────────────────────
# Helper — display sources
# ─────────────────────────────────────────
def display_sources(sources, col):
    """Display source citations in the right column."""
    with col:
        st.markdown("**📚 Sources**")
        for i, src in enumerate(sources, 1):
            is_figure = getattr(src, "is_figure", False)

            if is_figure:
                # Figure source — show image
                image_path = getattr(src, "image_path", "")
                figure_id  = getattr(src, "figure_id", "")

                st.markdown(
                    f"<div style='background:#1e2130;"
                    f"border-left:3px solid #f59e0b;"
                    f"border-radius:0 8px 8px 0;"
                    f"padding:0.6rem 0.8rem;margin:0.3rem 0'>"
                    f"<span style='color:#f59e0b;font-size:0.8rem'>"
                    f"🖼️ Figure | Page {src.page_number}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Show actual figure image
                if image_path and Path(image_path).exists():
                    with st.expander(
                        f"View Figure — Page {src.page_number}",
                        expanded=False
                    ):
                        st.image(
                            image_path,
                            caption=f"{figure_id} | Page {src.page_number}",
                            use_container_width=True
                        )
                        st.caption(src.text[:200] + "...")
            else:
                # Text source
                score = getattr(src, "similarity_score", 0)
                score = getattr(src, "rerank_score", score)

                st.markdown(
                    f"<div style='background:#1e2130;"
                    f"border-left:3px solid #7c83fd;"
                    f"border-radius:0 8px 8px 0;"
                    f"padding:0.6rem 0.8rem;margin:0.3rem 0'>"
                    f"<span style='color:#7c83fd;font-size:0.8rem'>"
                    f"📝 Page {src.page_number} | "
                    f"Para {src.paragraph_index} | "
                    f"Score {round(float(score), 3)}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                with st.expander("View chunk", expanded=False):
                    st.caption(src.text[:300] + "...")


# ─────────────────────────────────────────
# Display chat history
# ─────────────────────────────────────────
for entry in st.session_state.chat_history:
    # User message
    with st.chat_message("user"):
        st.markdown(entry["question"])

    # Assistant message
    with st.chat_message("assistant"):
        col_answer, col_sources = st.columns([3, 1], gap="large")

        with col_answer:
            # Confidence badge
            st.markdown(
                confidence_badge(entry.get("confidence", "")),
                unsafe_allow_html=True
            )
            st.markdown("")   # spacer

            # Answer
            st.markdown(entry["answer"])

            # Response time
            st.caption(
                f"⏱️ {entry.get('response_ms', 0)}ms | "
                f"Pipeline: {entry.get('pipeline', 'phase2').upper()}"
            )

        # Sources
        if entry.get("sources"):
            display_sources(entry["sources"], col_sources)


# ─────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────
# Handle suggestion button clicks
if "pending_question" in st.session_state and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    question = st.chat_input(
        "Ask a question about your PDF...",
        disabled=not st.session_state.ingested_pdf
    )


if question:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Run pipeline
    with st.chat_message("assistant"):
        col_answer, col_sources = st.columns([3, 1], gap="large")

        with col_answer:
            with st.spinner("🔍 Searching and generating answer..."):
                start_time = time.time()

                try:
                    pipeline = st.session_state.pipeline

                    if pipeline == "phase2":
                        # Use multimodal retriever if figures exist
                        if st.session_state.chunk_stats and \
                           st.session_state.chunk_stats.get("figure_chunks", 0) > 0:
                            from phase4_production.multimodal.multimodal_pipeline \
                                import MultimodalRetriever
                            pdf_path  = (
                                f"data/raw/"
                                f"{st.session_state.ingested_pdf}"
                            )
                            retriever = MultimodalRetriever(pdf_path)
                            response  = retriever.ask(question)
                        else:
                            from phase2_production.generation.graph import RAGGraph
                            graph    = RAGGraph()
                            response = graph.ask(question)
                    else:
                        from phase1_fundamentals.generation.chain import RAGChain
                        chain    = RAGChain()
                        response = chain.ask(question)

                    elapsed_ms = int(
                        (time.time() - start_time) * 1000
                    )

                    answer     = response.get("answer", "No answer generated")
                    confidence = response.get("confidence", "UNKNOWN")
                    sources    = response.get("sources", [])

                    # Display confidence badge
                    st.markdown(
                        confidence_badge(confidence),
                        unsafe_allow_html=True
                    )
                    st.markdown("")

                    # Display answer
                    st.markdown(answer)

                    # Response time
                    st.caption(
                        f"⏱️ {elapsed_ms}ms | "
                        f"Pipeline: {pipeline.upper()}"
                    )

                    # Save to history
                    st.session_state.chat_history.append({
                        "question":    question,
                        "answer":      answer,
                        "confidence":  confidence,
                        "sources":     sources,
                        "pipeline":    pipeline,
                        "response_ms": elapsed_ms
                    })

                    # Save to query logs for analytics
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
                    sources = []
                    elapsed_ms = 0

        # Display sources
        if "sources" in locals() and sources:
            display_sources(sources, col_sources)

    st.rerun()


# ─────────────────────────────────────────
# Suggested questions
# ─────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("### 💡 Try asking...")
    suggestions = [
        "What is the main method proposed in this paper?",
        "What are the key contributions of this work?",
        "What datasets were used for evaluation?",
        "What does the pipeline diagram show?",
        "How does the method compare to existing approaches?",
        "What are the limitations of the proposed method?",
    ]
    col1, col2 = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(
                suggestion,
                key=f"suggestion_{i}",
                use_container_width=True
            ):
                st.session_state["pending_question"] = suggestion
                