import streamlit as st
import sys
import time
import shutil
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auth.auth import require_login, get_current_user_id, render_auth_sidebar
from database.connection import get_streamlit_db
from database.queries import DocumentQueries, ConversationQueries

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "My Documents — RAG Assistant",
    page_icon  = "📚",
    layout     = "wide"
)

# ─────────────────────────────────────────
# Auth guard + sidebar
# ─────────────────────────────────────────
render_auth_sidebar()
require_login()

user_id = get_current_user_id()

# ─────────────────────────────────────────
# Session state
# ─────────────────────────────────────────
if "current_doc_id"  not in st.session_state:
    st.session_state.current_doc_id  = None
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "chunk_stats"     not in st.session_state:
    st.session_state.chunk_stats     = None
if "ingested_pdf"    not in st.session_state:
    st.session_state.ingested_pdf    = None
if "chat_history"    not in st.session_state:
    st.session_state.chat_history    = []
if "pipeline"        not in st.session_state:
    st.session_state.pipeline        = "phase2"

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("📚 My Documents")
st.markdown("Manage your ingested PDFs and conversations.")
st.divider()

# ─────────────────────────────────────────
# Helper — load document into session
# ─────────────────────────────────────────
def load_document(doc):
    """
    Load a document into session state for chat.

    What this does:
      Sets collection name → retriever uses this collection
      Sets chunk stats → sidebar shows counts
      Sets ingested_pdf → chat page knows what's loaded
      Clears chat history → fresh start for this document
    """
    st.session_state.current_doc_id = doc.id
    st.session_state.ingested_pdf   = doc.filename
    st.session_state.chunk_stats    = {
        "pdf_name":      doc.filename,
        "pages":         doc.page_count,
        "text_chunks":   doc.text_chunks,
        "figure_chunks": doc.figure_chunks,
        "total_chunks":  doc.total_chunks,
    }
    # Store collection name so retriever uses right ChromaDB collection
    st.session_state.collection_name = doc.collection_name


# ─────────────────────────────────────────
# Section 1 — Existing Documents
# ─────────────────────────────────────────
st.markdown("### 📂 Your Documents")

db   = get_streamlit_db()
docs = DocumentQueries.get_user_documents(db, user_id)

if not docs:
    st.info(
        "No documents yet. "
        "Upload your first PDF below!"
    )
else:
    for doc in docs:
        # Status color
        status_colors = {
            "ready":     "🟢",
            "ingesting": "🟡",
            "failed":    "🔴"
        }
        status_icon = status_colors.get(doc.status, "⚪")

        # Is this the currently loaded document?
        is_active = st.session_state.current_doc_id == doc.id
        border_color = "#7c83fd" if is_active else "#2d3250"

        # Document card
        st.markdown(
            f"""
            <div style='border:1px solid {border_color};
                        border-radius:12px;padding:1rem;
                        margin:0.5rem 0;background:#1e2130'>
                <div style='display:flex;justify-content:space-between'>
                    <span style='font-size:1rem;font-weight:500;
                                 color:#e5e7eb'>
                        📄 {doc.filename}
                        {"&nbsp;&nbsp;<span style='color:#7c83fd;font-size:0.75rem'>● Active</span>" if is_active else ""}
                    </span>
                    <span style='color:#6b7280;font-size:0.8rem'>
                        {status_icon} {doc.status.upper()}
                    </span>
                </div>
                <div style='color:#9ca3af;font-size:0.8rem;margin-top:0.4rem'>
                    📝 {doc.text_chunks} text &nbsp;|&nbsp;
                    🖼️ {doc.figure_chunks} figures &nbsp;|&nbsp;
                    🔢 {doc.total_chunks} total &nbsp;|&nbsp;
                    📄 {doc.page_count} pages &nbsp;|&nbsp;
                    🕐 {doc.upload_time.strftime('%b %d, %Y') if doc.upload_time else 'Unknown'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Conversations for this document
        convs = ConversationQueries.get_document_conversations(
            db, doc.id, user_id
        )

        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            if doc.status == "ready":
                if st.button(
                    f"💬 New Chat",
                    key  = f"new_chat_{doc.id}",
                    use_container_width = True
                ):
                    load_document(doc)
                    # Create new conversation in DB
                    new_conv = ConversationQueries.create_conversation(
                        db, user_id, doc.id, "New conversation"
                    )
                    db.commit()
                    st.session_state.current_conv_id = new_conv.id
                    st.session_state.chat_history    = []
                    st.switch_page("pages/2_chat.py")

        with col2:
            if convs and doc.status == "ready":
                # Continue most recent conversation
                latest_conv = convs[0]
                if st.button(
                    f"▶️ Continue ({len(convs)} conv{'s' if len(convs)>1 else ''})",
                    key  = f"continue_{doc.id}",
                    use_container_width = True
                ):
                    load_document(doc)
                    st.session_state.current_conv_id = latest_conv.id
                    # Load conversation history from DB
                    from database.queries import MessageQueries
                    msgs = MessageQueries.get_messages_as_dicts(
                        db, latest_conv.id
                    )
                    # Convert to chat_history format
                    st.session_state.chat_history = []
                    for i in range(0, len(msgs) - 1, 2):
                        if i + 1 < len(msgs):
                            st.session_state.chat_history.append({
                                "question":    msgs[i]["content"],
                                "answer":      msgs[i+1]["content"],
                                "confidence":  msgs[i+1]["confidence"],
                                "sources":     msgs[i+1]["sources"],
                                "pipeline":    msgs[i+1]["pipeline"],
                                "response_ms": msgs[i+1]["response_ms"]
                            })
                    st.switch_page("pages/2_chat.py")

        with col3:
            if convs:
                with st.expander(
                    f"📋 {len(convs)} conversation(s)",
                    expanded=False
                ):
                    for conv in convs:
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.caption(
                                f"💬 {conv.title[:40]}... | "
                                f"{conv.updated_at.strftime('%b %d') if conv.updated_at else ''}"
                            )
                        with c2:
                            if st.button(
                                "Open",
                                key=f"open_{conv.id}",
                                use_container_width=True
                            ):
                                load_document(doc)
                                st.session_state.current_conv_id = conv.id
                                from database.queries import MessageQueries
                                msgs = MessageQueries.get_messages_as_dicts(
                                    db, conv.id
                                )
                                st.session_state.chat_history = []
                                for i in range(0, len(msgs) - 1, 2):
                                    if i + 1 < len(msgs):
                                        st.session_state.chat_history.append({
                                            "question":    msgs[i]["content"],
                                            "answer":      msgs[i+1]["content"],
                                            "confidence":  msgs[i+1]["confidence"],
                                            "sources":     msgs[i+1]["sources"],
                                            "pipeline":    msgs[i+1]["pipeline"],
                                            "response_ms": msgs[i+1]["response_ms"]
                                        })
                                st.switch_page("pages/2_chat.py")

        with col4:
            if st.button(
                "🗑️",
                key  = f"delete_{doc.id}",
                help = "Delete this document and all conversations",
                use_container_width = True
            ):
                st.session_state[f"confirm_delete_{doc.id}"] = True

            # Confirm delete
            if st.session_state.get(f"confirm_delete_{doc.id}"):
                st.warning(f"Delete **{doc.filename}**?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        "Yes, delete",
                        key  = f"yes_delete_{doc.id}",
                        type = "primary"
                    ):
                        # Delete from DB
                        DocumentQueries.delete_document(
                            db, doc.id, user_id
                        )
                        db.commit()

                        # Delete ChromaDB collection
                        try:
                            import chromadb
                            from config.settings import CHROMA_DB_PATH
                            chroma = chromadb.PersistentClient(
                                path=CHROMA_DB_PATH
                            )
                            chroma.delete_collection(doc.collection_name)
                        except Exception:
                            pass  # Collection may not exist

                        # Clear session if this was active doc
                        if st.session_state.current_doc_id == doc.id:
                            st.session_state.current_doc_id  = None
                            st.session_state.ingested_pdf    = None
                            st.session_state.chunk_stats     = None
                            st.session_state.chat_history    = []

                        st.success(f"Deleted {doc.filename}")
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"cancel_delete_{doc.id}"):
                        st.session_state[f"confirm_delete_{doc.id}"] = False
                        st.rerun()

        st.markdown("---")

db.close()

# ─────────────────────────────────────────
# Section 2 — Upload New PDF
# ─────────────────────────────────────────
st.markdown("### ⬆️ Upload New PDF")

with st.expander("Upload and ingest a new document", expanded=not docs):
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="doc_uploader"
    )

    if uploaded_file:
        file_size_mb = round(uploaded_file.size / 1024 / 1024, 2)
        st.caption(f"📄 {uploaded_file.name} ({file_size_mb} MB)")

        col1, col2 = st.columns(2)
        with col1:
            skip_vision = st.checkbox(
                "Skip figure extraction (faster)",
                value=False
            )
        with col2:
            pipeline_choice = st.selectbox(
                "Pipeline",
                ["Phase 2 (Hybrid)", "Phase 1 (Vector)"],
                index=0
            )

        if st.button(
            "🚀 Ingest PDF",
            type="primary",
            use_container_width=True
        ):
            # Check if already ingested
            db2      = get_streamlit_db()
            existing = DocumentQueries.document_exists(
                db2, user_id, uploaded_file.name
            )

            if existing:
                st.warning(
                    f"**{uploaded_file.name}** is already ingested! "
                    f"Loading existing document..."
                )
                load_document(existing)
                db2.close()
                st.rerun()
            else:
                # Create document record
                doc_record = DocumentQueries.create_document(
                    db2, user_id, uploaded_file.name
                )
                doc_id = doc_record.id
                db2.commit()
                db2.close()

                # Run ingestion pipeline
                progress = st.progress(0)
                status   = st.empty()

                def update(msg, pct):
                    status.markdown(f"⏳ {msg}")
                    progress.progress(pct)

                try:
                    # Save file
                    update("Saving file...", 5)
                    raw_dir    = Path("data/raw")
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    final_path = raw_dir / uploaded_file.name
                    with open(final_path, "wb") as f:
                        f.write(uploaded_file.read())

                    # Get collection name
                    collection_name = DocumentQueries.get_collection_name(
                        user_id, uploaded_file.name
                    )

                    # Load PDF
                    update("Loading PDF pages...", 15)
                    from phase1_fundamentals.ingestion.loader import PDFLoader
                    pages = PDFLoader(str(final_path)).load()

                    # Chunk
                    update("Chunking text...", 30)
                    from phase1_fundamentals.ingestion.chunker import (
                        TokenChunker, Chunk
                    )
                    chunker     = TokenChunker()
                    text_chunks = chunker.chunk_pages(pages)

                    # Embed into user-specific collection
                    update("Embedding into ChromaDB...", 50)
                    from phase1_fundamentals.ingestion.embedder import Embedder
                    embedder = Embedder(
                        collection_name=collection_name
                    )
                    embedder.embed_chunks(text_chunks)

                    # Figures
                    figure_chunk_objs = []
                    figures_json_path = ""

                    if not skip_vision:
                        update("Extracting figures...", 65)
                        from phase4_production.multimodal.figure_extractor \
                            import FigureExtractor
                        from phase4_production.multimodal.vision_chain \
                            import VisionChain

                        extractor = FigureExtractor()
                        figures   = extractor.extract_from_pdf(str(final_path))

                        if figures:
                            update(
                                f"Describing {len(figures)} figures "
                                f"with LLaVA...", 75
                            )
                            stem             = Path(final_path).stem
                            figures_json_path = (
                                f"data/figures/{stem}_figures.json"
                            )
                            Path("data/figures").mkdir(
                                parents=True, exist_ok=True
                            )
                            vision    = VisionChain()
                            described = vision.describe_all(
                                figures, figures_json_path
                            )
                            fig_chunks        = vision.build_text_chunks(described)
                            figure_chunk_objs = [
                                Chunk(
                                    chunk_id        = fc["chunk_id"],
                                    source_file     = fc["source_file"],
                                    page_number     = fc["page_number"],
                                    paragraph_index = fc["paragraph_index"],
                                    text            = fc["text"],
                                    token_count     = fc["token_count"],
                                    char_start      = 0,
                                    char_end        = len(fc["text"])
                                )
                                for fc in fig_chunks
                            ]
                            if figure_chunk_objs:
                                embedder.embed_chunks(figure_chunk_objs)

                    # BM25
                    update("Building BM25 index...", 88)
                    from phase2_production.retrieval.bm25_retriever \
                        import BM25Retriever
                    all_chunks = text_chunks + figure_chunk_objs
                    bm25       = BM25Retriever(
                        index_path=f"data/chunks/bm25_{collection_name}.json"
                    )
                    bm25.build_index(all_chunks)

                    # Update DB record
                    update("Saving metadata...", 95)
                    db3 = get_streamlit_db()
                    DocumentQueries.update_document_stats(
                        db          = db3,
                        document_id = doc_id,
                        page_count  = len(pages),
                        text_chunks = len(text_chunks),
                        figure_chunks = len(figure_chunk_objs),
                        figures_json  = figures_json_path,
                        status        = "ready"
                    )
                    db3.commit()
                    db3.close()

                    progress.progress(100)
                    status.empty()
                    st.success(
                        f"✅ {uploaded_file.name} ingested successfully!"
                    )

                    # Load into session
                    db4     = get_streamlit_db()
                    new_doc = DocumentQueries.get_document_by_id(
                        db4, doc_id, user_id
                    )
                    if new_doc:
                        load_document(new_doc)
                    db4.close()

                    st.balloons()
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    # Mark as failed in DB
                    db5 = get_streamlit_db()
                    doc_fail = DocumentQueries.get_document_by_id(
                        db5, doc_id, user_id
                    )
                    if doc_fail:
                        doc_fail.status = "failed"
                        db5.commit()
                    db5.close()

                    st.error(f"Ingestion failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())