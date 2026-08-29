import streamlit as st
import tempfile
import time
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Upload PDF — RAG Assistant",
    page_icon  = "📄",
    layout     = "wide"
)

st.title("📄 Upload & Ingest PDF")
st.markdown(
    "Upload a PDF document to make it searchable. "
    "The system will extract text chunks and figures automatically."
)
st.divider()


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


# ─────────────────────────────────────────
# Helper — step display
# ─────────────────────────────────────────
def show_step(container, icon, label, status="running"):
    """
    Show a step with icon and status.
    status: running | done | error
    """
    colors = {
        "running": "#fcd34d",
        "done":    "#6ee7b7",
        "error":   "#fca5a5"
    }
    icons = {
        "running": "⏳",
        "done":    "✅",
        "error":   "❌"
    }
    color = colors.get(status, "#9ca3af")
    icon_str = icons.get(status, "•")
    container.markdown(
        f"<p style='color:{color};margin:2px 0'>"
        f"{icon_str} {icon} {label}</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────
# Layout — two columns
# ─────────────────────────────────────────
col_upload, col_status = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📂 Select PDF")

    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["pdf"],
        help="Upload any research paper or document (PDF format)"
    )

    if uploaded_file:
        # Show file info
        file_size_mb = round(uploaded_file.size / 1024 / 1024, 2)
        st.markdown(f"""
        <div style='background:#1e2130;border-radius:8px;
                    padding:1rem;margin:1rem 0'>
            <p style='margin:0;color:#9ca3af;font-size:0.85rem'>
                📄 <strong style='color:#e5e7eb'>{uploaded_file.name}</strong>
            </p>
            <p style='margin:4px 0 0 0;color:#6b7280;font-size:0.8rem'>
                Size: {file_size_mb} MB
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Ingestion Options")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        pipeline_choice = st.selectbox(
            "Pipeline",
            ["Phase 2 (Hybrid)", "Phase 1 (Vector)"],
            index=0
        )
    with col_opt2:
        clear_existing = st.checkbox(
            "Clear existing data",
            value=False,
            help="Clear ChromaDB collection before ingesting"
        )

    skip_vision = st.checkbox(
        "Skip figure extraction (faster)",
        value=False,
        help="Skip LLaVA figure description — use if figures already processed"
    )

    st.divider()

    # Ingest button
    ingest_clicked = st.button(
        "🚀 Ingest PDF",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True
    )


# ─────────────────────────────────────────
# Status column — progress display
# ─────────────────────────────────────────
with col_status:
    st.markdown("### 📊 Ingestion Progress")

    status_container = st.container()

    # Show current document info if already ingested
    if st.session_state.ingested_pdf and not ingest_clicked:
        with status_container:
            st.success(
                f"✅ **{st.session_state.ingested_pdf}** is ready!"
            )
            if st.session_state.chunk_stats:
                s = st.session_state.chunk_stats
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pages",   s.get("pages", 0))
                m2.metric("Text",    s.get("text_chunks", 0))
                m3.metric("Figures", s.get("figure_chunks", 0))
                m4.metric("Total",   s.get("total_chunks", 0))

            st.info("👈 Upload a new PDF to replace, or go to Chat page to ask questions")
    elif not ingest_clicked:
        with status_container:
            st.markdown("""
            <div style='background:#1e2130;border-radius:8px;
                        padding:2rem;text-align:center;color:#6b7280'>
                <p style='font-size:2rem;margin:0'>⬆️</p>
                <p>Upload a PDF and click Ingest to begin</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# Ingestion pipeline
# ─────────────────────────────────────────
if ingest_clicked and uploaded_file:

    with col_status:
        st.markdown("### ⚡ Running Pipeline...")

        progress_bar = st.progress(0)
        log_area     = st.empty()
        logs         = []

        def update_log(msg, status="done"):
            """Add message to log display."""
            icons = {"done": "✅", "running": "⏳", "error": "❌"}
            logs.append(f"{icons.get(status, '•')} {msg}")
            log_area.markdown(
                "\n".join(
                    f"<p style='margin:2px 0;font-size:0.85rem;"
                    f"color:#9ca3af'>{l}</p>"
                    for l in logs
                ),
                unsafe_allow_html=True
            )

        try:
            # Initialize lists FIRST before any if/else blocks
            figure_chunks     = []
            figure_chunk_objs = []

            # ── Save uploaded file ──
            update_log("Saving uploaded file...", "running")
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            raw_dir    = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            final_path = raw_dir / uploaded_file.name
            import shutil
            shutil.copy(tmp_path, final_path)
            update_log(f"Saved to data/raw/{uploaded_file.name}")
            progress_bar.progress(10)

            # ── Step 1: Load ──
            update_log("Loading PDF pages...", "running")
            from phase1_fundamentals.ingestion.loader import PDFLoader
            loader = PDFLoader(str(final_path))
            pages  = loader.load()
            update_log(f"Loaded {len(pages)} pages")
            progress_bar.progress(25)

            # ── Step 2: Chunk ──
            update_log("Chunking text (512 tokens, 100 overlap)...", "running")
            from phase1_fundamentals.ingestion.chunker import TokenChunker, Chunk
            chunker     = TokenChunker()
            text_chunks = chunker.chunk_pages(pages)
            update_log(f"Created {len(text_chunks)} text chunks")
            progress_bar.progress(40)

            # ── Step 3: Embed text ──
            update_log("Embedding chunks → ChromaDB...", "running")
            from phase1_fundamentals.ingestion.embedder import Embedder
            embedder = Embedder()
            if clear_existing:
                embedder.clear_collection()
                update_log("Cleared existing ChromaDB collection")
            embedder.embed_chunks(text_chunks)
            update_log(f"Stored {len(text_chunks)} vectors in ChromaDB")
            progress_bar.progress(55)

            # ── Step 4: Figures ──
            update_log("Building BM25 keyword index...", "running")
            from phase2_production.retrieval.bm25_retriever import BM25Retriever
            bm25 = BM25Retriever()

            if not skip_vision:
                update_log("Extracting figures from PDF...", "running")
                from phase4_production.multimodal.figure_extractor import FigureExtractor
                from phase4_production.multimodal.vision_chain import VisionChain

                extractor = FigureExtractor()
                figures   = extractor.extract_from_pdf(str(final_path))

                if figures:
                    update_log(
                        f"Found {len(figures)} figures — "
                        f"generating LLaVA descriptions...", "running"
                    )
                    stem         = Path(final_path).stem
                    figures_json = f"data/figures/{stem}_figures.json"
                    Path("data/figures").mkdir(parents=True, exist_ok=True)

                    vision    = VisionChain()
                    described = vision.describe_all(figures, figures_json)
                    
                    update_log(f"Described {len(described)} figures with LLaVA")
                    progress_bar.progress(75)

                    # Build figure text chunks
                    figure_chunks = vision.build_text_chunks(described)
                    update_log(f"Built {len(figure_chunks)} figure text chunks")

                    # Convert to Chunk objects
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
                        for fc in figure_chunks
                    ]

                    # Embed figure chunks into ChromaDB
                    if figure_chunk_objs:
                        embedder.embed_chunks(figure_chunk_objs)
                        update_log(
                            f"Embedded {len(figure_chunk_objs)} "
                            f"figure chunks into ChromaDB"
                        )
                else:
                    update_log("No figures found in PDF")
            else:
                update_log("Skipping figure extraction (fast mode)")

            progress_bar.progress(85)

            # ── Step 5: BM25 on ALL chunks ──
            all_chunks = text_chunks + figure_chunk_objs
            bm25.build_index(all_chunks)
            update_log(f"BM25 index built ({len(all_chunks)} total chunks)")
            progress_bar.progress(100)

            # ── Summary ──
            stats = {
                "pages":         len(pages),
                "text_chunks":   len(text_chunks),
                "figure_chunks": len(figure_chunk_objs),
                "total_chunks":  len(all_chunks),
                "pdf_name":      uploaded_file.name,
            }

            st.session_state.ingested_pdf = uploaded_file.name
            st.session_state.chunk_stats  = stats
            st.session_state.chat_history = []

            st.success("🎉 Ingestion complete!")
            st.divider()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📄 Pages",   stats["pages"])
            m2.metric("📝 Text",    stats["text_chunks"])
            m3.metric("🖼️ Figures", stats["figure_chunks"])
            m4.metric("🔢 Total",   stats["total_chunks"])

            st.info("👉 Go to the **Chat** page to ask questions!")
            os.unlink(tmp_path)

        except Exception as e:
            update_log(f"Error: {e}", "error")
            st.error(f"Ingestion failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# ─────────────────────────────────────────
# Tips section
# ─────────────────────────────────────────
st.divider()
st.markdown("### 💡 Tips")

t1, t2, t3 = st.columns(3)
with t1:
    st.markdown("""
    **Best PDF types:**
    - Research papers
    - Technical documents
    - Reports with figures
    - Academic papers
    """)
with t2:
    st.markdown("""
    **Ingestion time:**
    - Text only: ~30 seconds
    - With figures: ~3-5 minutes
    - (LLaVA describes each figure)
    - Use "Skip figures" for speed
    """)
with t3:
    st.markdown("""
    **After ingestion:**
    - Go to Chat page
    - Ask any question
    - See page + figure citations
    - Switch Phase 1 vs Phase 2
    """)