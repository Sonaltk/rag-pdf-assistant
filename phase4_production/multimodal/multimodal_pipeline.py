import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from phase1_fundamentals.ingestion.loader import PDFLoader
from phase1_fundamentals.ingestion.chunker import TokenChunker, Chunk
from phase1_fundamentals.ingestion.embedder import Embedder
from phase2_production.retrieval.bm25_retriever import BM25Retriever
from phase4_production.multimodal.figure_extractor import FigureExtractor
from phase4_production.multimodal.vision_chain import VisionChain


# ─────────────────────────────────────────
# Multimodal Result — extends normal retrieval
# ─────────────────────────────────────────
@dataclass
class MultimodalResult:
    """
    A retrieved result that can be either text or figure.

    Why extend rather than modify Phase 2 results?
      Open/Closed principle — extend, don't modify
      Phase 2 pipeline still works unchanged
      Multimodal is an optional upgrade layer
      Easy to disable: just use Phase 2 directly

    Fields:
      is_figure   → True if this result is a figure chunk
      image_path  → path to actual image (if is_figure)
      figure_id   → e.g. "moac_p3_f1" (if is_figure)
      All other fields same as normal retrieved chunk
    """
    chunk_id:        str
    source_file:     str
    page_number:     int
    paragraph_index: int
    text:            str
    similarity_score: float
    is_figure:       bool  = False
    image_path:      str   = ""
    figure_id:       str   = ""


# ─────────────────────────────────────────
# Multimodal Ingestor
# ─────────────────────────────────────────
class MultimodalIngestor:
    """
    Extends Phase 2 ingestion to include figure chunks.

    Normal Phase 2 ingest:
      PDF → text chunks → embeddings → ChromaDB
      PDF → text chunks → BM25 index

    Multimodal ingest adds:
      PDF → figures extracted
      figures → LLaVA descriptions
      descriptions → figure chunks
      figure chunks → embeddings → ChromaDB (same collection!)
      figure chunks → BM25 index (same index!)

    Why same ChromaDB collection?
      Unified search — one query searches both
      text and figure chunks simultaneously
      No separate "figure index" to maintain
      Simpler retrieval logic
    """

    def __init__(self):
        self.loader    = PDFLoader
        self.chunker   = TokenChunker()
        self.embedder  = Embedder()
        self.bm25      = BM25Retriever()
        self.extractor = FigureExtractor()
        self.vision    = VisionChain()
        print("[MultimodalIngestor] Ready")

    def _figures_already_processed(self, pdf_path: str) -> bool:
        """
        Check if figures were already extracted and described.

        Why this check?
          Optimization — skip re-extraction if already done
          figure_extractor saves JSON with descriptions
          If all figures have descriptions → skip LLaVA

        This is part of the caching strategy:
          First run:  extract + describe (~3-5 mins)
          Later runs: load from JSON (instant)
        """
        stem         = Path(pdf_path).stem
        figures_json = f"data/figures/{stem}_figures.json"

        if not Path(figures_json).exists():
            return False

        with open(figures_json) as f:
            figures = json.load(f)

        # All figures must have non-null descriptions
        all_described = all(
            fig.get("description") for fig in figures
        )
        return all_described

    def _load_figure_chunks(self, pdf_path: str) -> List[dict]:
        """
        Load figure descriptions and convert to chunk format.

        Why load from JSON rather than re-run?
          LLaVA takes 20-40s per figure
          Re-running on every ingest = very slow
          JSON is the cache — use it
        """
        stem         = Path(pdf_path).stem
        figures_json = f"data/figures/{stem}_figures.json"

        with open(figures_json) as f:
            figures_data = json.load(f)

        # Re-create ExtractedFigure objects
        from phase4_production.multimodal.figure_extractor import ExtractedFigure
        figures = [ExtractedFigure(**fig) for fig in figures_data]

        # Convert to text chunks
        return self.vision.build_text_chunks(figures)

    def ingest(
        self,
        pdf_path:    str,
        clear:       bool = False,
        skip_vision: bool = False
    ) -> Dict:
        """
        Full multimodal ingestion pipeline.

        Steps:
          1. Load PDF → pages (same as Phase 1/2)
          2. Chunk pages → text chunks (same as Phase 1/2)
          3. Extract figures → image files (NEW)
          4. Describe figures → LLaVA descriptions (NEW)
          5. Build figure chunks → text (NEW)
          6. Embed ALL chunks (text + figure) → ChromaDB
          7. Build BM25 index on ALL chunks

        Args:
          pdf_path:    path to PDF
          clear:       clear existing ChromaDB collection
          skip_vision: skip LLaVA if descriptions exist
                       useful for re-ingesting without
                       waiting for LLaVA again

        Returns:
          summary dict with counts
        """
        print("\n" + "=" * 60)
        print("MULTIMODAL INGESTION PIPELINE")
        print("=" * 60)

        # ── Step 1: Load PDF ──
        print("\nStep 1/5 — Loading PDF...")
        loader = PDFLoader(pdf_path)
        pages  = loader.load()
        print(f"  Loaded {len(pages)} pages")

        # ── Step 2: Chunk text ──
        print("\nStep 2/5 — Chunking text...")
        text_chunks = self.chunker.chunk_pages(pages)
        print(f"  Created {len(text_chunks)} text chunks")

        # ── Step 3 & 4: Extract + describe figures ──
        print("\nStep 3/5 — Processing figures...")
        stem         = Path(pdf_path).stem
        figures_json = f"data/figures/{stem}_figures.json"

        if self._figures_already_processed(pdf_path) or skip_vision:
            print("  Figures already described — loading from cache")
            figure_chunks = self._load_figure_chunks(pdf_path)
        else:
            print("  Extracting figures...")
            figures = self.extractor.extract_from_pdf(pdf_path)

            if figures:
                print(f"  Extracted {len(figures)} figures")
                print("  Generating LLaVA descriptions...")
                described     = self.vision.describe_all(figures, figures_json)
                figure_chunks = self.vision.build_text_chunks(described)
            else:
                print("  No figures found in PDF")
                figure_chunks = []

        print(f"  Figure chunks ready: {len(figure_chunks)}")

        # ── Step 5: Embed ALL chunks ──
        print("\nStep 4/5 — Embedding text + figure chunks...")

        if clear:
            self.embedder.clear_collection()

        # Embed text chunks (normal Phase 2 behaviour)
        self.embedder.embed_chunks(text_chunks)

        # Embed figure chunks
        # Convert figure chunk dicts → Chunk objects
        if figure_chunks:
            figure_chunk_objects = [
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
            self.embedder.embed_chunks(figure_chunk_objects)
            print(f"  Embedded {len(figure_chunk_objects)} figure chunks")

        # ── Step 6: Build BM25 index ──
        print("\nStep 5/5 — Building BM25 index...")
        all_chunks = text_chunks + (
            figure_chunk_objects if figure_chunks else []
        )
        self.bm25.build_index(all_chunks)

        # Summary
        total = len(text_chunks) + len(figure_chunks)
        info  = self.embedder.get_collection_info()

        summary = {
            "pdf":           pdf_path,
            "pages":         len(pages),
            "text_chunks":   len(text_chunks),
            "figure_chunks": len(figure_chunks),
            "total_chunks":  total,
            "vectors":       info["total_chunks"]
        }

        print("\n" + "=" * 60)
        print("MULTIMODAL INGESTION COMPLETE")
        print(f"  Text chunks   : {summary['text_chunks']}")
        print(f"  Figure chunks : {summary['figure_chunks']}")
        print(f"  Total vectors : {summary['vectors']}")
        print("=" * 60)

        return summary


# ─────────────────────────────────────────
# Multimodal Retriever
# ─────────────────────────────────────────
class MultimodalRetriever:
    """
    Wraps Phase 2 retrieval to handle figure chunks.

    What it adds over Phase 2 retriever:
      Detects figure chunks in results (paragraph_index=999)
      Enriches them with image_path from figures JSON
      Returns MultimodalResult objects instead of plain chunks

    Why detect by paragraph_index=999?
      ChromaDB stores metadata per chunk
      paragraph_index=999 is our figure marker
      No separate index or lookup needed
      Consistent with how we stored figure chunks
    """

    def __init__(self, pdf_path: str, collection_name: str = None):
        """
        Args:
            pdf_path: needed to load figure metadata
                      so we can return image paths
        """
        from phase2_production.generation.graph import RAGGraph
        self.graph    = RAGGraph(collection_name=collection_name)
        self.pdf_path = pdf_path

        # Load figure metadata for image path lookup
        stem              = Path(pdf_path).stem
        figures_json_path = f"data/figures/{stem}_figures.json"
        self.figure_map   = {}   # figure_id → image_path

        if Path(figures_json_path).exists():
            with open(figures_json_path) as f:
                figures = json.load(f)
            self.figure_map = {
                fig["figure_id"]: fig["image_path"]
                for fig in figures
            }
            print(f"[MultimodalRetriever] Loaded {len(self.figure_map)} figure paths")

        print("[MultimodalRetriever] Ready")

    def ask(self, question: str) -> Dict:
        """
        Run Phase 2 pipeline and enrich results with figure metadata.

        Flow:
          question → Phase 2 RAGGraph.ask()
          → response with sources
          → detect figure chunks in sources
          → add image_path to figure chunks
          → return enriched response

        Why run Phase 2 as-is?
          Phase 2 already handles hybrid retrieval + reranking
          Figure chunks are in ChromaDB → retrieved naturally
          No special figure-specific retrieval needed
          Multimodal "just works" because figures are chunks
        """
        # Run Phase 2 pipeline
        response = self.graph.ask(question)

        # Enrich sources with figure metadata
        enriched_sources = []
        figure_sources   = []

        for source in response.get("sources", []):
            # Detect figure chunk by paragraph_index
            is_figure = getattr(source, "paragraph_index", 0) == 999

            if is_figure:
                # Extract figure_id from chunk_id
                # chunk_id format: "fig_moac_p3_f1"
                figure_id  = source.chunk_id.replace("fig_", "")
                image_path = self.figure_map.get(figure_id, "")

                result = MultimodalResult(
                    chunk_id         = source.chunk_id,
                    source_file      = source.source_file,
                    page_number      = source.page_number,
                    paragraph_index  = source.paragraph_index,
                    text             = source.text,
                    similarity_score = getattr(source, "rerank_score",
                                       getattr(source, "similarity_score", 0.0)),
                    is_figure        = True,
                    image_path       = image_path,
                    figure_id        = figure_id
                )
                figure_sources.append(result)
            else:
                result = MultimodalResult(
                    chunk_id         = source.chunk_id,
                    source_file      = source.source_file,
                    page_number      = source.page_number,
                    paragraph_index  = source.paragraph_index,
                    text             = source.text,
                    similarity_score = getattr(source, "rerank_score",
                                       getattr(source, "similarity_score", 0.0)),
                    is_figure        = False
                )
            enriched_sources.append(result)

        # Add figure info to response
        response["sources"]        = enriched_sources
        response["figure_sources"] = figure_sources
        response["has_figures"]    = len(figure_sources) > 0

        return response


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/moac.pdf"
    action   = sys.argv[2] if len(sys.argv) > 2 else "ingest"

    if action == "ingest":
        ingestor = MultimodalIngestor()
        summary  = ingestor.ingest(pdf_path)
        print(f"\nSummary: {summary}")

    elif action == "ask":
        question  = sys.argv[3] if len(sys.argv) > 3 else \
                    "What does the pipeline diagram show?"
        retriever = MultimodalRetriever(pdf_path)
        response  = retriever.ask(question)

        print(f"\nQuestion: {question}")
        print(f"Answer: {response['answer'][:1000]}...")
        print(f"\nSources ({len(response['sources'])}):")
        for src in response["sources"]:
            if src.is_figure:
                print(f"  [FIGURE] Page {src.page_number} | {src.image_path}")
            else:
                print(f"  [TEXT]   Page {src.page_number} | Para {src.paragraph_index}")

        if response["has_figures"]:
            print(f"\nFigures cited: {len(response['figure_sources'])}")