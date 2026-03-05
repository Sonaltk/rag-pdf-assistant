import json
import os
from dataclasses import dataclass
from typing import List
from rank_bm25 import BM25Okapi

from config.settings import BM25_INDEX_PATH, BM25_TOP_K


# ─────────────────────────────────────────
# Data Model — one BM25 result
# ─────────────────────────────────────────
@dataclass
class BM25Result:
    chunk_id: str
    source_file: str
    page_number: int
    paragraph_index: int
    text: str
    bm25_score: float           # raw BM25 score — higher is more relevant


# ─────────────────────────────────────────
# BM25 Retriever
# ─────────────────────────────────────────
class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks_metadata = []   # stores chunk metadata parallel to BM25 index
        self.is_loaded = False

    def build_index(self, chunks: list) -> None:
        """
        Build BM25 index from a list of Chunk objects.
        Tokenizes each chunk by whitespace for BM25.
        Saves index to disk for reuse.
        """
        print(f"[BM25] Building index from {len(chunks)} chunks...")

        # Tokenize each chunk — BM25 works on token lists
        tokenized_chunks = [chunk.text.lower().split() for chunk in chunks]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_chunks)

        # Store metadata for each chunk (parallel to BM25 index)
        self.chunks_metadata = [
            {
                "chunk_id":        chunk.chunk_id,
                "source_file":     chunk.source_file,
                "page_number":     chunk.page_number,
                "paragraph_index": chunk.paragraph_index,
                "text":            chunk.text,
            }
            for chunk in chunks
        ]

        # Save to disk
        self._save_index()
        self.is_loaded = True
        print(f"[BM25] Index built and saved to {BM25_INDEX_PATH}")

    def _save_index(self) -> None:
        """Save chunk metadata to disk so we don't rebuild every time."""
        os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
        with open(BM25_INDEX_PATH, "w") as f:
            json.dump(self.chunks_metadata, f, indent=2)

    def load_index(self) -> bool:
        """
        Load chunk metadata from disk and rebuild BM25 index.
        Returns True if successful, False if index doesn't exist.
        """
        if not os.path.exists(BM25_INDEX_PATH):
            print(f"[BM25] No saved index found at {BM25_INDEX_PATH}")
            print(f"[BM25] Run build_index() first")
            return False

        with open(BM25_INDEX_PATH, "r") as f:
            self.chunks_metadata = json.load(f)

        # Rebuild BM25 from saved text
        tokenized_chunks = [
            meta["text"].lower().split()
            for meta in self.chunks_metadata
        ]
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.is_loaded = True
        print(f"[BM25] Index loaded — {len(self.chunks_metadata)} chunks")
        return True

    def retrieve(self, query: str, top_k: int = BM25_TOP_K) -> List[BM25Result]:
        """
        Search chunks using BM25 keyword matching.
        Returns top_k results sorted by BM25 score (highest first).
        """
        if not self.is_loaded:
            raise RuntimeError(
                "BM25 index not loaded. Call build_index() or load_index() first."
            )

        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Tokenize query same way as chunks
        tokenized_query = query.lower().split()

        # Get BM25 scores for all chunks
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Build results
        results = []
        for idx in top_indices:
            meta = self.chunks_metadata[idx]
            score = float(scores[idx])

            # Skip zero-score chunks (no keyword overlap at all)
            if score == 0.0:
                continue

            results.append(BM25Result(
                chunk_id=meta["chunk_id"],
                source_file=meta["source_file"],
                page_number=meta["page_number"],
                paragraph_index=meta["paragraph_index"],
                text=meta["text"],
                bm25_score=round(score, 4)
            ))

        print(f"[BM25] Query: '{query[:60]}...'")
        print(f"[BM25] Found {len(results)} keyword matches")
        return results


# ─────────────────────────────────────────
# Quick test — run directly to verify
# python phase2_production/retrieval/bm25_retriever.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from phase1_fundamentals.ingestion.loader import PDFLoader
    from phase1_fundamentals.ingestion.chunker import TokenChunker

    if len(sys.argv) < 2:
        print("Usage: python bm25_retriever.py <path_to_pdf>")
        sys.exit(1)

    # Load and chunk PDF
    loader = PDFLoader(sys.argv[1])
    pages = loader.load()
    chunker = TokenChunker()
    chunks = chunker.chunk_pages(pages)

    # Build BM25 index
    retriever = BM25Retriever()
    retriever.build_index(chunks)

    # Test query
    query = "optical attenuation compensation underwater"
    results = retriever.retrieve(query, top_k=3)

    print(f"\n--- Top BM25 Results ---")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.bm25_score}")
        print(f"    Page: {result.page_number} | Para: {result.paragraph_index}")
        print(f"    Text: {result.text[:200]}...")