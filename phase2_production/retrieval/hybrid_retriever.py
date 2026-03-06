from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from config.settings import (
    HYBRID_TOP_K,
    BM25_TOP_K,
    VECTOR_TOP_K,
    validate_settings
)
from phase2_production.retrieval.bm25_retriever import BM25Retriever, BM25Result
from phase2_production.retrieval.vector_retriever import VectorRetriever, VectorResult


# ─────────────────────────────────────────
# Data Model — one fused hybrid result
# ─────────────────────────────────────────
@dataclass
class HybridResult:
    chunk_id: str
    source_file: str
    page_number: int
    paragraph_index: int
    text: str
    rrf_score: float        # combined RRF score — higher is better
    bm25_rank: int          # original BM25 rank (0 = not found by BM25)
    vector_rank: int        # original vector rank (0 = not found by vector)
    bm25_score: float       # original BM25 score
    vector_score: float     # original vector score


# ─────────────────────────────────────────
# RRF (Reciprocal Rank Fusion) Formula
# ─────────────────────────────────────────
def reciprocal_rank_fusion(
    bm25_results: List[BM25Result],
    vector_results: List[VectorResult],
    k: int = 60             # RRF constant — 60 is standard
) -> Dict[str, float]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    RRF formula: score(doc) = sum(1 / (k + rank))

    A document ranked #1 in BM25 and #3 in vector gets:
    score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

    A document only in vector at rank #1 gets:
    score = 1/(60+1) = 0.0164

    Documents appearing in BOTH lists get boosted scores.
    """
    rrf_scores = {}

    # Add BM25 scores
    for rank, result in enumerate(bm25_results, start=1):
        cid = result.chunk_id
        if cid not in rrf_scores:
            rrf_scores[cid] = 0.0
        rrf_scores[cid] += 1.0 / (k + rank)

    # Add vector scores
    for rank, result in enumerate(vector_results, start=1):
        cid = result.chunk_id
        if cid not in rrf_scores:
            rrf_scores[cid] = 0.0
        rrf_scores[cid] += 1.0 / (k + rank)

    return rrf_scores


# ─────────────────────────────────────────
# Hybrid Retriever
# ─────────────────────────────────────────
class HybridRetriever:
    def __init__(self):
        validate_settings()

        # Initialize both retrievers
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever()

        # Try loading BM25 index from disk
        loaded = self.bm25_retriever.load_index()
        if not loaded:
            print("[Hybrid] WARNING: BM25 index not found.")
            print("[Hybrid] Run ingestion first to build BM25 index.")

        print("[Hybrid] Hybrid retriever ready")

    def retrieve(self, query: str, top_k: int = HYBRID_TOP_K) -> List[HybridResult]:
        """
        Run BM25 + vector search in parallel, fuse with RRF,
        return top_k candidates sorted by combined score.
        """
        print(f"\n[Hybrid] Running hybrid retrieval for: '{query[:60]}...'")

        # Step 1 — Run both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k=BM25_TOP_K)
        vector_results = self.vector_retriever.retrieve(query, top_k=VECTOR_TOP_K)

        # Step 2 — Build lookup maps for metadata
        bm25_map = {r.chunk_id: r for r in bm25_results}
        vector_map = {r.chunk_id: r for r in vector_results}

        # Step 3 — RRF fusion
        rrf_scores = reciprocal_rank_fusion(bm25_results, vector_results)

        # Step 4 — Build rank lookup for metadata
        bm25_ranks = {r.chunk_id: rank for rank, r in enumerate(bm25_results, 1)}
        vector_ranks = {r.chunk_id: rank for rank, r in enumerate(vector_results, 1)}

        # Step 5 — Build HybridResult objects
        all_chunk_ids = set(rrf_scores.keys())
        hybrid_results = []

        for chunk_id in all_chunk_ids:
            # Get metadata from whichever retriever found this chunk
            if chunk_id in bm25_map:
                meta = bm25_map[chunk_id]
            else:
                meta = vector_map[chunk_id]

            hybrid_results.append(HybridResult(
                chunk_id=chunk_id,
                source_file=meta.source_file,
                page_number=meta.page_number,
                paragraph_index=meta.paragraph_index,
                text=meta.text,
                rrf_score=round(rrf_scores[chunk_id], 6),
                bm25_rank=bm25_ranks.get(chunk_id, 0),
                vector_rank=vector_ranks.get(chunk_id, 0),
                bm25_score=bm25_map[chunk_id].bm25_score if chunk_id in bm25_map else 0.0,
                vector_score=vector_map[chunk_id].vector_score if chunk_id in vector_map else 0.0,
            ))

        # Step 6 — Sort by RRF score and return top_k
        hybrid_results.sort(key=lambda x: x.rrf_score, reverse=True)
        top_results = hybrid_results[:top_k]

        # Summary stats
        both_found = sum(1 for r in top_results if r.bm25_rank > 0 and r.vector_rank > 0)
        print(f"[Hybrid] BM25 found: {len(bm25_results)} | "
              f"Vector found: {len(vector_results)} | "
              f"After fusion: {len(hybrid_results)}")
        print(f"[Hybrid] Chunks found by BOTH methods: {both_found}/{len(top_results)}")
        print(f"[Hybrid] Returning top {len(top_results)} candidates for re-ranking")

        return top_results


if __name__ == "__main__":
    retriever = HybridRetriever()

    query = "What method is used for underwater image color correction?"
    results = retriever.retrieve(query, top_k=5)

    print(f"\n--- Top Hybrid Results (RRF Fused) ---")
    for i, result in enumerate(results, 1):
        found_by = []
        if result.bm25_rank > 0:
            found_by.append(f"BM25 #{result.bm25_rank}")
        if result.vector_rank > 0:
            found_by.append(f"Vector #{result.vector_rank}")
        print(f"\n[{i}] RRF Score: {result.rrf_score} | Found by: {', '.join(found_by)}")
        print(f"    Page: {result.page_number} | Para: {result.paragraph_index}")
        print(f"    Text: {result.text[:200]}...")