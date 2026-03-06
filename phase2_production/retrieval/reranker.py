import cohere
from dataclasses import dataclass
from typing import List

from config.settings import (
    COHERE_API_KEY,
    RERANK_MODEL,
    TOP_K,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    validate_phase2_settings
)
from phase2_production.retrieval.hybrid_retriever import HybridResult


# ─────────────────────────────────────────
# Data Model — one re-ranked result
# ─────────────────────────────────────────
@dataclass
class RerankedChunk:
    chunk_id: str
    source_file: str
    page_number: int
    paragraph_index: int
    text: str
    rerank_score: float         # Cohere relevance score 0.0 to 1.0
    rrf_score: float            # original hybrid RRF score
    confidence: str             # "high", "medium", or "low"


def _get_confidence(score: float) -> str:
    """Classify rerank score into confidence level."""
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif score >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"


# ─────────────────────────────────────────
# Cohere Re-ranker
# ─────────────────────────────────────────
class Reranker:
    def __init__(self):
        validate_phase2_settings()

        self.client = cohere.Client(api_key=COHERE_API_KEY)
        self.model = RERANK_MODEL
        print(f"[Reranker] Cohere reranker ready — model: {self.model}")

    def rerank(
        self,
        query: str,
        candidates: List[HybridResult],
        top_k: int = TOP_K
    ) -> List[RerankedChunk]:
        """
        Re-rank hybrid candidates using Cohere's cross-encoder model.

        Unlike vector similarity (query vs chunk independently),
        Cohere evaluates each (query, chunk) PAIR together —
        giving much more accurate relevance scores.

        Steps:
        1. Send query + all candidate texts to Cohere API
        2. Cohere scores each pair from 0.0 to 1.0
        3. Return top_k chunks sorted by Cohere score
        """
        if not candidates:
            print("[Reranker] No candidates to rerank")
            return []

        print(f"[Reranker] Re-ranking {len(candidates)} candidates → top {top_k}...")

        # Extract texts for Cohere
        documents = [candidate.text for candidate in candidates]

        # Call Cohere Rerank API
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=True
        )

        # Build RerankedChunk results
        reranked = []
        for hit in response.results:
            original = candidates[hit.index]
            score = hit.relevance_score

            reranked.append(RerankedChunk(
                chunk_id=original.chunk_id,
                source_file=original.source_file,
                page_number=original.page_number,
                paragraph_index=original.paragraph_index,
                text=original.text,
                rerank_score=round(score, 4),
                rrf_score=original.rrf_score,
                confidence=_get_confidence(score)
            ))

        # Summary
        high = sum(1 for r in reranked if r.confidence == "high")
        medium = sum(1 for r in reranked if r.confidence == "medium")
        low = sum(1 for r in reranked if r.confidence == "low")

        print(f"[Reranker] Done — top {len(reranked)} chunks selected")
        print(f"[Reranker] Confidence: {high} high | {medium} medium | {low} low")

        return reranked


if __name__ == "__main__":
    from phase2_production.retrieval.hybrid_retriever import HybridRetriever

    # Step 1 — Hybrid retrieval
    hybrid = HybridRetriever()
    query = "What method is used for underwater image color correction?"
    candidates = hybrid.retrieve(query, top_k=20)

    # Step 2 — Re-rank
    reranker = Reranker()
    results = reranker.rerank(query, candidates, top_k=5)

    print(f"\n--- Re-ranked Top {len(results)} Chunks ---")
    for i, chunk in enumerate(results, 1):
        print(f"\n[{i}] Cohere Score : {chunk.rerank_score} ({chunk.confidence} confidence)")
        print(f"    RRF Score    : {chunk.rrf_score}")
        print(f"    Page         : {chunk.page_number} | Para: {chunk.paragraph_index}")
        print(f"    Text         : {chunk.text[:200]}...")