from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from dataclasses import dataclass
from typing import List

from config.settings import (
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    TOP_K,
    validate_settings
)


# ─────────────────────────────────────────
# Data Model — one retrieved result
# ─────────────────────────────────────────
@dataclass
class RetrievedChunk:
    chunk_id: str
    source_file: str
    page_number: int
    paragraph_index: int
    text: str
    similarity_score: float     # 0.0 to 1.0 — higher is more relevant


# ─────────────────────────────────────────
# Retriever — searches ChromaDB with a query
# ─────────────────────────────────────────
class Retriever:
    def __init__(self):
        validate_settings()

        # Same embedding model used during ingestion — must match
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Connect to existing ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_collection(
            name=CHROMA_COLLECTION_NAME
        )

        print(f"[Retriever] Connected to collection with "
              f"{self.collection.count()} chunks")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
        """
        Embed the query and find the top_k most similar chunks.
        Returns a list of RetrievedChunk sorted by relevance (best first).
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Embed the query using same model as ingestion
        query_embedding = self.embedding_model.embed_query(query)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Parse results into RetrievedChunk objects
        retrieved = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score (1=identical, 0=opposite)
            similarity = 1 - (distance / 2)

            retrieved.append(RetrievedChunk(
                chunk_id=meta["chunk_id"],
                source_file=meta["source_file"],
                page_number=meta["page_number"],
                paragraph_index=meta["paragraph_index"],
                text=doc,
                similarity_score=round(similarity, 4)
            ))

        print(f"[Retriever] Query: {query[:60]}...")
        print(f"[Retriever] Retrieved {len(retrieved)} chunks")
        return retrieved

    def format_for_display(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks for readable display.
        """
        lines = []
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{i}] Source: {chunk.source_file} | "
                f"Page: {chunk.page_number} | "
                f"Para: {chunk.paragraph_index} | "
                f"Score: {chunk.similarity_score}"
            )
            #lines.append(f"    {chunk.text[:200]}...")
            lines.append(f"    {chunk.text}")
            lines.append("")
        return " ".join(lines)


# ─────────────────────────────────────────
# Quick test — run directly to verify
# python phase1_fundamentals/retrieval/retriever.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    retriever = Retriever()

    # Test query — change this to something relevant to your PDF
    query = "Explain the introduction section of the document."
    results = retriever.retrieve(query, top_k=3)

    print("--- Top Retrieved Chunks ---")
    print(retriever.format_for_display(results))