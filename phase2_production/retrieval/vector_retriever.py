from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from dataclasses import dataclass
from typing import List

from config.settings import (
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    VECTOR_TOP_K,
    validate_settings
)


@dataclass
class VectorResult:
    chunk_id: str
    source_file: str
    page_number: int
    paragraph_index: int
    text: str
    vector_score: float


class VectorRetriever:
    def __init__(self):
        validate_settings()

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_collection(
            name=CHROMA_COLLECTION_NAME
        )

        print(f"[Vector] Connected to ChromaDB — {self.collection.count()} chunks available")

    def retrieve(self, query: str, top_k: int = VECTOR_TOP_K) -> List[VectorResult]:
        # Embed query and find top_k most similar chunks via cosine similarity
        if not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            similarity = 1 - (distance / 2)
            retrieved.append(VectorResult(
                chunk_id=meta["chunk_id"],
                source_file=meta["source_file"],
                page_number=meta["page_number"],
                paragraph_index=meta["paragraph_index"],
                text=doc,
                vector_score=round(similarity, 4)
            ))

        print(f"[Vector] Query: '{query[:60]}...'")
        print(f"[Vector] Retrieved {len(retrieved)} semantic matches")
        return retrieved


if __name__ == "__main__":
    retriever = VectorRetriever()

    query = "How does the underwater image restoration method work?"
    results = retriever.retrieve(query, top_k=3)

    print(f"\n--- Top Vector Results ---")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.vector_score}")
        print(f"    Page: {result.page_number} | Para: {result.paragraph_index}")
        print(f"    Text: {result.text[:200]}...")