from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

from config.settings import (
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    validate_settings
)
from phase1_fundamentals.ingestion.chunker import Chunk
from typing import List


class Embedder:
    def __init__(self):
        validate_settings()

        print(f"[Embedder] Loading embedding model {EMBEDDING_MODEL}...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[Embedder] Embedding model loaded")

        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"[Embedder] Connected to ChromaDB at {CHROMA_DB_PATH}")
        print(f"[Embedder] Collection {CHROMA_COLLECTION_NAME} ready")

    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 50) -> None:
        print(f"[Embedder] Embedding {len(chunks)} chunks in batches of {batch_size}...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.embedding_model.embed_documents(texts)

            metadatas = [
                {
                    "chunk_id":        chunk.chunk_id,
                    "source_file":     chunk.source_file,
                    "page_number":     chunk.page_number,
                    "paragraph_index": chunk.paragraph_index,
                    "token_count":     chunk.token_count,
                    "char_start":      chunk.char_start,
                    "char_end":        chunk.char_end,
                }
                for chunk in batch
            ]

            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            print(f"[Embedder] Stored batch {i // batch_size + 1} ({len(batch)} chunks)")

        print(f"[Embedder] All {len(chunks)} chunks embedded and stored")
        print(f"[Embedder] Total vectors in collection: {self.collection.count()}")

    def get_collection_info(self) -> dict:
        return {
            "collection_name": CHROMA_COLLECTION_NAME,
            "total_chunks":    self.collection.count(),
            "db_path":         CHROMA_DB_PATH,
        }

    def clear_collection(self) -> None:
        self.chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("[Embedder] Collection cleared")


if __name__ == "__main__":
    import sys
    from phase1_fundamentals.ingestion.loader import PDFLoader
    from phase1_fundamentals.ingestion.chunker import TokenChunker

    if len(sys.argv) < 2:
        print("Usage: python embedder.py <path_to_pdf>")
        sys.exit(1)

    loader = PDFLoader(sys.argv[1])
    pages = loader.load()

    chunker = TokenChunker()
    chunks = chunker.chunk_pages(pages)

    embedder = Embedder()
    embedder.embed_chunks(chunks)

    info = embedder.get_collection_info()
    print(f"Collection          : {info['collection_name']}")
    print(f"Total chunks stored : {info['total_chunks']}")
    print(f"DB path             : {info['db_path']}")