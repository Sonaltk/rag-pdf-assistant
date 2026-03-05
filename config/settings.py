import os
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

# ─────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")      # not used but kept for reference
COHERE_API_KEY = os.getenv("COHERE_API_KEY")       # Phase 2 — re-ranking

# ─────────────────────────────────────────
# Model Config
# ─────────────────────────────────────────
LLM_MODEL = "llama-3.1-8b-instant"                # Groq LLM (free)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"              # HuggingFace local embeddings (free)
RERANK_MODEL = "rerank-english-v3.0"              # Cohere re-ranking model

# ─────────────────────────────────────────
# Chunking Config
# ─────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# ─────────────────────────────────────────
# Retrieval Config
# ─────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 5))                # final chunks sent to LLM
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", 20)) # candidates before re-ranking
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 20))     # BM25 candidates
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", 20)) # vector search candidates

# ─────────────────────────────────────────
# ChromaDB Config
# ─────────────────────────────────────────
CHROMA_DB_PATH = "data/chroma_db"
CHROMA_COLLECTION_NAME = "rag_pdf_phase1"          # reuse Phase 1 collection

# ─────────────────────────────────────────
# BM25 Config
# ─────────────────────────────────────────
BM25_INDEX_PATH = "data/chunks/bm25_index.json"   # persisted BM25 index

# ─────────────────────────────────────────
# Confidence Thresholds
# ─────────────────────────────────────────
HIGH_CONFIDENCE_THRESHOLD = 0.7   # rerank score above this = high confidence
LOW_CONFIDENCE_THRESHOLD = 0.4    # rerank score below this = flag answer

# ─────────────────────────────────────────
# Validation on startup
# ─────────────────────────────────────────
def validate_settings():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please copy .env.example to .env and fill in your keys."
        )

def validate_phase2_settings():
    """Additional validation for Phase 2 specific keys."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not COHERE_API_KEY:
        missing.append("COHERE_API_KEY")
    if missing:
        raise ValueError(
            f"Missing Phase 2 environment variables: {', '.join(missing)}\n"
            f"Please add them to your .env file."
        )