import os
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

# ─────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────
# Model Config
# ─────────────────────────────────────────
LLM_MODEL = "llama-3.1-8b-instant"        # Groq model (fast + free)
EMBEDDING_MODEL = "text-embedding-3-small" # OpenAI embedding model

# ─────────────────────────────────────────
# Chunking Config
# ─────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))      # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100)) # overlap between chunks
TOP_K = int(os.getenv("TOP_K", 5))                   # chunks to retrieve

# ─────────────────────────────────────────
# ChromaDB Config
# ─────────────────────────────────────────
CHROMA_DB_PATH = "data/chroma_db"          # where ChromaDB persists locally
CHROMA_COLLECTION_NAME = "rag_pdf_phase1"

# ─────────────────────────────────────────
# Validation on startup
# ─────────────────────────────────────────
def validate_settings():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please copy .env.example to .env and fill in your keys."
        )
