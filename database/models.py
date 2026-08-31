from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text,
    DateTime, ForeignKey, JSON, Boolean
)
from sqlalchemy.orm import relationship, declarative_base

# ─────────────────────────────────────────
# Base class
# All models inherit from this
# SQLAlchemy uses it to track all tables
# ─────────────────────────────────────────
Base = declarative_base()


# ─────────────────────────────────────────
# User table
# ─────────────────────────────────────────
class User(Base):
    """
    Stores registered users.

    Why password_hash not password?
      NEVER store plain text passwords
      bcrypt hash: "$2b$12$..." (60 chars)
      Even if DB is leaked → passwords safe
      bcrypt.verify("input", hash) → True/False

    Why username + email both unique?
      Username: for login ("sonal")
      Email: for password reset later
      Both must be unique across all users
    """
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(50),  unique=True, nullable=False)
    email         = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
    is_active     = Column(Boolean, default=True)

    # Relationships
    # Why relationship()?
    #   user.documents → returns all documents for this user
    #   No manual JOIN needed — SQLAlchemy handles it
    #   cascade="all, delete-orphan" → delete user
    #   → auto-delete all their documents too
    documents = relationship(
        "Document",
        back_populates = "user",
        cascade        = "all, delete-orphan"
    )

    def __repr__(self):
        return f"<User id={self.id} username={self.username}>"


# ─────────────────────────────────────────
# Document table
# ─────────────────────────────────────────
class Document(Base):
    """
    Stores ingested PDFs — one row per PDF per user.

    Why collection_name stored here?
      ChromaDB collection = "rag_{user_id}_{pdf_stem}"
      We need to know which collection to load
      when user selects this document

    Why store chunk counts?
      Quick display without hitting ChromaDB
      "moac.pdf | 38 text | 22 figures" → from DB

    Status values:
      "ingesting" → pipeline running
      "ready"     → available for chat
      "failed"    → ingestion failed
    """
    __tablename__ = "documents"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename        = Column(String(255), nullable=False)
    collection_name = Column(String(255), nullable=False)
    # Why unique=True on collection_name?
    # Each document gets its own ChromaDB collection
    # Two users can have same filename but different collections:
    #   User 1 moac.pdf → "rag_1_moac"
    #   User 2 moac.pdf → "rag_2_moac"
    page_count      = Column(Integer, default=0)
    text_chunks     = Column(Integer, default=0)
    figure_chunks   = Column(Integer, default=0)
    total_chunks    = Column(Integer, default=0)
    upload_time     = Column(DateTime, default=datetime.utcnow)
    status          = Column(String(20), default="ingesting")
    figures_json    = Column(String(500), default="")
    # Path to figures JSON file for this document

    # Relationships
    user          = relationship("User", back_populates="documents")
    conversations = relationship(
        "Conversation",
        back_populates = "document",
        cascade        = "all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"<Document id={self.id} "
            f"filename={self.filename} "
            f"user_id={self.user_id}>"
        )


# ─────────────────────────────────────────
# Conversation table
# ─────────────────────────────────────────
class Conversation(Base):
    """
    Stores chat sessions — one row per conversation.

    A user can have multiple conversations
    about the same document:
      moac.pdf → Conversation 1 (about methods)
      moac.pdf → Conversation 2 (about results)
      moac.pdf → Conversation 3 (about figures)

    Why title auto-generated from first question?
      First question: "What is the main method?"
      Title: "What is the main method?" (truncated to 60 chars)
      Shows in document library as conversation preview
      User can identify which conversation is which
    """
    __tablename__ = "conversations"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(Integer, ForeignKey("users.id"),     nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    title       = Column(String(200), default="New conversation")
    # Auto-set to first question when first message saved
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow,
                         onupdate=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates = "conversation",
        cascade        = "all, delete-orphan",
        order_by       = "Message.created_at"
        # order_by ensures messages load in chronological order
    )

    def __repr__(self):
        return (
            f"<Conversation id={self.id} "
            f"title={self.title[:30]} "
            f"doc_id={self.document_id}>"
        )


# ─────────────────────────────────────────
# Message table
# ─────────────────────────────────────────
class Message(Base):
    """
    Stores individual Q&A pairs in a conversation.

    Why role column?
      Same pattern as OpenAI/LangChain:
        role="user"      → question
        role="assistant" → answer
      Makes it easy to reconstruct conversation:
        messages → filter by role → format for LLM context

    Why sources as JSON?
      Sources = list of dicts (page, para, score, is_figure)
      JSON column stores arbitrary structure
      SQLite: stores as TEXT
      PostgreSQL: stores as JSONB (queryable, indexed)

    Why response_ms?
      Analytics: "average response time for Phase 2"
      Comparison: "did Phase 2 get faster over time?"
      Shown in analytics dashboard
    """
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id"), nullable=False
    )
    role            = Column(String(20), nullable=False)
    # "user" or "assistant"
    content         = Column(Text, nullable=False)
    confidence      = Column(String(20), default="")
    # "HIGH", "MEDIUM", "LOW" — only for assistant messages
    sources         = Column(JSON, default=list)
    # List of source dicts — stored as JSON
    pipeline        = Column(String(20), default="phase2")
    # "phase1" or "phase2"
    response_ms     = Column(Integer, default=0)
    # Response time in milliseconds
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Relationship
    conversation = relationship("Message.__class__", back_populates="messages")
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        preview = self.content[:30] if self.content else ""
        return (
            f"<Message id={self.id} "
            f"role={self.role} "
            f"content={preview}>"
        )

    def to_dict(self):
        """
        Convert to dict for Streamlit session state.

        Why needed?
          SQLAlchemy objects can't be stored directly
          in Streamlit session state (not serializable)
          to_dict() gives a plain Python dict
        """
        return {
            "id":          self.id,
            "role":        self.role,
            "content":     self.content,
            "confidence":  self.confidence,
            "sources":     self.sources or [],
            "pipeline":    self.pipeline,
            "response_ms": self.response_ms,
            "created_at":  self.created_at.isoformat()
                           if self.created_at else ""
        }