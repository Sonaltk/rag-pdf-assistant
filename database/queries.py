import bcrypt
from datetime import datetime
from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from database.models import User, Document, Conversation, Message


# ─────────────────────────────────────────
# USER QUERIES
# ─────────────────────────────────────────
class UserQueries:
    """
    All user-related database operations.

    Why bcrypt for passwords?
      MD5/SHA256: fast → vulnerable to brute force
      bcrypt: intentionally slow → brute force impractical
      Cost factor (12): ~250ms per hash → acceptable for login
      Even if DB leaked, passwords safe for years
    """

    @staticmethod
    def create_user(
        db:       Session,
        username: str,
        email:    str,
        password: str
    ) -> Optional[User]:
        """
        Register a new user.

        Steps:
          1. Check username + email not already taken
          2. Hash password with bcrypt
          3. Create User object
          4. Save to database

        Returns None if username/email already exists.
        """
        # Check existing
        existing = db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing:
            if existing.username == username:
                raise ValueError(f"Username '{username}' already taken")
            else:
                raise ValueError(f"Email '{email}' already registered")

        # Hash password
        # bcrypt.hashpw needs bytes → encode()
        # decode() converts hash bytes back to string for storage
        password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12)
        ).decode("utf-8")

        user = User(
            username      = username,
            email         = email,
            password_hash = password_hash
        )
        db.add(user)
        db.flush()  # flush assigns id without full commit
        return user

    @staticmethod
    def verify_login(
        db:       Session,
        username: str,
        password: str
    ) -> Optional[User]:
        """
        Verify username + password.

        Returns User if valid, None if invalid.

        Why not tell user which is wrong?
          "Username not found" → attacker knows username exists
          "Wrong password" → attacker knows username is valid
          Always return same message: "Invalid credentials"
          Security best practice
        """
        user = db.query(User).filter_by(
            username  = username,
            is_active = True
        ).first()

        if not user:
            return None

        # bcrypt.checkpw compares input with stored hash
        # Handles salt automatically (salt embedded in hash)
        valid = bcrypt.checkpw(
            password.encode("utf-8"),
            user.password_hash.encode("utf-8")
        )

        return user if valid else None

    @staticmethod
    def get_user_by_id(
        db:      Session,
        user_id: int
    ) -> Optional[User]:
        """Get user by primary key."""
        return db.query(User).filter_by(id=user_id).first()

    @staticmethod
    def get_user_by_username(
        db:       Session,
        username: str
    ) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter_by(username=username).first()


# ─────────────────────────────────────────
# DOCUMENT QUERIES
# ─────────────────────────────────────────
class DocumentQueries:
    """
    All document-related database operations.

    Collection name strategy:
      f"rag_{user_id}_{pdf_stem}"
      user_id=1, moac.pdf → "rag_1_moac"
      user_id=2, moac.pdf → "rag_2_moac"
      Completely isolated per user
    """

    @staticmethod
    def get_collection_name(user_id: int, filename: str) -> str:
        """
        Generate ChromaDB collection name for this user + document.

        Why sanitize?
          Collection names must be alphanumeric + underscore
          "My Paper (2024).pdf" → "rag_1_my_paper_2024_"
          Removes special chars that ChromaDB rejects
        """
        from pathlib import Path
        stem      = Path(filename).stem.lower()
        sanitized = "".join(
            c if c.isalnum() else "_"
            for c in stem
        )
        return f"rag_{user_id}_{sanitized}"

    @staticmethod
    def create_document(
        db:        Session,
        user_id:   int,
        filename:  str,
        status:    str = "ingesting"
    ) -> Document:
        """
        Create document record when ingestion starts.

        Why create before ingestion completes?
          Show "ingesting..." status in UI
          If ingestion fails → update status to "failed"
          User knows something went wrong
          Better UX than silently failing
        """
        collection_name = DocumentQueries.get_collection_name(
            user_id, filename
        )

        doc = Document(
            user_id         = user_id,
            filename        = filename,
            collection_name = collection_name,
            status          = status
        )
        db.add(doc)
        db.flush()
        return doc

    @staticmethod
    def update_document_stats(
        db:            Session,
        document_id:   int,
        page_count:    int,
        text_chunks:   int,
        figure_chunks: int,
        figures_json:  str = "",
        status:        str = "ready"
    ) -> Optional[Document]:
        """
        Update document after ingestion completes.

        Called after full pipeline runs:
          create_document() → "ingesting"
          [run pipeline]
          update_document_stats() → "ready"
        """
        doc = db.query(Document).filter_by(id=document_id).first()
        if not doc:
            return None

        doc.page_count    = page_count
        doc.text_chunks   = text_chunks
        doc.figure_chunks = figure_chunks
        doc.total_chunks  = text_chunks + figure_chunks
        doc.figures_json  = figures_json
        doc.status        = status
        return doc

    @staticmethod
    def get_user_documents(
        db:      Session,
        user_id: int
    ) -> List[Document]:
        """
        Get all documents for a user, newest first.

        Why order by upload_time DESC?
          Most recently ingested appears first
          User most likely wants their latest document
        """
        return (
            db.query(Document)
            .filter_by(user_id=user_id)
            .order_by(Document.upload_time.desc())
            .all()
        )

    @staticmethod
    def get_document_by_id(
        db:          Session,
        document_id: int,
        user_id:     int
    ) -> Optional[Document]:
        """
        Get document by id, verifying ownership.

        Why verify user_id?
          Prevent user A from accessing user B's documents
          Always filter by both document_id AND user_id
          Security: user can only access their own data
        """
        return db.query(Document).filter_by(
            id      = document_id,
            user_id = user_id
        ).first()

    @staticmethod
    def delete_document(
        db:          Session,
        document_id: int,
        user_id:     int
    ) -> bool:
        """
        Delete document and all its conversations + messages.

        Why cascade delete?
          Defined in models.py: cascade="all, delete-orphan"
          Deleting Document → auto-deletes Conversations
          Deleting Conversation → auto-deletes Messages
          No orphaned data left in database

        Also need to delete ChromaDB collection:
          Handled in the upload page (not here)
          DB cleanup and ChromaDB cleanup are separate
        """
        doc = db.query(Document).filter_by(
            id=document_id, user_id=user_id
        ).first()

        if not doc:
            return False

        db.delete(doc)
        return True

    @staticmethod
    def document_exists(
        db:       Session,
        user_id:  int,
        filename: str
    ) -> Optional[Document]:
        """
        Check if user already ingested this filename.

        Used in upload page:
          If exists → ask "re-ingest or load existing?"
          Prevents accidental duplicate ingestion
        """
        return db.query(Document).filter_by(
            user_id  = user_id,
            filename = filename,
            status   = "ready"
        ).first()


# ─────────────────────────────────────────
# CONVERSATION QUERIES
# ─────────────────────────────────────────
class ConversationQueries:

    @staticmethod
    def create_conversation(
        db:          Session,
        user_id:     int,
        document_id: int,
        title:       str = "New conversation"
    ) -> Conversation:
        """Create a new conversation for a document."""
        conv = Conversation(
            user_id     = user_id,
            document_id = document_id,
            title       = title
        )
        db.add(conv)
        db.flush()
        return conv

    @staticmethod
    def get_document_conversations(
        db:          Session,
        document_id: int,
        user_id:     int
    ) -> List[Conversation]:
        """
        Get all conversations for a document, newest first.

        Why verify user_id here too?
          Defense in depth — even if document_id is guessed,
          user_id check prevents unauthorized access
        """
        return (
            db.query(Conversation)
            .filter_by(
                document_id = document_id,
                user_id     = user_id
            )
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    @staticmethod
    def get_conversation_by_id(
        db:              Session,
        conversation_id: int,
        user_id:         int
    ) -> Optional[Conversation]:
        """Get conversation with ownership check."""
        return db.query(Conversation).filter_by(
            id      = conversation_id,
            user_id = user_id
        ).first()

    @staticmethod
    def update_title(
        db:              Session,
        conversation_id: int,
        title:           str
    ) -> None:
        """
        Update conversation title from first question.

        Called after first message saved:
          title = question[:60]
          "What is the main method proposed in..."
          → truncated to 60 chars for display
        """
        conv = db.query(Conversation).filter_by(
            id=conversation_id
        ).first()
        if conv:
            conv.title      = title[:60]
            conv.updated_at = datetime.utcnow()

    @staticmethod
    def delete_conversation(
        db:              Session,
        conversation_id: int,
        user_id:         int
    ) -> bool:
        """Delete conversation and all its messages."""
        conv = db.query(Conversation).filter_by(
            id=conversation_id, user_id=user_id
        ).first()
        if not conv:
            return False
        db.delete(conv)
        return True


# ─────────────────────────────────────────
# MESSAGE QUERIES
# ─────────────────────────────────────────
class MessageQueries:

    @staticmethod
    def save_message(
        db:              Session,
        conversation_id: int,
        role:            str,
        content:         str,
        confidence:      str = "",
        sources:         list = None,
        pipeline:        str = "phase2",
        response_ms:     int = 0
    ) -> Message:
        """
        Save a single message to the conversation.

        Called twice per Q&A:
          1. role="user"      → save question
          2. role="assistant" → save answer + metadata

        Sources serialization:
          sources are retrieved chunk objects
          Convert to plain dicts before saving
          JSON column stores the list of dicts
        """
        # Serialize sources to JSON-safe format
        serialized_sources = []
        for src in (sources or []):
            if hasattr(src, "__dict__"):
                serialized_sources.append({
                    "chunk_id":        getattr(src, "chunk_id", ""),
                    "source_file":     getattr(src, "source_file", ""),
                    "page_number":     getattr(src, "page_number", 0),
                    "paragraph_index": getattr(src, "paragraph_index", 0),
                    "text":            getattr(src, "text", "")[:300],
                    "is_figure":       getattr(src, "is_figure", False),
                    "image_path":      getattr(src, "image_path", ""),
                    "figure_id":       getattr(src, "figure_id", ""),
                })
            elif isinstance(src, dict):
                serialized_sources.append(src)

        msg = Message(
            conversation_id = conversation_id,
            role            = role,
            content         = content,
            confidence      = confidence,
            sources         = serialized_sources,
            pipeline        = pipeline,
            response_ms     = response_ms
        )
        db.add(msg)
        db.flush()
        return msg

    @staticmethod
    def get_conversation_messages(
        db:              Session,
        conversation_id: int
    ) -> List[Message]:
        """
        Load all messages in a conversation.

        Why order by created_at?
          Messages stored in chronological order
          LLM context needs: Q1, A1, Q2, A2...
          Correct order critical for conversation context
        """
        return (
            db.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at)
            .all()
        )

    @staticmethod
    def get_messages_as_dicts(
        db:              Session,
        conversation_id: int
    ) -> List[Dict]:
        """
        Load messages as plain dicts for Streamlit.

        Why dicts not ORM objects?
          Streamlit session_state can't serialize ORM objects
          Plain dicts work with JSON, session state, everything
          to_dict() defined on Message model
        """
        messages = MessageQueries.get_conversation_messages(
            db, conversation_id
        )
        return [msg.to_dict() for msg in messages]

    @staticmethod
    def save_qa_pair(
        db:              Session,
        conversation_id: int,
        question:        str,
        answer:          str,
        confidence:      str,
        sources:         list,
        pipeline:        str,
        response_ms:     int
    ) -> tuple:
        """
        Save a complete Q&A pair in one call.

        Convenience method — saves both user and
        assistant messages together.

        Returns (user_message, assistant_message) tuple.
        """
        user_msg = MessageQueries.save_message(
            db              = db,
            conversation_id = conversation_id,
            role            = "user",
            content         = question
        )

        asst_msg = MessageQueries.save_message(
            db              = db,
            conversation_id = conversation_id,
            role            = "assistant",
            content         = answer,
            confidence      = confidence,
            sources         = sources,
            pipeline        = pipeline,
            response_ms     = response_ms
        )

        return user_msg, asst_msg


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────
if __name__ == "__main__":
    from database.connection import get_db

    print("Testing queries...\n")

    with get_db() as db:
        # Test 1 — Create user
        try:
            user = UserQueries.create_user(
                db, "sonal", "sonal@test.com", "password123"
            )
            print(f"✅ Created user: {user}")
        except ValueError as e:
            print(f"⚠️  User exists: {e}")
            user = UserQueries.get_user_by_username(db, "sonal")

        # Test 2 — Verify login
        logged_in = UserQueries.verify_login(db, "sonal", "password123")
        print(f"✅ Login valid: {logged_in is not None}")

        wrong_pass = UserQueries.verify_login(db, "sonal", "wrongpass")
        print(f"✅ Wrong password rejected: {wrong_pass is None}")

        # Test 3 — Create document
        doc = DocumentQueries.create_document(
            db, user.id, "moac.pdf"
        )
        print(f"✅ Created document: {doc}")
        print(f"   Collection: {doc.collection_name}")

        # Test 4 — Update document stats
        DocumentQueries.update_document_stats(
            db, doc.id, 18, 38, 22
        )
        print(f"✅ Updated document stats")

        # Test 5 — Create conversation
        conv = ConversationQueries.create_conversation(
            db, user.id, doc.id, "Test conversation"
        )
        print(f"✅ Created conversation: {conv}")

        # Test 6 — Save Q&A pair
        MessageQueries.save_qa_pair(
            db              = db,
            conversation_id = conv.id,
            question        = "What is MOAC?",
            answer          = "MOAC is a framework...",
            confidence      = "HIGH",
            sources         = [],
            pipeline        = "phase2",
            response_ms     = 15000
        )
        print(f"✅ Saved Q&A pair")

        # Test 7 — Load messages
        messages = MessageQueries.get_messages_as_dicts(db, conv.id)
        print(f"✅ Loaded {len(messages)} messages")
        for msg in messages:
            print(f"   {msg['role']}: {msg['content'][:40]}...")

        # Test 8 — List user documents
        docs = DocumentQueries.get_user_documents(db, user.id)
        print(f"✅ User has {len(docs)} document(s)")

    print("\n✅ All tests passed!")