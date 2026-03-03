import hashlib
import tiktoken
from dataclasses import dataclass
from typing import List

from phase1_fundamentals.ingestion.loader import PageContent
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


# ─────────────────────────────────────────
# Data Model — one Chunk with full metadata
# This schema carries through ALL phases
# ─────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: str           # sha256 hash of content — unique identifier
    source_file: str        # original PDF filename
    page_number: int        # which page this chunk came from
    paragraph_index: int    # position of chunk within the page (0-indexed)
    text: str               # actual chunk text
    token_count: int        # number of tokens in this chunk
    char_start: int         # character start position in original page text
    char_end: int           # character end position in original page text


def _generate_chunk_id(text: str) -> str:
    """Generate a unique ID for a chunk based on its content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────
# Token-Aware Chunker
# ─────────────────────────────────────────
class TokenChunker:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base"  # used by GPT-4, text-embedding-3
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string."""
        return len(self.tokenizer.encode(text))

    def _split_text_into_chunks(self, text: str) -> List[tuple]:
        """
        Split text into (chunk_text, char_start, char_end) tuples.
        Uses a sliding window of tokens with overlap.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            # Decode this window of tokens back to text
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Find char positions in original text
            char_start = text.find(chunk_text[:50])  # anchor on first 50 chars
            char_end = char_start + len(chunk_text) if char_start != -1 else -1

            chunks.append((chunk_text.strip(), char_start, char_end))

            # If we've reached the end, stop
            if end == len(tokens):
                break

            # Slide window forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        """
        Take a list of PageContent and return a flat list of Chunks.
        Each chunk knows which page and paragraph it came from.
        """
        all_chunks = []

        for page in pages:
            raw_chunks = self._split_text_into_chunks(page.text)

            for para_index, (chunk_text, char_start, char_end) in enumerate(raw_chunks):
                if not chunk_text.strip():
                    continue  # skip empty chunks

                chunk = Chunk(
                    chunk_id=_generate_chunk_id(chunk_text),
                    source_file=page.source_file,
                    page_number=page.page_number,
                    paragraph_index=para_index,
                    text=chunk_text,
                    token_count=self._count_tokens(chunk_text),
                    char_start=char_start,
                    char_end=char_end,
                )
                all_chunks.append(chunk)

        print(f"[Chunker] Created {len(all_chunks)} chunks from {len(pages)} pages")
        print(f"[Chunker] Avg tokens per chunk: {sum(c.token_count for c in all_chunks) // len(all_chunks)}")
        return all_chunks


# ─────────────────────────────────────────
# Quick test — run directly to verify
# python phase1_fundamentals/ingestion/chunker.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from phase1_fundamentals.ingestion.loader import PDFLoader

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_pdf>")
        sys.exit(1)

    # Load
    loader = PDFLoader(sys.argv[1])
    pages = loader.load()

    # Chunk
    chunker = TokenChunker()
    chunks = chunker.chunk_pages(pages)

    # Preview first 3 chunks
    print(f"\n--- Preview of first 3 chunks ---")
    for chunk in chunks[:3]:
        print(f"\nChunk ID   : {chunk.chunk_id}")
        print(f"Source     : {chunk.source_file}")
        print(f"Page       : {chunk.page_number}")
        print(f"Paragraph  : {chunk.paragraph_index}")
        print(f"Tokens     : {chunk.token_count}")
        print(f"Text       : {chunk.text[:150]}...")