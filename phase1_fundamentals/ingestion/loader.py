import fitz  # PyMuPDF
from dataclasses import dataclass
from pathlib import Path
from typing import List


# ─────────────────────────────────────────
# Data Model — one Page of raw extracted text
# ─────────────────────────────────────────
@dataclass
class PageContent:
    page_number: int        # 1-indexed page number
    text: str               # raw extracted text from that page
    source_file: str        # original PDF filename


# ─────────────────────────────────────────
# PDF Loader
# ─────────────────────────────────────────
class PDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF: {pdf_path}")

    def load(self) -> List[PageContent]:
        """
        Load all pages from the PDF.
        Returns a list of PageContent — one per page.
        """
        pages = []

        doc = fitz.open(str(self.pdf_path))
        print(f"[Loader] Opened '{self.pdf_path.name}' — {len(doc)} pages found")

        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")  # plain text extraction

            # Skip completely empty pages
            if not text.strip():
                print(f"[Loader] Skipping empty page {page_index + 1}")
                continue

            pages.append(PageContent(
                page_number=page_index + 1,   # human-readable: starts at 1
                text=text.strip(),
                source_file=self.pdf_path.name
            ))

        doc.close()
        print(f"[Loader] Successfully loaded {len(pages)} non-empty pages")
        return pages


# ─────────────────────────────────────────
# Quick test — run this file directly to verify
# python phase1_fundamentals/ingestion/loader.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python loader.py <path_to_pdf>")
        sys.exit(1)

    loader = PDFLoader(sys.argv[1])
    pages = loader.load()

    print(f"\n--- Preview of first page ---")
    print(f"Page: {pages[0].page_number}")
    print(f"Source: {pages[0].source_file}")
    print(f"Text preview: {pages[0].text[:300]}...")