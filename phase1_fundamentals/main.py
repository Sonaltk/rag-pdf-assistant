import argparse
import sys
from pathlib import Path

from phase1_fundamentals.ingestion.loader import PDFLoader
from phase1_fundamentals.ingestion.chunker import TokenChunker
from phase1_fundamentals.ingestion.embedder import Embedder
from phase1_fundamentals.generation.chain import RAGChain


def ingest(pdf_path: str, clear: bool = False):
    print("\n" + "=" * 60)
    print("PHASE 1 - INGESTION PIPELINE")
    print("=" * 60)

    if not Path(pdf_path).exists():
        print(f"[Error] PDF not found: {pdf_path}")
        sys.exit(1)

    print("\nStep 1/3 - Loading PDF...")
    loader = PDFLoader(pdf_path)
    pages = loader.load()

    print("\nStep 2/3 - Chunking pages...")
    chunker = TokenChunker()
    chunks = chunker.chunk_pages(pages)

    print("\nStep 3/3 - Embedding and storing in ChromaDB...")
    embedder = Embedder()

    if clear:
        print("[Main] Clearing existing collection first...")
        embedder.clear_collection()

    embedder.embed_chunks(chunks)

    info = embedder.get_collection_info()
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"  PDF          : {pdf_path}")
    print(f"  Pages loaded : {len(pages)}")
    print(f"  Chunks stored: {info['total_chunks']}")
    print(f"  ChromaDB     : {info['db_path']}")
    print("=" * 60)


def query_interactive():
    print("\n" + "=" * 60)
    print("PHASE 1 - INTERACTIVE Q&A")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    chain = RAGChain()

    while True:
        print()
        question = input("Your question: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        try:
            response = chain.ask(question)
            chain.display_response(response)
        except Exception as e:
            print(f"[Error] {e}")


def query_once(question: str):
    chain = RAGChain()
    response = chain.ask(question)
    chain.display_response(response)


def main():
    parser = argparse.ArgumentParser(
        description="RAG PDF Assistant - Phase 1",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF into ChromaDB")
    ingest_parser.add_argument("pdf", type=str, help="Path to the PDF file")
    ingest_parser.add_argument("--clear", action="store_true", help="Clear existing collection before ingesting")

    ask_parser = subparsers.add_parser("ask", help="Ask a question about the ingested PDF")
    ask_parser.add_argument("--question", "-q", type=str, default=None, help="Question to ask (omit for interactive mode)")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.pdf, clear=args.clear)
    elif args.command == "ask":
        if args.question:
            query_once(args.question)
        else:
            query_interactive()


if __name__ == "__main__":
    main()