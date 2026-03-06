import argparse
import sys
from pathlib import Path

from phase1_fundamentals.ingestion.loader import PDFLoader
from phase1_fundamentals.ingestion.chunker import TokenChunker
from phase1_fundamentals.ingestion.embedder import Embedder
from phase2_production.retrieval.bm25_retriever import BM25Retriever
from phase2_production.generation.graph import RAGGraph


# ─────────────────────────────────────────
# INGEST — Phase 2 ingestion
# Runs Phase 1 ingestion + builds BM25 index
# ─────────────────────────────────────────
def ingest(pdf_path: str, clear: bool = False):
    print("\n" + "=" * 60)
    print("PHASE 2 — INGESTION PIPELINE")
    print("=" * 60)

    if not Path(pdf_path).exists():
        print(f"[Error] PDF not found: {pdf_path}")
        sys.exit(1)

    # Step 1 — Load
    print("\nStep 1/4 — Loading PDF...")
    loader = PDFLoader(pdf_path)
    pages = loader.load()

    # Step 2 — Chunk
    print("\nStep 2/4 — Chunking pages...")
    chunker = TokenChunker()
    chunks = chunker.chunk_pages(pages)

    # Step 3 — Embed + Store in ChromaDB
    print("\nStep 3/4 — Embedding and storing in ChromaDB...")
    embedder = Embedder()
    if clear:
        embedder.clear_collection()
    embedder.embed_chunks(chunks)

    # Step 4 — Build BM25 index (new in Phase 2)
    print("\nStep 4/4 — Building BM25 keyword index...")
    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    info = embedder.get_collection_info()
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"  PDF           : {pdf_path}")
    print(f"  Pages loaded  : {len(pages)}")
    print(f"  Chunks stored : {info['total_chunks']} (ChromaDB)")
    print(f"  BM25 index    : data/chunks/bm25_index.json")
    print("=" * 60)


# ─────────────────────────────────────────
# QUERY — single question
# ─────────────────────────────────────────
def query_once(question: str):
    graph = RAGGraph()
    response = graph.ask(question)
    graph.display_response(response)


# ─────────────────────────────────────────
# QUERY — interactive mode
# ─────────────────────────────────────────
def query_interactive():
    print("\n" + "=" * 60)
    print("PHASE 2 — INTERACTIVE Q&A")
    print("Hybrid retrieval + Cohere reranking + LangGraph")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    graph = RAGGraph()

    while True:
        print()
        question = input("Your question: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        try:
            response = graph.ask(question)
            graph.display_response(response)
        except Exception as e:
            print(f"[Error] {e}")


# ─────────────────────────────────────────
# COMPARE — Phase 1 vs Phase 2 side by side
# ─────────────────────────────────────────
def compare(question: str):
    print("\n" + "=" * 60)
    print("PHASE 1 vs PHASE 2 COMPARISON")
    print(f"Question: {question}")
    print("=" * 60)

    # Phase 1 answer
    print("\n--- PHASE 1 (Vector only) ---")
    from phase1_fundamentals.generation.chain import RAGChain
    chain = RAGChain()
    p1_response = chain.ask(question)
    print(p1_response["answer"])
    print(f"\nSources: {len(p1_response['sources'])} chunks")
    for c in p1_response["sources"]:
        print(f"  Page {c.page_number} | Para {c.paragraph_index} | Score {c.similarity_score}")

    # Phase 2 answer
    print("\n--- PHASE 2 (Hybrid + Reranked) ---")
    graph = RAGGraph()
    p2_response = graph.ask(question)
    print(p2_response["answer"])
    print(f"\nSources: {len(p2_response['sources'])} chunks")
    for c in p2_response["sources"]:
        print(f"  Page {c.page_number} | Para {c.paragraph_index} | Cohere {c.rerank_score} ({c.confidence})")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print(f"  Phase 1 chunks : {len(p1_response['sources'])}")
    print(f"  Phase 2 chunks : {len(p2_response['sources'])}")
    print(f"  Phase 2 confidence : {p2_response['confidence']}")
    print("=" * 60)


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAG PDF Assistant — Phase 2",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF (ChromaDB + BM25)")
    ingest_parser.add_argument("pdf", type=str, help="Path to PDF file")
    ingest_parser.add_argument("--clear", action="store_true", help="Clear existing data first")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question (Phase 2 pipeline)")
    ask_parser.add_argument("--question", "-q", type=str, default=None, help="Question to ask")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare Phase 1 vs Phase 2 answers")
    compare_parser.add_argument("--question", "-q", type=str, required=True, help="Question to compare")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.pdf, clear=args.clear)
    elif args.command == "ask":
        if args.question:
            query_once(args.question)
        else:
            query_interactive()
    elif args.command == "compare":
        compare(args.question)


if __name__ == "__main__":
    main()