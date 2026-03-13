import argparse
import sys


def run_generate(pdf_path: str):
    """Generate raw QA pairs from PDF."""
    from phase1_fundamentals.ingestion.loader import PDFLoader
    from phase1_fundamentals.ingestion.chunker import TokenChunker
    from phase3_evaluation.golden_dataset.generator import QAGenerator

    print("\n" + "=" * 60)
    print("PHASE 3 — STEP 1: QA PAIR GENERATION")
    print("=" * 60)

    loader = PDFLoader(pdf_path)
    pages = loader.load()

    chunker = TokenChunker()
    chunks = chunker.chunk_pages(pages)

    generator = QAGenerator()
    pairs = generator.generate(chunks)

    print(f"\n✅ Generated {len(pairs)} raw QA pairs")
    print(f"   Saved to: phase3_evaluation/golden_dataset/raw_dataset.json")
    print(f"\nNext step: python phase3_evaluation/main.py curate")


def run_curate():
    """Curate raw QA pairs into golden dataset."""
    from phase3_evaluation.golden_dataset.curator import DatasetCurator

    print("\n" + "=" * 60)
    print("PHASE 3 — STEP 2: DATASET CURATION")
    print("=" * 60)

    curator = DatasetCurator()
    pairs = curator.curate()

    print(f"\n✅ Curated dataset ready — {len(pairs)} high quality pairs")
    print(f"   Saved to: phase3_evaluation/golden_dataset/dataset.json")
    print(f"\nNext step: python phase3_evaluation/main.py evaluate")


def run_evaluate(max_pairs: int = 10, pipeline: str = "both"):
    """Run RAGAS evaluation on one or both pipelines."""
    from phase3_evaluation.eval.ragas_eval import RAGASEvaluator

    print("\n" + "=" * 60)
    print("PHASE 3 — STEP 3: RAGAS EVALUATION")
    print("=" * 60)
    print(f"Pipeline : {pipeline}")
    print(f"Max pairs: {max_pairs}")

    evaluator = RAGASEvaluator()

    import json
    import os

    dataset_path = "phase3_evaluation/golden_dataset/dataset.json"
    scores_path  = "phase3_evaluation/eval/scores.json"

    with open(dataset_path, "r") as f:
        data = json.load(f)
    qa_pairs = data["qa_pairs"]

    # Load existing scores if available
    existing = {}
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            existing = json.load(f)

    if pipeline in ("both", "phase1"):
        p1_scores = evaluator.evaluate_pipeline(qa_pairs, "phase1", max_pairs)
        existing["phase1"] = p1_scores

    if pipeline in ("both", "phase2"):
        p2_scores = evaluator.evaluate_pipeline(qa_pairs, "phase2", max_pairs)
        existing["phase2"] = p2_scores

    existing["dataset_info"] = data.get("metadata", {})

    os.makedirs("phase3_evaluation/eval", exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n✅ Evaluation complete — scores saved to {scores_path}")
    print(f"\nNext step: python phase3_evaluation/main.py report")


def run_report():
    """Generate comparison report."""
    from phase3_evaluation.eval.report import generate_report

    print("\n" + "=" * 60)
    print("PHASE 3 — STEP 4: EVALUATION REPORT")
    print("=" * 60)

    generate_report()
    print(f"\n✅ Report saved to: phase3_evaluation/eval/evaluation_report.txt")


def run_all(pdf_path: str, max_pairs: int = 10):
    """Run the complete Phase 3 pipeline end to end."""
    print("\n" + "=" * 60)
    print("PHASE 3 — FULL PIPELINE")
    print("=" * 60)
    print(f"PDF      : {pdf_path}")
    print(f"Max pairs: {max_pairs}")

    run_generate(pdf_path)
    run_curate()
    run_evaluate(max_pairs, "both")
    run_report()

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE!")
    print("=" * 60)
    print("Files generated:")
    print("  phase3_evaluation/golden_dataset/raw_dataset.json")
    print("  phase3_evaluation/golden_dataset/dataset.json")
    print("  phase3_evaluation/eval/scores.json")
    print("  phase3_evaluation/eval/evaluation_report.txt")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAG PDF Assistant — Phase 3 Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Commands:
  generate  → auto-generate QA pairs from PDF
  curate    → filter and clean raw QA pairs
  evaluate  → run RAGAS metrics on pipelines
  report    → generate comparison report
  all       → run full pipeline end to end

Examples:
  python phase3_evaluation/main.py generate data/raw/moac.pdf
  python phase3_evaluation/main.py curate
  python phase3_evaluation/main.py evaluate --max-pairs 10 --pipeline phase2
  python phase3_evaluation/main.py report
  python phase3_evaluation/main.py all data/raw/moac.pdf
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate raw QA pairs from PDF")
    gen_parser.add_argument("pdf", type=str, help="Path to PDF file")

    # curate
    subparsers.add_parser("curate", help="Curate raw QA pairs into golden dataset")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run RAGAS evaluation")
    eval_parser.add_argument(
        "--max-pairs", "-m",
        type=int, default=10,
        help="Number of QA pairs to evaluate (default: 10)"
    )
    eval_parser.add_argument(
        "--pipeline", "-p",
        choices=["phase1", "phase2", "both"],
        default="both",
        help="Which pipeline to evaluate (default: both)"
    )

    # report
    subparsers.add_parser("report", help="Generate comparison report")

    # all
    all_parser = subparsers.add_parser("all", help="Run full Phase 3 pipeline")
    all_parser.add_argument("pdf", type=str, help="Path to PDF file")
    all_parser.add_argument(
        "--max-pairs", "-m",
        type=int, default=10,
        help="Number of QA pairs to evaluate (default: 10)"
    )

    args = parser.parse_args()

    if args.command == "generate":
        run_generate(args.pdf)
    elif args.command == "curate":
        run_curate()
    elif args.command == "evaluate":
        run_evaluate(args.max_pairs, args.pipeline)
    elif args.command == "report":
        run_report()
    elif args.command == "all":
        run_all(args.pdf, args.max_pairs)


if __name__ == "__main__":
    main()