"""
Run Phase 2 evaluation only.
Use this after Phase 1 scores are already saved in scores.json
"""
import json
import os
import time
from phase3_evaluation.eval.ragas_eval import RAGASEvaluator

def run_phase2_only(
    dataset_path = "phase3_evaluation/golden_dataset/dataset.json",
    scores_path  = "phase3_evaluation/eval/scores.json",
    max_pairs    = 10
):
    # Load existing scores (keep Phase 1 results)
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            existing = json.load(f)
        print(f"[Runner] Loaded existing Phase 1 scores")
        print(f"  Faithfulness      : {existing['phase1']['faithfulness']}")
        print(f"  Answer Relevancy  : {existing['phase1']['answer_relevancy']}")
        print(f"  Context Precision : {existing['phase1']['context_precision']}")
        print(f"  Context Recall    : {existing['phase1']['context_recall']}")
    else:
        existing = {}
        print("[Runner] WARNING: No existing scores found")

    # Load dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)
    qa_pairs = data["qa_pairs"]

    print(f"\n[Runner] Running Phase 2 evaluation only...")
    print(f"[Runner] Pairs: {max_pairs} | Delay: 10s between pairs")
    print(f"[Runner] Estimated time: ~{max_pairs * 3} minutes\n")

    # Initialize evaluator with longer delay
    evaluator = RAGASEvaluator()

    # Monkey-patch delay to 10 seconds
    original_score = evaluator._score_single_pair
    def slow_score(pair, pipeline):
        result = original_score(pair, pipeline)
        print(f"[Runner] Waiting 10s before next pair...")
        time.sleep(10)
        return result
    evaluator._score_single_pair = slow_score

    # Run Phase 2 only
    phase2_scores = evaluator.evaluate_pipeline(qa_pairs, "phase2", max_pairs)

    # Merge with existing Phase 1 scores and save
    all_scores = {
        "phase1": existing.get("phase1", {}),
        "phase2": phase2_scores,
        "dataset_info": data.get("metadata", {})
    }

    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\n[Runner] Phase 2 scores saved!")
    print(f"[Runner] Run report.py to see final comparison")


if __name__ == "__main__":
    run_phase2_only(max_pairs=10)