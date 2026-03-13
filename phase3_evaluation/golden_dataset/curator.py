import json
import os
from typing import List, Dict, Tuple
from datetime import datetime


# ─────────────────────────────────────────
# Quality filters
# ─────────────────────────────────────────
MIN_QUESTION_LENGTH = 15        # characters
MIN_ANSWER_LENGTH = 30          # characters
MAX_ANSWER_LENGTH = 1000        # characters
MIN_QUESTION_WORDS = 5          # words
TARGET_MIN = 50                 # minimum pairs in final dataset
TARGET_MAX = 100                # maximum pairs in final dataset


class DatasetCurator:
    def __init__(self):
        self.removed = {
            "too_short_question":  0,
            "too_short_answer":    0,
            "too_long_answer":     0,
            "duplicate_question":  0,
            "low_quality":         0,
            "yes_no_question":     0,
        }
        print("[Curator] Dataset curator ready")

    # ─────────────────────────────────────────
    # Individual quality checks
    # ─────────────────────────────────────────
    def _is_too_short(self, pair: Dict) -> Tuple[bool, str]:
        """Check if question or answer is too short."""
        q = pair.get("question", "")
        a = pair.get("ground_truth", "")

        if len(q) < MIN_QUESTION_LENGTH:
            return True, "too_short_question"
        if len(q.split()) < MIN_QUESTION_WORDS:
            return True, "too_short_question"
        if len(a) < MIN_ANSWER_LENGTH:
            return True, "too_short_answer"
        return False, ""

    def _is_too_long(self, pair: Dict) -> Tuple[bool, str]:
        """Check if answer is too long."""
        a = pair.get("ground_truth", "")
        if len(a) > MAX_ANSWER_LENGTH:
            return True, "too_long_answer"
        return False, ""

    def _is_yes_no(self, pair: Dict) -> Tuple[bool, str]:
        """Detect yes/no questions — not useful for evaluation."""
        q = pair.get("question", "").lower().strip()
        yes_no_starters = ["is ", "are ", "was ", "were ", "do ", "does ",
                           "did ", "can ", "could ", "should ", "would ",
                           "has ", "have ", "had "]
        for starter in yes_no_starters:
            if q.startswith(starter):
                return True, "yes_no_question"
        return False, ""

    def _is_low_quality(self, pair: Dict) -> Tuple[bool, str]:
        """Detect low quality patterns."""
        q = pair.get("question", "").lower()
        a = pair.get("ground_truth", "").lower()

        # Question contains no question word
        question_words = ["what", "how", "why", "which", "when",
                         "where", "who", "describe", "explain"]
        has_question_word = any(w in q for w in question_words)
        if not has_question_word:
            return True, "low_quality"

        # Answer is just a repeat of the question
        if q.replace("?", "").strip() in a:
            return True, "low_quality"

        return False, ""

    def _is_duplicate(
        self,
        pair: Dict,
        seen_questions: set
    ) -> Tuple[bool, str]:
        """Detect duplicate or near-duplicate questions."""
        q = pair.get("question", "").lower().strip()

        # Exact duplicate
        if q in seen_questions:
            return True, "duplicate_question"

        # Near duplicate — first 50 chars match
        prefix = q[:50]
        for seen in seen_questions:
            if seen[:50] == prefix:
                return True, "duplicate_question"

        return False, ""

    # ─────────────────────────────────────────
    # Main curation pipeline
    # ─────────────────────────────────────────
    def curate(
        self,
        input_path: str = "phase3_evaluation/golden_dataset/raw_dataset.json",
        output_path: str = "phase3_evaluation/golden_dataset/dataset.json",
        target_min: int = TARGET_MIN,
        target_max: int = TARGET_MAX
    ) -> List[Dict]:
        """
        Load raw dataset, apply quality filters,
        balance question types, save curated dataset.
        """
        # Load raw dataset
        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"Raw dataset not found: {input_path}\n"
                f"Run generator.py first."
            )

        with open(input_path, "r") as f:
            raw_data = json.load(f)

        raw_pairs = raw_data.get("qa_pairs", [])
        source_pdf = raw_data.get("metadata", {}).get("source_pdf", "unknown")

        print(f"\n[Curator] Loaded {len(raw_pairs)} raw QA pairs")
        print(f"[Curator] Applying quality filters...")

        # Apply filters
        seen_questions = set()
        passed = []

        for pair in raw_pairs:
            # Run all quality checks
            checks = [
                self._is_too_short(pair),
                self._is_too_long(pair),
                self._is_yes_no(pair),
                self._is_low_quality(pair),
                self._is_duplicate(pair, seen_questions),
            ]

            failed = [(removed, reason) for removed, reason in checks if removed]

            if failed:
                _, reason = failed[0]
                self.removed[reason] += 1
                continue

            # Passed all checks
            seen_questions.add(pair["question"].lower().strip())
            passed.append(pair)

        print(f"[Curator] Passed quality filters: {len(passed)} pairs")

        # Balance question types
        balanced = self._balance_types(passed, target_max)

        # Enforce target range
        if len(balanced) > target_max:
            balanced = balanced[:target_max]
        elif len(balanced) < target_min:
            print(f"[Curator] WARNING: Only {len(balanced)} pairs available "
                  f"(target minimum: {target_min})")
            print(f"[Curator] Consider re-running generator with more chunks")

        # Re-index with clean IDs
        for i, pair in enumerate(balanced, 1):
            pair["id"] = f"qa_{i:03d}"

        # Save curated dataset
        output = {
            "metadata": {
                "source_pdf":    source_pdf,
                "total_pairs":   len(balanced),
                "question_types": self._count_types(balanced),
                "curated_by":    "DatasetCurator",
                "created_at":    datetime.now().isoformat(),
                "status":        "curated — ready for evaluation",
                "filters_applied": {
                    "min_question_length": MIN_QUESTION_LENGTH,
                    "min_answer_length":   MIN_ANSWER_LENGTH,
                    "max_answer_length":   MAX_ANSWER_LENGTH,
                }
            },
            "qa_pairs": balanced
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        # Print summary
        self._print_summary(len(raw_pairs), len(passed), len(balanced))

        return balanced

    def _balance_types(
        self,
        pairs: List[Dict],
        target: int
    ) -> List[Dict]:
        """
        Balance factual and inferential questions.
        Aim for roughly 70% factual, 30% inferential.
        """
        factual = [p for p in pairs if p.get("question_type") == "factual"]
        inferential = [p for p in pairs if p.get("question_type") == "inferential"]

        # Target split
        target_inferential = min(len(inferential), int(target * 0.3))
        target_factual = min(len(factual), target - target_inferential)

        balanced = factual[:target_factual] + inferential[:target_inferential]
        return balanced

    def _count_types(self, pairs: List[Dict]) -> Dict:
        """Count question types in dataset."""
        counts = {}
        for pair in pairs:
            qtype = pair.get("question_type", "unknown")
            counts[qtype] = counts.get(qtype, 0) + 1
        return counts

    def _print_summary(
        self,
        raw_count: int,
        passed_count: int,
        final_count: int
    ) -> None:
        """Print curation summary."""
        print(f"\n[Curator] ── Curation Summary ──")
        print(f"  Raw pairs        : {raw_count}")
        print(f"  Passed filters   : {passed_count}")
        print(f"  Final dataset    : {final_count}")
        print(f"\n  Removed by reason:")
        for reason, count in self.removed.items():
            if count > 0:
                print(f"    {reason:<25}: {count}")
        print(f"\n[Curator] Saved to: phase3_evaluation/golden_dataset/dataset.json")


if __name__ == "__main__":
    curator = DatasetCurator()
    pairs = curator.curate()

    print(f"\n--- Preview of first 3 curated pairs ---")
    for pair in pairs[:3]:
        print(f"\nID       : {pair['id']}")
        print(f"Type     : {pair['question_type']}")
        print(f"Page     : {pair['source_page']}")
        print(f"Question : {pair['question']}")
        print(f"Answer   : {pair['ground_truth'][:150]}...")