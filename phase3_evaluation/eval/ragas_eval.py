import json
import os
from typing import List, Dict
from datetime import datetime
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import (
    GROQ_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    validate_settings
)


class RAGASEvaluator:
    def __init__(self):
        validate_settings()

        base_llm = ChatGroq(
            model=LLM_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.0
        )
        base_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.llm        = LangchainLLMWrapper(base_llm)
        self.embeddings = LangchainEmbeddingsWrapper(base_embeddings)

        # Configure metrics to use our LLM and embeddings
        self.metrics = [faithfulness, answer_relevancy,
                        context_precision, context_recall]
        for m in self.metrics:
            m.llm = self.llm
            if hasattr(m, "embeddings"):
                m.embeddings = self.embeddings

        print(f"[RAGAS] Evaluator ready — model: {LLM_MODEL}")

    def _run_phase1(self, question: str) -> Dict:
        from phase1_fundamentals.generation.chain import RAGChain
        chain = RAGChain()
        response = chain.ask(question)
        return {
            "answer":   response["answer"],
            "contexts": [chunk.text for chunk in response["sources"]]
        }

    def _run_phase2(self, question: str) -> Dict:
        from phase2_production.generation.graph import RAGGraph
        graph = RAGGraph()
        response = graph.ask(question)
        return {
            "answer":   response["answer"],
            "contexts": [chunk.text for chunk in response["sources"]]
        }

    def _score_single_pair(self, pair: Dict, pipeline: str) -> Dict:
        """
        Score a single QA pair using RAGAS.
        Running one pair at a time avoids Groq rate limit errors.
        """
        import time

        if pipeline == "phase1":
            result = self._run_phase1(pair["question"])
        else:
            result = self._run_phase2(pair["question"])

        # Build single-row dataset
        dataset = Dataset.from_dict({
            "question":     [pair["question"]],
            "answer":       [result["answer"]],
            "contexts":     [result["contexts"]],
            "ground_truth": [pair["ground_truth"]]
        })

        # Score with RAGAS — one pair at a time
        from ragas import evaluate
        scores = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False
        )

        # Extract scores safely
        def safe_score(key):
            try:
                val = scores[key]
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    vals = [v for v in list(val) if v is not None
                            and str(v) != 'nan']
                    return round(float(sum(vals)/len(vals)), 4) if vals else None
                v = float(val)
                return None if str(v) == 'nan' else round(v, 4)
            except:
                return None

        time.sleep(2)  # 2 second delay between pairs to respect rate limits

        return {
            "faithfulness":      safe_score("faithfulness"),
            "answer_relevancy":  safe_score("answer_relevancy"),
            "context_precision": safe_score("context_precision"),
            "context_recall":    safe_score("context_recall"),
        }

    def evaluate_pipeline(
        self,
        qa_pairs: List[Dict],
        pipeline: str,
        max_pairs: int = 10
    ) -> Dict:
        """Evaluate pipeline one QA pair at a time."""
        print(f"\n[RAGAS] Evaluating {pipeline} — {max_pairs} pairs one by one...")

        all_scores = {
            "faithfulness":      [],
            "answer_relevancy":  [],
            "context_precision": [],
            "context_recall":    [],
        }

        eval_pairs = qa_pairs[:max_pairs]

        for i, pair in enumerate(eval_pairs, 1):
            print(f"\n[RAGAS] {pipeline} pair {i}/{max_pairs}: "
                  f"{pair['question'][:60]}...")
            try:
                scores = self._score_single_pair(pair, pipeline)
                for key in all_scores:
                    if scores[key] is not None:
                        all_scores[key].append(scores[key])
                        print(f"  {key:<22}: {scores[key]}")
                    else:
                        print(f"  {key:<22}: skipped (NaN)")
            except Exception as e:
                print(f"  [RAGAS] Error: {e}")
                continue

        # Average scores across all pairs
        def avg(lst):
            return round(sum(lst)/len(lst), 4) if lst else 0.0

        final = {
            "pipeline":          pipeline,
            "num_pairs":         max_pairs,
            "num_scored":        len(all_scores["faithfulness"]),
            "faithfulness":      avg(all_scores["faithfulness"]),
            "answer_relevancy":  avg(all_scores["answer_relevancy"]),
            "context_precision": avg(all_scores["context_precision"]),
            "context_recall":    avg(all_scores["context_recall"]),
            "evaluated_at":      datetime.now().isoformat()
        }

        print(f"\n[RAGAS] {pipeline.upper()} Final Scores:")
        print(f"  Faithfulness      : {final['faithfulness']}")
        print(f"  Answer Relevancy  : {final['answer_relevancy']}")
        print(f"  Context Precision : {final['context_precision']}")
        print(f"  Context Recall    : {final['context_recall']}")

        return final

    def evaluate_both(
        self,
        dataset_path: str = "phase3_evaluation/golden_dataset/dataset.json",
        output_path: str  = "phase3_evaluation/eval/scores.json",
        max_pairs: int    = 10
    ) -> Dict:
        """Evaluate both pipelines and save scores."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                "Run generator.py and curator.py first."
            )

        with open(dataset_path, "r") as f:
            data = json.load(f)

        qa_pairs = data["qa_pairs"]
        print(f"[RAGAS] Loaded {len(qa_pairs)} QA pairs")
        print(f"[RAGAS] Scoring first {max_pairs} pairs per pipeline")
        print(f"[RAGAS] Running sequentially to avoid rate limits\n")

        phase1_scores = self.evaluate_pipeline(qa_pairs, "phase1", max_pairs)
        phase2_scores = self.evaluate_pipeline(qa_pairs, "phase2", max_pairs)

        all_scores = {
            "phase1":       phase1_scores,
            "phase2":       phase2_scores,
            "dataset_info": data.get("metadata", {})
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_scores, f, indent=2)

        print(f"\n[RAGAS] All scores saved to: {output_path}")
        return all_scores


if __name__ == "__main__":
    evaluator = RAGASEvaluator()
    # max_pairs=10 keeps runtime to ~30-40 mins on free Groq tier
    scores = evaluator.evaluate_both(max_pairs=10)
    print("\n[RAGAS] Done! Run report.py to see comparison table.")
        