import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import GROQ_API_KEY, LLM_MODEL, validate_settings
from phase1_fundamentals.ingestion.loader import PDFLoader
from phase1_fundamentals.ingestion.chunker import TokenChunker


# ─────────────────────────────────────────
# Prompts for QA generation
# ─────────────────────────────────────────
FACTUAL_PROMPT = ChatPromptTemplate.from_template("""
You are creating evaluation data for a RAG system.
Given the following text chunk from a research paper, generate 2 factual questions
and their precise answers based ONLY on the provided text.

Rules:
1. Questions must be answerable ONLY from this chunk
2. Answers must be specific and grounded in the text
3. Do NOT generate yes/no questions
4. Return ONLY valid JSON, no extra text

Text chunk:
{chunk_text}

Return this exact JSON format:
{{
  "pairs": [
    {{
      "question": "specific factual question here",
      "answer": "precise answer from the text here",
      "question_type": "factual"
    }},
    {{
      "question": "another specific question here",
      "answer": "precise answer from the text here",
      "question_type": "factual"
    }}
  ]
}}
""")

INFERENTIAL_PROMPT = ChatPromptTemplate.from_template("""
You are creating evaluation data for a RAG system.
Given the following text chunk from a research paper, generate 1 inferential question
that requires understanding and reasoning about the text, not just copying facts.

Rules:
1. Question should require understanding relationships or implications in the text
2. Answer must still be grounded in the provided text
3. Do NOT generate yes/no questions
4. Return ONLY valid JSON, no extra text

Text chunk:
{chunk_text}

Return this exact JSON format:
{{
  "pairs": [
    {{
      "question": "inferential question requiring reasoning here",
      "answer": "reasoned answer grounded in the text here",
      "question_type": "inferential"
    }}
  ]
}}
""")


# ─────────────────────────────────────────
# QA Generator
# ─────────────────────────────────────────
class QAGenerator:
    def __init__(self):
        validate_settings()

        self.llm = ChatGroq(
            model=LLM_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.3,        # slight creativity for varied questions
            max_tokens=1024
        )
        self.parser = StrOutputParser()
        print(f"[Generator] QA Generator ready — model: {LLM_MODEL}")

    def _generate_for_chunk(
        self,
        chunk,
        prompt: ChatPromptTemplate,
        prompt_type: str
    ) -> List[Dict]:
        """Generate QA pairs for a single chunk using given prompt."""
        try:
            chain = prompt | self.llm | self.parser
            raw = chain.invoke({"chunk_text": chunk.text})

            # Clean response — remove markdown code blocks if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            pairs = parsed.get("pairs", [])

            # Enrich each pair with chunk metadata
            enriched = []
            for pair in pairs:
                if not pair.get("question") or not pair.get("answer"):
                    continue
                enriched.append({
                    "id": hashlib.sha256(
                        pair["question"].encode()
                    ).hexdigest()[:12],
                    "question":      pair["question"],
                    "ground_truth":  pair["answer"],
                    "question_type": pair.get("question_type", prompt_type),
                    "source_chunk_id": chunk.chunk_id,
                    "source_page":   chunk.page_number,
                    "source_para":   chunk.paragraph_index,
                    "source_file":   chunk.source_file,
                })
            return enriched

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Generator] Parse error on chunk {chunk.chunk_id}: {e}")
            return []
        except Exception as e:
            print(f"[Generator] Error on chunk {chunk.chunk_id}: {e}")
            return []

    def generate(
        self,
        chunks: list,
        output_path: str = "phase3_evaluation/golden_dataset/raw_dataset.json",
        delay: float = 1.5      # seconds between API calls to avoid rate limits
    ) -> List[Dict]:
        """
        Generate QA pairs for all chunks.
        Each chunk gets:
          - 2 factual questions
          - 1 inferential question
        Total expected: 38 chunks × 3 = ~114 raw pairs
        """
        print(f"\n[Generator] Generating QA pairs for {len(chunks)} chunks...")
        print(f"[Generator] Expected output: ~{len(chunks) * 3} raw pairs")
        print(f"[Generator] Delay between calls: {delay}s (avoids rate limits)\n")

        all_pairs = []
        failed = 0

        for i, chunk in enumerate(chunks, 1):
            print(f"[Generator] Processing chunk {i}/{len(chunks)} "
                  f"(Page {chunk.page_number}, Para {chunk.paragraph_index})...")

            # Skip very short chunks — not enough content to generate questions
            if chunk.token_count < 50:
                print(f"[Generator] Skipping — too short ({chunk.token_count} tokens)")
                continue

            # Generate factual questions
            factual = self._generate_for_chunk(chunk, FACTUAL_PROMPT, "factual")
            all_pairs.extend(factual)
            time.sleep(delay)

            # Generate inferential question
            inferential = self._generate_for_chunk(
                chunk, INFERENTIAL_PROMPT, "inferential"
            )
            all_pairs.extend(inferential)
            time.sleep(delay)

            if not factual and not inferential:
                failed += 1

            print(f"[Generator] Running total: {len(all_pairs)} pairs generated")

        # Save raw dataset
        output = {
            "metadata": {
                "source_pdf":    chunks[0].source_file if chunks else "unknown",
                "total_chunks":  len(chunks),
                "total_pairs":   len(all_pairs),
                "failed_chunks": failed,
                "generated_by":  LLM_MODEL,
                "created_at":    datetime.now().isoformat(),
                "status":        "raw — needs curation"
            },
            "qa_pairs": all_pairs
        }

        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n[Generator] Done!")
        print(f"[Generator] Total pairs generated : {len(all_pairs)}")
        print(f"[Generator] Failed chunks         : {failed}")
        print(f"[Generator] Saved to              : {output_path}")

        return all_pairs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generator.py <path_to_pdf>")
        sys.exit(1)

    # Load and chunk PDF
    loader = PDFLoader(sys.argv[1])
    pages = loader.load()

    chunker = TokenChunker()
    chunks = chunker.chunk_pages(pages)

    # Generate QA pairs
    generator = QAGenerator()
    pairs = generator.generate(chunks)

    # Preview first 3
    print(f"\n--- Preview of first 3 QA pairs ---")
    for pair in pairs[:3]:
        print(f"\nID       : {pair['id']}")
        print(f"Type     : {pair['question_type']}")
        print(f"Page     : {pair['source_page']}")
        print(f"Question : {pair['question']}")
        print(f"Answer   : {pair['ground_truth'][:150]}...")