import re
from dataclasses import dataclass
from typing import List, Tuple

from phase2_production.retrieval.reranker import RerankedChunk


# ─────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────
@dataclass
class CitationValidation:
    is_valid: bool                  # True if all citations check out
    total_citations: int            # how many citations found in answer
    valid_citations: int            # how many matched a real chunk
    invalid_citations: List[str]    # citations that didnt match any chunk
    missing_citations: bool         # True if sentences have no citation at all
    confidence_flag: str            # "high", "medium", "low", "uncited"
    validated_answer: str           # answer with validation markers added


# ─────────────────────────────────────────
# Citation Enforcer
# ─────────────────────────────────────────
class CitationEnforcer:
    def __init__(self):
        # Pattern to find citations like [Source: moac.pdf, Page 3, Para 1]
        self.citation_pattern = re.compile(
            r'\[Source:\s*([^,]+),\s*Page\s*(\d+),\s*Para\s*(\d+)\]',
            re.IGNORECASE
        )
        print("[CitationEnforcer] Ready")

    def _extract_citations(self, answer: str) -> List[Tuple[str, int, int]]:
        """
        Extract all citations from answer text.
        Returns list of (source_file, page_number, paragraph_index) tuples.
        """
        matches = self.citation_pattern.findall(answer)
        return [
            (source.strip(), int(page), int(para))
            for source, page, para in matches
        ]

    def _build_chunk_lookup(
        self,
        chunks: List[RerankedChunk]
    ) -> dict:
        """Build a lookup set of valid (source, page, para) tuples."""
        return {
            (chunk.source_file, chunk.page_number, chunk.paragraph_index)
            for chunk in chunks
        }

    def _check_uncited_sentences(self, answer: str) -> bool:
        """
        Check if any sentences are missing citations entirely.
        Splits answer into sentences and checks each one.
        """
        sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
        uncited = [
            s for s in sentences
            if not self.citation_pattern.search(s)
        ]
        return len(uncited) > 0

    def validate(
        self,
        answer: str,
        chunks: List[RerankedChunk]
    ) -> CitationValidation:
        """
        Validate all citations in the answer against the source chunks.

        Checks:
        1. Every citation references a real chunk (source, page, para)
        2. No sentences are completely uncited
        3. Overall confidence based on chunk scores
        """
        # Extract citations from answer
        found_citations = self._extract_citations(answer)
        valid_chunk_set = self._build_chunk_lookup(chunks)

        # Validate each citation
        invalid_citations = []
        for source, page, para in found_citations:
            if (source, page, para) not in valid_chunk_set:
                invalid_citations.append(
                    f"[Source: {source}, Page {page}, Para {para}]"
                )

        valid_count = len(found_citations) - len(invalid_citations)
        has_uncited = self._check_uncited_sentences(answer)

        # Determine overall confidence
        if not chunks:
            confidence = "uncited"
        elif invalid_citations:
            confidence = "low"
        elif has_uncited:
            confidence = "medium"
        else:
            # Use average rerank score of source chunks
            avg_score = sum(c.rerank_score for c in chunks) / len(chunks)
            if avg_score >= 0.7:
                confidence = "high"
            elif avg_score >= 0.4:
                confidence = "medium"
            else:
                confidence = "low"

        is_valid = len(invalid_citations) == 0 and not has_uncited

        # Add validation footer to answer
        validated_answer = self._add_validation_footer(
            answer, confidence, invalid_citations, has_uncited
        )

        return CitationValidation(
            is_valid=is_valid,
            total_citations=len(found_citations),
            valid_citations=valid_count,
            invalid_citations=invalid_citations,
            missing_citations=has_uncited,
            confidence_flag=confidence,
            validated_answer=validated_answer
        )

    def _add_validation_footer(
        self,
        answer: str,
        confidence: str,
        invalid_citations: List[str],
        has_uncited: bool
    ) -> str:
        """Add a validation summary footer to the answer."""
        footer_lines = ["\n\n---"]

        if confidence == "high":
            footer_lines.append("confidence: HIGH — all claims fully supported by source chunks")
        elif confidence == "medium":
            footer_lines.append("confidence: MEDIUM — most claims supported, some may lack citations")
        elif confidence == "low":
            footer_lines.append("confidence: LOW — some citations could not be verified")
        else:
            footer_lines.append("confidence: UNCITED — answer generated without source support")

        if invalid_citations:
            footer_lines.append(f"unverified citations: {', '.join(invalid_citations)}")

        if has_uncited:
            footer_lines.append("note: some sentences were generated without explicit citations")

        return answer + "\n".join(footer_lines)


if __name__ == "__main__":
    # Test with a sample answer
    enforcer = CitationEnforcer()

    sample_answer = """
    The paper proposes a multiscale optical attenuation compensation framework.
    [Source: moac.pdf, Page 2, Para 0]
    The method uses dark channel dehazing to enhance underwater images.
    [Source: moac.pdf, Page 3, Para 1]
    The system achieves state-of-the-art results on benchmark datasets.
    [Source: moac.pdf, Page 99, Para 5]
    """

    # Mock chunks
    from phase2_production.retrieval.reranker import RerankedChunk
    mock_chunks = [
        RerankedChunk(
            chunk_id="abc123",
            source_file="moac.pdf",
            page_number=2,
            paragraph_index=0,
            text="sample text",
            rerank_score=0.85,
            rrf_score=0.032,
            confidence="high"
        ),
        RerankedChunk(
            chunk_id="def456",
            source_file="moac.pdf",
            page_number=3,
            paragraph_index=1,
            text="sample text",
            rerank_score=0.72,
            rrf_score=0.028,
            confidence="high"
        ),
    ]

    validation = enforcer.validate(sample_answer, mock_chunks)

    print(f"\n--- Citation Validation Report ---")
    print(f"Valid           : {validation.is_valid}")
    print(f"Total citations : {validation.total_citations}")
    print(f"Valid citations : {validation.valid_citations}")
    print(f"Invalid         : {validation.invalid_citations}")
    print(f"Confidence      : {validation.confidence_flag}")
    print(f"\nValidated Answer:\n{validation.validated_answer}")