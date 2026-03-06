from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import (
    GROQ_API_KEY,
    LLM_MODEL,
    TOP_K,
    validate_phase2_settings
)
from phase2_production.retrieval.hybrid_retriever import HybridRetriever
from phase2_production.retrieval.reranker import Reranker, RerankedChunk
from phase2_production.generation.citation_enforcer import CitationEnforcer, CitationValidation


# ─────────────────────────────────────────
# Graph State — shared across all nodes
# ─────────────────────────────────────────
class RAGState(TypedDict):
    question: str                           # user's input question
    hybrid_candidates: List               # raw hybrid retrieval results
    reranked_chunks: List[RerankedChunk]  # after Cohere reranking
    context: str                           # formatted context for LLM
    raw_answer: str                        # LLM answer before validation
    validation: Optional[CitationValidation]  # citation validation result
    final_answer: str                      # final answer shown to user
    confidence: str                        # overall confidence level


# ─────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────
HIGH_CONFIDENCE_PROMPT = ChatPromptTemplate.from_template("""
You are an expert research assistant. Answer the question based ONLY on the provided context.

RULES:
1. Always answer — never say "I don't know" or refuse
2. After EVERY key claim add a citation: [Source: {source_file}, Page {page}, Para {para}]
3. Be precise and professional
4. Structure your answer clearly

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (with inline citations after every key claim):
""")

LOW_CONFIDENCE_PROMPT = ChatPromptTemplate.from_template("""
You are an expert research assistant. The retrieved context may only partially answer the question.

RULES:
1. Always answer using whatever information is available in the context
2. After EVERY key claim add a citation: [Source: {source_file}, Page {page}, Para {para}]
3. If context is insufficient, clearly note what is and isn't covered
4. Never refuse to answer — always provide the best possible answer from available context

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (with citations, noting any gaps in available information):
""")


# ─────────────────────────────────────────
# Node Functions
# ─────────────────────────────────────────
def format_context(chunks: List[RerankedChunk]) -> str:
    """Format reranked chunks into structured context for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"--- Chunk {i} (Confidence: {chunk.confidence}) ---\n"
            f"Source: {chunk.source_file}\n"
            f"Page: {chunk.page_number}\n"
            f"Para: {chunk.paragraph_index}\n"
            f"Relevance Score: {chunk.rerank_score}\n"
            f"Text:\n{chunk.text}\n"
        )
    return "\n".join(parts)


def get_source_info(chunks: List[RerankedChunk]) -> dict:
    """Get source file info for prompt template."""
    if chunks:
        return {
            "source_file": chunks[0].source_file,
            "page": "{page_number}",
            "para": "{paragraph_index}"
        }
    return {"source_file": "unknown", "page": "?", "para": "?"}


class RAGGraph:
    def __init__(self):
        validate_phase2_settings()

        # Initialize all components
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
        self.citation_enforcer = CitationEnforcer()
        self.llm = ChatGroq(
            model=LLM_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.1,
            max_tokens=1024
        )
        self.parser = StrOutputParser()

        # Build the graph
        self.graph = self._build_graph()
        print("[Graph] LangGraph RAG pipeline ready")

    # ── Node 1: Retrieve ──
    def node_retrieve(self, state: RAGState) -> RAGState:
        print("\n[Graph] Node 1: Hybrid Retrieval...")
        candidates = self.hybrid_retriever.retrieve(
            state["question"], top_k=20
        )
        state["hybrid_candidates"] = candidates
        return state

    # ── Node 2: Rerank ──
    def node_rerank(self, state: RAGState) -> RAGState:
        print("[Graph] Node 2: Cohere Re-ranking...")
        reranked = self.reranker.rerank(
            state["question"],
            state["hybrid_candidates"],
            top_k=TOP_K
        )
        state["reranked_chunks"] = reranked
        state["context"] = format_context(reranked)

        # Set confidence based on top chunk
        if reranked:
            state["confidence"] = reranked[0].confidence
        else:
            state["confidence"] = "low"
        return state

    # ── Node 3a: Generate (high confidence) ──
    def node_generate_high(self, state: RAGState) -> RAGState:
        print("[Graph] Node 3a: Generating answer (high confidence path)...")
        source_info = get_source_info(state["reranked_chunks"])
        chain = HIGH_CONFIDENCE_PROMPT | self.llm | self.parser
        answer = chain.invoke({
            "context": state["context"],
            "question": state["question"],
            "source_file": source_info["source_file"],
            "page": source_info["page"],
            "para": source_info["para"]
        })
        state["raw_answer"] = answer
        return state

    # ── Node 3b: Generate (low confidence) ──
    def node_generate_low(self, state: RAGState) -> RAGState:
        print("[Graph] Node 3b: Generating answer (low confidence path)...")
        source_info = get_source_info(state["reranked_chunks"])
        chain = LOW_CONFIDENCE_PROMPT | self.llm | self.parser
        answer = chain.invoke({
            "context": state["context"],
            "question": state["question"],
            "source_file": source_info["source_file"],
            "page": source_info["page"],
            "para": source_info["para"]
        })
        state["raw_answer"] = answer
        return state

    # ── Node 4: Validate Citations ──
    def node_validate(self, state: RAGState) -> RAGState:
        print("[Graph] Node 4: Validating citations...")
        validation = self.citation_enforcer.validate(
            state["raw_answer"],
            state["reranked_chunks"]
        )
        state["validation"] = validation
        state["final_answer"] = validation.validated_answer
        state["confidence"] = validation.confidence_flag
        return state

    # ── Conditional Edge: route based on confidence ──
    def route_by_confidence(self, state: RAGState) -> str:
        confidence = state.get("confidence", "low")
        if confidence == "high":
            print("[Graph] Routing → high confidence path")
            return "generate_high"
        else:
            print("[Graph] Routing → low confidence path")
            return "generate_low"

    # ── Build the Graph ──
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("rerank", self.node_rerank)
        workflow.add_node("generate_high", self.node_generate_high)
        workflow.add_node("generate_low", self.node_generate_low)
        workflow.add_node("validate", self.node_validate)

        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")

        # Conditional edge after rerank
        workflow.add_conditional_edges(
            "rerank",
            self.route_by_confidence,
            {
                "generate_high": "generate_high",
                "generate_low": "generate_low"
            }
        )

        # Both paths converge at validate
        workflow.add_edge("generate_high", "validate")
        workflow.add_edge("generate_low", "validate")
        workflow.add_edge("validate", END)

        return workflow.compile()

    def ask(self, question: str) -> dict:
        """Run the full RAG pipeline for a question."""
        initial_state = RAGState(
            question=question,
            hybrid_candidates=[],
            reranked_chunks=[],
            context="",
            raw_answer="",
            validation=None,
            final_answer="",
            confidence=""
        )

        final_state = self.graph.invoke(initial_state)

        return {
            "question":    final_state["question"],
            "answer":      final_state["final_answer"],
            "confidence":  final_state["confidence"],
            "sources":     final_state["reranked_chunks"],
            "validation":  final_state["validation"]
        }

    def display_response(self, response: dict) -> None:
        """Pretty print the full response."""
        print("\n" + "=" * 60)
        print("QUESTION:")
        print(response["question"])
        print("\n" + "-" * 60)
        print("ANSWER:")
        print(response["answer"])
        print("\n" + "-" * 60)
        print(f"SOURCES USED ({len(response['sources'])} chunks):")
        for i, chunk in enumerate(response["sources"], 1):
            print(
                f"  [{i}] {chunk.source_file} | "
                f"Page {chunk.page_number} | "
                f"Para {chunk.paragraph_index} | "
                f"Cohere: {chunk.rerank_score} ({chunk.confidence})"
            )
        print("=" * 60)


if __name__ == "__main__":
    graph = RAGGraph()

    questions = [
        "What is the main method proposed in this paper?",
        "How does the dark channel dehazing work?",
        "What datasets were used to evaluate the method?"
    ]

    for question in questions:
        response = graph.ask(question)
        graph.display_response(response)
        print()