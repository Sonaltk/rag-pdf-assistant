from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

from config.settings import (
    GROQ_API_KEY,
    LLM_MODEL,
    TOP_K,
    validate_settings
)
from phase1_fundamentals.retrieval.retriever import Retriever, RetrievedChunk


# ─────────────────────────────────────────
# Citation-enforced prompt
# Forces LLM to always cite sources
# and never refuse to answer
# ─────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert research assistant. Your job is to answer questions
based ONLY on the provided context chunks from a PDF document.

STRICT RULES:
1. Always answer the question — never say "I don't know" or "I cannot answer"
2. Base your answer strictly on the context provided below
3. After EVERY key claim or sentence in your answer, add a citation like:
   [Source: {{source_file}}, Page {{page}}, Para {{para}}]
4. If the context only partially answers the question, answer what you can
   and note what is not covered
5. Keep your answer clear, structured and professional

CONTEXT CHUNKS:
{context}

QUESTION:
{question}

ANSWER (with inline citations after every key claim):
""")


# ─────────────────────────────────────────
# RAG Chain — full pipeline
# ─────────────────────────────────────────
class RAGChain:
    def __init__(self):
        validate_settings()

        # Groq LLM
        self.llm = ChatGroq(
            model=LLM_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.1,        # low temperature = more factual answers
            max_tokens=1024
        )

        # Retriever
        self.retriever = Retriever()

        # Output parser
        self.parser = StrOutputParser()

        print(f"[Chain] RAG chain initialized with model: {LLM_MODEL}")

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a structured context string
        that tells the LLM exactly where each piece of text came from.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"--- Chunk {i} ---"
                f"Source: {chunk.source_file}"
                f"Page: {chunk.page_number}"
                f"Paragraph: {chunk.paragraph_index}"
                f"Relevance Score: {chunk.similarity_score}"
                f"Text:{chunk.text}"
            )
        return " ".join(context_parts)

    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks
        2. Format as context
        3. Generate cited answer via LLM

        Returns a dict with answer + source chunks used
        """
        print(f"[Chain] Processing question: {question[:80]}...")

        # Step 1 — Retrieve
        chunks = self.retriever.retrieve(question, top_k=top_k)

        # Step 2 — Format context
        context = self._format_context(chunks)

        # Step 3 — Generate answer with citations
        print(f"[Chain] Sending {len(chunks)} chunks to {LLM_MODEL}...")
        chain = RAG_PROMPT | self.llm | self.parser
        answer = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "question":   question,
            "answer":     answer,
            "sources":    chunks,      # full chunk objects for reference
            "num_chunks": len(chunks)
        }

    def display_response(self, response: dict) -> None:
        """Pretty print the full response with sources."""
        print("" + "=" * 60)
        print("QUESTION:")
        print(response["question"])
        print("" + "-" * 60)
        print("ANSWER:")
        print(response["answer"])
        print(" " + "-" * 60)
        print(f"SOURCES USED ({response['num_chunks']} chunks):")
        for i, chunk in enumerate(response["sources"], 1):
            print(
                f"  [{i}] {chunk.source_file} | "
                f"Page {chunk.page_number} | "
                f"Para {chunk.paragraph_index} | "
                f"Score: {chunk.similarity_score}"
            )
        print("=" * 60)


# ─────────────────────────────────────────
# Quick test — run directly to verify
# python phase1_fundamentals/generation/chain.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    chain = RAGChain()

    # Test questions — relevant to moac.pdf
    questions = [
        "What is the main method proposed in this paper?",
        "How does the underwater image restoration work?",
        "What datasets were used to evaluate the method?"
    ]

    for question in questions:
        response = chain.ask(question)
        chain.display_response(response)
        print()