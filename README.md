# 🔍 RAG PDF Assistant

A production-quality **Multimodal RAG (Retrieval Augmented Generation)** system that answers questions about PDF documents with cited sources, figure citations, and full conversation history. Built in 4 phases — from a basic pipeline to a full-stack multi-user application.



---

## 📋 Table of Contents

- [Overview](#overview)
- [Evolution — Phase by Phase](#evolution)
- [System Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [Tech Stack](#tech-stack)

---

## Overview

This system takes any PDF document, understands it deeply — including text AND figures — and answers questions about it, telling you exactly which page, paragraph, or figure the answer came from. Built progressively across 4 phases, each adding production-grade capabilities on top of the previous.

**What makes it different from a basic RAG tutorial:**
- Hybrid BM25 + vector retrieval (not just vector search)
- Cohere cross-encoder re-ranking (industry standard two-stage retrieval)
- Multimodal — LLaVA vision model describes PDF figures and makes them searchable
- Full user system — login, per-user document library, conversation history
- Quantified with RAGAS evaluation metrics

---

## Evolution — Phase by Phase

### 🔵 Phase 1 — Fundamentals
> *"Get a basic pipeline working end to end"*

Built the core RAG pipeline from scratch:
- **PDF Loading** — PyMuPDF extracts text preserving page structure
- **Chunking** — tiktoken splits text into 512-token chunks with 100-token overlap
- **Embeddings** — HuggingFace `all-MiniLM-L6-v2` converts chunks to 384-dim vectors
- **Vector Store** — ChromaDB stores and searches vectors with cosine similarity
- **Generation** — Groq LLM (llama-3.1-8b) generates cited answers

```
PDF → Loader → Chunker → Embedder → ChromaDB
                                         ↓
Question → Retriever (cosine similarity) → Chain (Groq) → Cited Answer
```

**Result:** Working Q&A on any PDF with page + paragraph citations.

---

### 🟡 Phase 2 — Production Quality
> *"Make retrieval smarter and pipeline more intelligent"*

Identified Phase 1's weakness: vector-only search misses exact keyword matches (author names, model names, specific metrics). Added:

- **BM25 Retrieval** — keyword-based search using rank-bm25
- **Hybrid Retrieval** — RRF (Reciprocal Rank Fusion) combines BM25 + vector scores
- **Cohere Re-ranking** — cross-encoder scores (question, chunk) pairs for precision
- **LangGraph Pipeline** — stateful conditional routing based on confidence
- **Citation Enforcer** — post-generation validation of every citation

```
Question
   ↓
BM25 (keywords) ──┐
Vector (semantic)─┤→ RRF Fusion (33 candidates)
                  ↓
           Cohere Re-ranker (top 5, scored)
                  ↓
           LangGraph (high/low confidence routing)
                  ↓
           Groq LLM → Citation Enforcer → Answer
```

**Result:** Context Precision improved **+30.2%**, Answer Relevancy **+26.4%** over Phase 1.

---

### 🟢 Phase 3 — Evaluation
> *"Prove with numbers that the system works"*

Built a complete evaluation framework:
- **Golden Dataset** — auto-generated 54 curated QA pairs from moac.pdf using Groq
- **RAGAS Metrics** — Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Comparison Report** — Phase 1 vs Phase 2 across all metrics

| Metric | Phase 1 | Phase 2 | Change |
|---|---|---|---|
| Faithfulness | 0.9543 | 0.8472 | -11.2% (expected — longer answers) |
| Answer Relevancy | 0.7756 | 0.9800 | **+26.4%** ✅ |
| Context Precision | 0.7339 | 0.9554 | **+30.2%** ✅ |
| Context Recall | 0.8333 | 0.9500 | **+14.0%** ✅ |
| **Average** | **0.8243** | **0.9282** | **+12.6%** ✅ |

---

### 🟣 Phase 4 — Multimodal + Full-Stack
> *"Extend to figures, add user system, build production UI"*

Extended the system with four major additions:

**Multimodal RAG:**
- PyMuPDF extracts all figures from PDF pages
- LLaVA vision model (via Ollama) generates detailed descriptions of each figure
- Figure descriptions embedded as chunks alongside text — same ChromaDB collection
- Retrieval now finds both text and figure chunks
- Answer cites `[Page 3, Figure 1]` with expandable image preview

**User System:**
- Registration and login with bcrypt password hashing
- Per-user isolated ChromaDB collections (`rag_{user_id}_{pdf_stem}`)
- Multiple documents per user — switch instantly without re-ingesting
- Full conversation history saved to SQLite database

**Streamlit Dashboard:**
- Upload page — drag and drop PDF, real-time ingestion progress
- Chat page — ChatGPT-style interface with confidence badges and figure previews
- Analytics page — response time charts, confidence distribution, RAGAS comparison
- Home page — personalized dashboard with document and conversation stats

```
User Login
    ↓
Document Library (per user)
    ↓
Upload PDF → PyMuPDF (text) + LLaVA (figures) → ChromaDB (per user)
    ↓
Chat Interface
    ↓
Phase 2 Pipeline + Figure Retrieval
    ↓
Answer with Text Citations + Figure Images
    ↓
Saved to SQLite (conversation history)
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│   Login │ Document Library │ Chat │ Analytics           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  RAG Pipeline                            │
│                                                          │
│  Question                                                │
│     ↓                                                    │
│  BM25 Retriever ──┐                                      │
│  Vector Retriever─┤→ RRF Fusion → Cohere Re-ranker      │
│                   ↓                                      │
│           LangGraph (conditional routing)                │
│           ├── HIGH confidence → standard prompt          │
│           └── LOW  confidence → careful prompt           │
│                   ↓                                      │
│           Groq LLM → Citation Enforcer                   │
│                   ↓                                      │
│           Answer + Sources (text + figures)              │
└──────────────────────┬──────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
ChromaDB          SQLite DB           BM25 Index
(per-user         (users,             (per-document
 vectors)          docs, msgs)         JSON file)
```

---

## Features

```
✅ Hybrid retrieval    BM25 + vector search with RRF fusion
✅ Cross-encoder       Cohere reranking for precision
✅ Multimodal          PDF figures indexed via LLaVA vision model
✅ Figure citations    Expandable image preview in chat
✅ Confidence routing  LangGraph routes by HIGH/MEDIUM/LOW confidence
✅ Citation enforcement Post-generation validation of every source
✅ User auth           Register/login with bcrypt hashing
✅ Multi-document      Per-user isolated vector collections
✅ Conversation history Full Q&A history saved and resumable
✅ Analytics           Response time, confidence, RAGAS charts
✅ RAGAS evaluation    Quantified metrics across 54 QA pairs
```

---

## Project Structure

```
rag-pdf-assistant/
│
├── config/
│   └── settings.py                      # all config, API keys, thresholds
│
├── phase1_fundamentals/                 # Phase 1 — Basic RAG
│   ├── ingestion/
│   │   ├── loader.py                    # PDF → pages (PyMuPDF)
│   │   ├── chunker.py                   # pages → 512-token chunks
│   │   └── embedder.py                  # chunks → vectors → ChromaDB
│   ├── retrieval/
│   │   └── retriever.py                 # cosine similarity search
│   ├── generation/
│   │   └── chain.py                     # LangChain RAG + citations
│   └── main.py                          # CLI: ingest + ask
│
├── phase2_production/                   # Phase 2 — Production RAG
│   ├── retrieval/
│   │   ├── bm25_retriever.py            # BM25 keyword search
│   │   ├── vector_retriever.py          # semantic vector search
│   │   ├── hybrid_retriever.py          # RRF fusion
│   │   └── reranker.py                  # Cohere cross-encoder
│   ├── generation/
│   │   ├── graph.py                     # LangGraph pipeline
│   │   └── citation_enforcer.py         # citation validator
│   └── main.py                          # CLI: ingest + ask + compare
│
├── phase3_evaluation/                   # Phase 3 — Evaluation
│   ├── golden_dataset/
│   │   ├── generator.py                 # auto-generate QA pairs
│   │   ├── curator.py                   # filter + clean dataset
│   │   └── dataset.json                 # 54 curated QA pairs
│   ├── eval/
│   │   ├── ragas_eval.py                # RAGAS metrics runner
│   │   ├── report.py                    # comparison report
│   │   └── scores.json                  # evaluation results
│   └── main.py                          # CLI: generate + evaluate + report
│
├── phase4_production/                   # Phase 4 — Multimodal
│   ├── multimodal/
│   │   ├── figure_extractor.py          # PDF → figures (PyMuPDF)
│   │   ├── vision_chain.py              # LLaVA figure descriptions
│   │   └── multimodal_pipeline.py       # connects figures to Phase 2
│   ├── optimization/
│   │   └── cache.py                     # Redis query caching
│   └── monitoring/
│       └── mlflow_tracker.py            # experiment tracking
│
├── database/                            # SQLite / PostgreSQL
│   ├── models.py                        # SQLAlchemy tables
│   ├── connection.py                    # DB connection manager
│   └── queries.py                       # CRUD operations
│
├── auth/
│   └── auth.py                          # login, register, sessions
│
├── app/                                 # Streamlit Dashboard
│   ├── main.py                          # home page + user stats
│   └── pages/
│       ├── 0_login.py                   # login + register
│       ├── 1_documents.py               # document library
│       ├── 2_chat.py                    # Q&A chat interface
│       └── 3_analytics.py              # metrics dashboard
│
├── data/                                # gitignored — regenerate locally
│   ├── raw/                             # source PDFs
│   ├── chroma_db/                       # vector embeddings
│   ├── chunks/                          # BM25 indexes
│   └── figures/                         # extracted PDF figures
│
├── requirements.txt
├── setup.py
└── .env.example
```

---

## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) (for local LLaVA vision model)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Sonaltk/rag-pdf-assistant.git
cd rag-pdf-assistant
```

### 2. Create virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Configure API keys
```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_key          # free at console.groq.com
COHERE_API_KEY=your_cohere_key      # free at dashboard.cohere.com
```

### 5. Install and start Ollama (for multimodal)
```bash
# Mac
brew install ollama
ollama pull llava               # ~4.5GB — vision model
ollama serve                    # keep running in separate terminal
```

### 6. Run the app
```bash
python -m streamlit run app/main.py
```

---

## Usage

### Streamlit Dashboard (recommended)

```bash
# Terminal 1 — keep Ollama running
ollama serve

# Terminal 2 — run app
python -m streamlit run app/main.py
```

1. **Register** a new account on the Login page
2. **Upload** any PDF on the Documents page
3. **Ask** questions in the Chat page
4. **View** response times and RAGAS scores in Analytics

### CLI (Phase 1)

```bash
python phase1_fundamentals/main.py ingest data/raw/your_doc.pdf
python phase1_fundamentals/main.py ask -q "What is the main method?"
python phase1_fundamentals/main.py ask    # interactive mode
```

### CLI (Phase 2)

```bash
python phase2_production/main.py ingest data/raw/your_doc.pdf
python phase2_production/main.py ask -q "What is the main method?"
python phase2_production/main.py compare -q "What is the main method?"
```

### CLI (Phase 3 — Evaluation)

```bash
python phase3_evaluation/main.py generate data/raw/your_doc.pdf
python phase3_evaluation/main.py curate
python phase3_evaluation/main.py evaluate --pipeline both --max-pairs 10
python phase3_evaluation/main.py report
```

---

## Evaluation Results

Evaluated on `moac.pdf` (18 pages, 38 chunks) using 54 curated QA pairs.

| Metric | Phase 1 | Phase 2 | Improvement |
|---|---|---|---|
| Faithfulness | 0.9543 | 0.8472 | Expected — Phase 2 generates longer, more detailed answers |
| Answer Relevancy | 0.7756 | 0.9800 | **+26.4%** ✅ |
| Context Precision | 0.7339 | 0.9554 | **+30.2%** ✅ |
| Context Recall | 0.8333 | 0.9500 | **+14.0%** ✅ |
| **Average** | **0.8243** | **0.9282** | **+12.6%** ✅ |

**Key insight:** The 30.2% improvement in Context Precision confirms that Cohere cross-encoder re-ranking successfully filters out irrelevant chunks — Phase 1 would blindly use all top-5 chunks regardless of relevance.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| LLM | Groq (free API) | Answer generation |
| Vision | LLaVA via Ollama (local) | Figure description |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Text → vectors (local, free) |
| Vector Store | ChromaDB | Vector search (per-user collections) |
| Keyword Search | rank-bm25 | BM25 retrieval |
| Re-ranking | Cohere rerank-english-v3.0 | Cross-encoder scoring |
| Orchestration | LangChain + LangGraph | Pipeline + conditional routing |
| PDF Parsing | PyMuPDF | Text + figure extraction |
| Tokenization | tiktoken | Accurate token counting |
| Evaluation | RAGAS | RAG quality metrics |
| Database | SQLite (dev) / PostgreSQL (prod) | User data + conversation history |
| Auth | bcrypt | Password hashing |
| UI | Streamlit | Dashboard |
| Version Control | Git + GitHub | Source control |

---

## Notes

- `data/` is gitignored — run ingestion locally to regenerate
- `.env` is gitignored — never commit API keys
- ChromaDB collections are per-user: `rag_{user_id}_{pdf_stem}`
- BM25 indexes are per-document: `data/chunks/bm25_{collection}.json`
- Ollama must be running (`ollama serve`) for figure extraction
- Switch from SQLite to PostgreSQL by setting `DATABASE_URL` env var