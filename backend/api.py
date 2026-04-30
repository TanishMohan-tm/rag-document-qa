"""
api.py

WHAT THIS FILE DOES:
The FastAPI HTTP server. This is what runs on localhost:8000.
It exposes 5 endpoints that the Streamlit frontend calls.

ENDPOINTS:
GET  /              → health check
GET  /documents     → list all ingested doc IDs
POST /ingest        → upload PDF → returns doc_id
POST /ask           → {doc_id, question} → {answer, sources, latency}
POST /evaluate      → batch {question, reference_answer} pairs → quality scores

HOW TO RUN:
cd backend
uvicorn api:app --reload --port 8000
Then open http://localhost:8000/docs for interactive Swagger UI
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# CRITICAL: load_dotenv() must run BEFORE importing our modules
# because chain.py and ingest.py call os.getenv() at import time
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

from chain import answer_question
from ingest import ingest_pdf, list_documents
from retriever import retrieve, retrieve_with_scores


# ── FastAPI App Setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Document QA",
    description="Upload any PDF, ask questions, get answers with source citations.",
    version="1.0.0",
    # Swagger UI available at /docs — auto-generated from your code
    # ReDoc available at /redoc
)

# CORS — allows the Streamlit frontend (on port 8501) to call this API (on port 8000)
# Without CORS, browsers block cross-origin requests as a security measure
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # In production, restrict this to your frontend's domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Where to save uploaded PDFs
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Request / Response Models ─────────────────────────────────────────────────
# These Pydantic models define what the request body must look like.
# FastAPI validates incoming requests against these automatically.

class AskRequest(BaseModel):
    doc_id: str       # which document to query (returned by /ingest)
    question: str     # the user's natural language question
    k: int = 4        # number of chunks to retrieve (default 4, max ~8 is reasonable)


class EvalPair(BaseModel):
    question: str           # the question to test
    reference_answer: str   # the expected/correct answer for comparison


class EvaluateRequest(BaseModel):
    doc_id: str             # which document to evaluate against
    pairs: List[EvalPair]   # list of question-answer pairs to test


# ── Helper Functions ──────────────────────────────────────────────────────────

def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two text strings.
    Used by /evaluate to measure how close the generated answer is to the reference.

    Returns a float between 0 and 1:
    1.0 = identical meaning
    0.7 = roughly same idea
    0.3 = loosely related
    0.0 = completely different

    Uses sentence-transformers locally — NO API CALL, no cost.
    The model is cached after first load (module-level attribute trick).
    """
    from sentence_transformers import SentenceTransformer, util as st_util

    # Cache the model as a function attribute after first load
    # Without this, it would reload from disk on every call (slow)
    if not hasattr(_semantic_similarity, "_model"):
        _semantic_similarity._model = SentenceTransformer("all-MiniLM-L6-v2")

    model = _semantic_similarity._model
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    return float(st_util.cos_sim(emb_a, emb_b).item())


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    """
    Health check endpoint.
    Railway and Docker use this to verify the server is running.
    Returns: {"status": "ok", "api_docs": "/docs", "version": "1.0.0"}
    """
    return {"status": "ok", "api_docs": "/docs", "version": "1.0.0"}


@app.get("/documents", tags=["documents"])
def get_documents():
    """
    List all ingested documents.
    The Streamlit frontend uses this to populate the "select existing" dropdown.
    Returns: {"documents": ["paper_abc123", "manual_def456", ...]}
    """
    return {"documents": list_documents()}


@app.post("/ingest", tags=["documents"])
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF and index it.

    1. Validates it's actually a PDF
    2. Saves it to data/uploads/
    3. Runs the full ingestion pipeline (ingest.py)
    4. Returns the doc_id you'll use in /ask

    Note: 'async def' because file upload uses async I/O
    UploadFile streams the file in chunks — doesn't load whole PDF into RAM
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        # HTTPException converts to proper HTTP error response
        # status_code=400 means "Bad Request" (client sent wrong data)
        raise HTTPException(
            status_code=400,
            detail="Only .pdf files are accepted. Please upload a PDF."
        )

    # Create a unique doc_id from the filename + random suffix
    # Example: "attention_paper_a3f7b291"
    # Path(file.filename).stem strips the .pdf extension
    # [:40] prevents extremely long filenames from causing issues
    # .replace(" ", "_") prevents spaces in the folder name
    # uuid.uuid4().hex[:8] adds 8 random hex chars to make it unique
    stem = Path(file.filename).stem[:40].replace(" ", "_")
    doc_id = f"{stem}_{uuid.uuid4().hex[:8]}"

    # Save the uploaded file to disk
    tmp_path = UPLOAD_DIR / f"{doc_id}.pdf"

    try:
        # shutil.copyfileobj streams the file in chunks
        # This is memory-efficient — a 50MB PDF won't crash the server
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run ingestion pipeline
        result = ingest_pdf(str(tmp_path), doc_id)

        return {"status": "success", **result}
        # **result unpacks the dict: {"doc_id": ..., "chunks": ..., "pages": ..., "store_path": ...}

    except Exception as exc:
        # If anything goes wrong, clean up the partially-saved file
        if tmp_path.exists():
            tmp_path.unlink()   # delete the file

        # status_code=500 means "Internal Server Error" (something broke on the server side)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(exc)}"
        ) from exc


@app.post("/ask", tags=["qa"])
def ask(req: AskRequest):
    """
    Ask a question about an ingested document.

    Flow:
    1. Retrieve top-k chunks from FAISS (retriever.py)
    2. Generate answer using LLM with those chunks as context (chain.py)
    3. Return answer + page citations + latency

    Returns:
    {
    "answer": "The Transformer architecture...",
    "sources": [{"document": "paper_abc", "page": 3, "excerpt": "..."}],
    "latency_seconds": 0.847
    }
    """
    # Retrieve relevant chunks
    try:
        docs = retrieve(req.doc_id, req.question, k=req.k)
    except FileNotFoundError as exc:
        # status_code=404 means "Not Found" — document hasn't been ingested
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Handle case where no relevant chunks were found
    if not docs:
        return {
            "answer": "No relevant content found in the document for this question.",
            "sources": [],
            "latency_seconds": 0.0,
        }

    # Generate answer
    try:
        result = answer_question(docs, req.question)
        return result
    except EnvironmentError as exc:
        # status_code=503 means "Service Unavailable" — LLM not configured
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluate", tags=["evaluation"])
def evaluate(req: EvaluateRequest):
    """
    Batch evaluation of your RAG pipeline quality.

    For each (question, reference_answer) pair, this endpoint:
    1. Retrieves chunks for the question
    2. Generates an answer
    3. Measures: answer relevance, retrieval precision, latency

    ALL SCORING IS LOCAL — no API calls, no cost.
    Uses sentence-transformers cosine similarity.

    Metrics explained:
    - answer_relevance_score (0-1): semantic similarity between
    generated answer and your reference answer.
    0.8+ is good. Below 0.5 suggests the answer is off-topic.

    - retrieval_precision (0-1): how relevant the retrieved chunks were.
    Computed as 1/(1+L2_distance) — normalized L2 distance from FAISS.
    0.7+ is good. Low values suggest the document lacks relevant content.

    - latency_seconds: total time from question to answer.
    """
    if not req.pairs:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one question-answer pair in 'pairs'."
        )

    per_question = []

    for pair in req.pairs:
        # Step 1: Retrieve with scores (we need the raw distances for precision metric)
        try:
            docs_scores = retrieve_with_scores(req.doc_id, pair.question, k=4)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        docs = [d for d, _ in docs_scores]
        l2_distances = [float(s) for _, s in docs_scores]

        # Step 2: Generate answer and measure latency
        try:
            result = answer_question(docs, pair.question)
        except EnvironmentError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        # Step 3: Compute answer relevance (cosine similarity, 0-1)
        answer_relevance = _semantic_similarity(
            result["answer"],
            pair.reference_answer
        )

        # Step 4: Compute retrieval precision
        # Convert L2 distances to similarity scores: 1/(1+distance)
        # L2=0 → score=1.0 (perfect match)
        # L2=1 → score=0.5
        # L2=2 → score=0.33
        retrieval_scores = [1.0 / (1.0 + d) for d in l2_distances]
        avg_retrieval_precision = float(np.mean(retrieval_scores))

        per_question.append({
            "question": pair.question,
            "generated_answer": result["answer"],
            "reference_answer": pair.reference_answer,
            "answer_relevance_score": round(answer_relevance, 4),
            "retrieval_precision": round(avg_retrieval_precision, 4),
            "latency_seconds": result["latency_seconds"],
            "sources": result["sources"],
        })

    # Aggregate summary across all questions
    summary = {
        "avg_answer_relevance": round(
            float(np.mean([r["answer_relevance_score"] for r in per_question])), 4
        ),
        "avg_retrieval_precision": round(
            float(np.mean([r["retrieval_precision"] for r in per_question])), 4
        ),
        "avg_latency_seconds": round(
            float(np.mean([r["latency_seconds"] for r in per_question])), 4
        ),
        "total_questions": len(per_question),
    }

    return {"summary": summary, "per_question": per_question}