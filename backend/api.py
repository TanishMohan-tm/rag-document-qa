"""FastAPI application for the RAG Document QA backend."""

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")

from chain import answer_question, get_llm, RAG_PROMPT, _format_context, _build_sources
from ingest import ingest_pdf, list_documents
from retriever import retrieve, retrieve_with_scores

app = FastAPI(
    title="RAG Document QA",
    description="Upload PDFs, ask questions, and get answers with source citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_answer_cache: dict = {}


def _cache_key(doc_id: str, question: str, k: int) -> str:
    """Generate a unique cache key from request parameters."""
    raw = f"{doc_id}:{question.lower().strip()}:{k}"
    return hashlib.md5(raw.encode()).hexdigest()

class AskRequest(BaseModel):
    doc_id: str
    question: str
    k: int = 3


class EvalPair(BaseModel):
    question: str
    reference_answer: str


class EvaluateRequest(BaseModel):
    doc_id: str
    pairs: List[EvalPair]



def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Computes cosine similarity between two text strings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer, util as st_util

    if not hasattr(_semantic_similarity, "_model"):
        _semantic_similarity._model = SentenceTransformer("all-MiniLM-L6-v2")

    model = _semantic_similarity._model
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    return float(st_util.cos_sim(emb_a, emb_b).item())



@app.get("/", tags=["health"])
def root():
    """Health check endpoint."""
    return {"status": "ok", "api_docs": "/docs", "version": "1.0.0"}


@app.get("/documents", tags=["documents"])
def get_documents():
    """Returns a list of all ingested documents."""
    return {"documents": list_documents()}


@app.post("/ingest", tags=["documents"])
async def ingest(file: UploadFile = File(...)):
    """Uploads and indexes a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf files are accepted. Please upload a PDF."
        )

    stem = Path(file.filename).stem[:40].replace(" ", "_")
    doc_id = f"{stem}_{uuid.uuid4().hex[:8]}"
    tmp_path = UPLOAD_DIR / f"{doc_id}.pdf"

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = ingest_pdf(str(tmp_path), doc_id)
        return {"status": "success", **result}
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(exc)}"
        ) from exc


@app.post("/ask", tags=["qa"])
def ask(req: AskRequest):
    """Processes a question about an ingested document and returns an answer."""
    key = _cache_key(req.doc_id, req.question, req.k)
    if key in _answer_cache:
        cached = _answer_cache[key].copy()
        cached["cached"] = True
        return cached

    try:
        docs = retrieve(req.doc_id, req.question, k=req.k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not docs:
        return {
            "answer": "No relevant content found in the document for this question.",
            "sources": [],
            "latency_seconds": 0.0,
        }

    try:
        result = answer_question(docs, req.question)

        if len(_answer_cache) >= 100:
            oldest_key = next(iter(_answer_cache))
            del _answer_cache[oldest_key]
        _answer_cache[key] = result

        return result
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask-stream", tags=["qa"])
async def ask_stream(req: AskRequest):
    """Streaming version of /ask returning a text/plain stream of tokens."""
    try:
        docs = retrieve(req.doc_id, req.question, k=req.k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not docs:
        async def empty():
            yield "No relevant content found in the document for this question."
        return StreamingResponse(empty(), media_type="text/plain")

    from langchain_core.output_parsers import StrOutputParser

    try:
        llm = get_llm()
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    chain = RAG_PROMPT | llm | StrOutputParser()

    async def token_generator():
        async for token in chain.astream({
            "context": _format_context(docs),
            "question": req.question,
        }):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.post("/evaluate", tags=["evaluation"])
def evaluate(req: EvaluateRequest):
    """Evaluates the RAG pipeline quality across a batch of questions."""
    if not req.pairs:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one question-answer pair in 'pairs'."
        )

    per_question = []

    for pair in req.pairs:
        try:
            docs_scores = retrieve_with_scores(req.doc_id, pair.question, k=4)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        docs = [d for d, _ in docs_scores]
        l2_distances = [float(s) for _, s in docs_scores]

        try:
            result = answer_question(docs, pair.question)
        except EnvironmentError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        answer_relevance = _semantic_similarity(
            result["answer"],
            pair.reference_answer
        )

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