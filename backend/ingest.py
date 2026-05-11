"""PDF ingestion pipeline to create and store FAISS vector indexes."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv(Path(__file__).parent.parent / ".env")

VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "data/vector_stores"))
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def get_embeddings():
    """Returns the configured embedding model, cached for performance."""
    if not hasattr(get_embeddings, "_instance"):
        openai_key = os.getenv("OPENAI_API_KEY", "")

        if openai_key and openai_key != "your_openai_key_here":
            from langchain_openai import OpenAIEmbeddings
            get_embeddings._instance = OpenAIEmbeddings(model="text-embedding-ada-002")
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            get_embeddings._instance = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return get_embeddings._instance


def ingest_pdf(pdf_path: str, doc_id: str) -> dict:
    """Ingests a PDF file, splits it into chunks, and saves a FAISS vector index."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source"] = doc_id

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    
    store_path = VECTOR_STORE_DIR / doc_id
    vectorstore.save_local(str(store_path))

    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "pages": len(pages),
        "store_path": str(store_path),
    }


@lru_cache(maxsize=10)
def load_vectorstore(doc_id: str) -> FAISS:
    """Loads a previously saved FAISS index from disk, cached for rapid access."""
    store_path = VECTOR_STORE_DIR / doc_id

    if not store_path.exists():
        raise FileNotFoundError(
            f"No vector store found for doc_id='{doc_id}'. "
            "Please ingest the document first."
        )

    return FAISS.load_local(
        str(store_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def list_documents() -> List[str]:
    """Returns a list of all ingested document IDs."""
    if not VECTOR_STORE_DIR.exists():
        return []
    return sorted(d.name for d in VECTOR_STORE_DIR.iterdir() if d.is_dir())