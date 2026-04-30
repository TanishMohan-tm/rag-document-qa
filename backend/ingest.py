"""
ingest.py

WHAT THIS FILE DOES:
Turns a PDF file into a searchable FAISS vector index stored on disk.
This is the "ingestion pipeline" — runs once per document.

WHAT IT PRODUCES:
data/vector_stores/{doc_id}/index.faiss   ← the vectors
data/vector_stores/{doc_id}/index.pkl     ← text + page numbers
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
# MUST be called before any os.getenv() calls
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

# Where to save FAISS indexes on disk
# Path() creates a path object that works on Mac, Windows, Linux
# os.getenv("VECTOR_STORE_DIR", "data/vector_stores") reads the env var
# with "data/vector_stores" as the default if it's not set
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "data/vector_stores"))

# Create the directory if it doesn't exist
# parents=True: create parent dirs too (like mkdir -p)
# exist_ok=True: don't raise an error if it already exists
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def get_embeddings():
    """
    Return the embedding model to use.

    This function checks which API keys are configured and returns
    the appropriate embeddings model:
    - If OPENAI_API_KEY is set: use text-embedding-ada-002 (paid, ~$0.0001/1K tokens)
    - Otherwise: use all-MiniLM-L6-v2 via HuggingFace (FREE, runs on your CPU)

    The model converts text into vectors (lists of 384 numbers).
    All-MiniLM-L6-v2 is ~90MB and downloads once to ~/.cache/huggingface/
    """
    openai_key = os.getenv("OPENAI_API_KEY", "")

    # Check if a real OpenAI key is configured (not empty, not the placeholder)
    if openai_key and openai_key not in ("", "your_openai_key_here"):
        from langchain_openai import OpenAIEmbeddings
        print("Using OpenAI text-embedding-ada-002")
        return OpenAIEmbeddings(model="text-embedding-ada-002")

    # Default: HuggingFace — completely free, runs locally
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Using HuggingFace all-MiniLM-L6-v2 (free, local)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},      # Run on CPU (works on any Mac)
        encode_kwargs={"normalize_embeddings": True},  # Unit vectors → better cosine similarity
    )


def ingest_pdf(pdf_path: str, doc_id: str) -> dict:
    """
    Full ingestion pipeline: PDF file → FAISS index on disk.

    Arguments:
    pdf_path: path to the PDF file on disk, e.g. "data/uploads/paper_abc123.pdf"
    doc_id:   unique name for this document, e.g. "attention_paper_a3f7b291"

    Returns a dict with stats:
    {"doc_id": "...", "chunks": 87, "pages": 15, "store_path": "..."}
    """

    # ── STEP 1: Load PDF ──────────────────────────────────────────────
    # PyPDFLoader opens the PDF and reads each page as a separate Document
    # A Document has:
    #   .page_content = the text of that page (string)
    #   .metadata = {"page": 0, "source": "path/to/file.pdf"}
    #                page is 0-indexed (page 1 = index 0)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()   # Returns List[Document]

    print(f"Loaded {len(pages)} pages from {pdf_path}")

    # ── STEP 2: Split into chunks ─────────────────────────────────────
    # RecursiveCharacterTextSplitter splits text in this priority order:
    #   1. "\n\n"  → paragraph boundaries (best split point)
    #   2. "\n"    → line boundaries
    #   3. ". "    → sentence boundaries
    #   4. " "     → word boundaries (rarely needed)
    #   5. ""      → character boundaries (last resort)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # Maximum characters per chunk
        chunk_overlap=100,     # Overlap between adjacent chunks (prevents losing context)
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,   # Use character count (not token count)
    )

    chunks = splitter.split_documents(pages)

    print(f"Split into {len(chunks)} chunks")

    # ── STEP 3: Stamp metadata on every chunk ────────────────────────
    # Each chunk inherits metadata from its source page (including page number)
    # We add doc_id and source so the citation system knows which document
    # and which page each chunk came from
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source"] = doc_id  # used by chain.py for citations
        # chunk.metadata["page"] already exists from PyPDFLoader (0-indexed)

    # ── STEP 4: Embed and build FAISS index ──────────────────────────
    # This line does the heavy lifting:
    # 1. Calls embeddings.embed_documents([chunk.page_content for chunk in chunks])
    # 2. Gets back a list of 384-dimensional vectors
    # 3. Builds a FAISS index with those vectors + their metadata
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ── STEP 5: Save to disk ─────────────────────────────────────────
    # Creates two files:
    #   data/vector_stores/{doc_id}/index.faiss  ← the vectors
    #   data/vector_stores/{doc_id}/index.pkl    ← text + page numbers
    store_path = VECTOR_STORE_DIR / doc_id
    vectorstore.save_local(str(store_path))

    print(f"Saved FAISS index to {store_path}")

    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "pages": len(pages),
        "store_path": str(store_path),
    }


def load_vectorstore(doc_id: str) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Called by retriever.py every time a question needs to be answered.

    Raises FileNotFoundError if the document hasn't been ingested yet.
    """
    store_path = VECTOR_STORE_DIR / doc_id

    if not store_path.exists():
        raise FileNotFoundError(
            f"No vector store found for doc_id='{doc_id}'. "
            f"You need to ingest the document first via POST /ingest."
        )

    embeddings = get_embeddings()

    # allow_dangerous_deserialization=True is required by FAISS when loading .pkl files
    # This is safe here because WE created these files — they're not from an untrusted source
    return FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def list_documents() -> List[str]:
    """
    Return all document IDs that have been ingested.
    Used by the GET /documents endpoint.

    Returns a sorted list of folder names in data/vector_stores/
    Each folder name is a doc_id.
    """
    if not VECTOR_STORE_DIR.exists():
        return []
    return sorted(d.name for d in VECTOR_STORE_DIR.iterdir() if d.is_dir())