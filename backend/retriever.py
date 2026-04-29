"""
retriever.py

WHAT THIS FILE DOES:
Takes a user's question and finds the most relevant chunks from a FAISS index.

KEY CONCEPT — MMR (Maximal Marginal Relevance):
Standard similarity search returns the top-4 most similar chunks.
Problem: they might all be near-duplicates (same paragraph, slightly different wording).

MMR solves this: find chunks that are both RELEVANT to the question
AND DIFFERENT from each other.

Example: Question about "attention mechanism"
Standard search might return:
    Chunk 1: "Attention allows the model to..."
    Chunk 2: "The model uses attention to..."   ← basically same as 1
    Chunk 3: "Attention mechanisms enable..."   ← basically same as 1
    Chunk 4: "We use attention for..."          ← basically same as 1

MMR returns:
    Chunk 1: "Attention allows the model to..."          ← relevant
    Chunk 2: "Multi-head attention uses 8 parallel..."   ← relevant AND different
    Chunk 3: "The query-key-value formulation..."        ← relevant AND different
    Chunk 4: "We compare attention to recurrence..."     ← relevant AND different

4 different angles on the answer = much better context for the LLM.
"""

from typing import List, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# Import from our own ingest.py (same backend/ folder)
from ingest import load_vectorstore


def retrieve(doc_id: str, query: str, k: int = 4) -> List[Document]:
    """
    Find the k most relevant AND diverse chunks for a given query.

    Arguments:
    doc_id: which document to search (must have been ingested)
    query:  the user's question as plain text
    k:      how many chunks to return (default 4)

    Returns List[Document] — each Document has:
    .page_content = the chunk text
    .metadata = {"page": 0, "source": "doc_id", "doc_id": "doc_id"}
    """
    # Load the FAISS index from disk
    vectorstore: FAISS = load_vectorstore(doc_id)

    # Convert to a LangChain Retriever object
    # search_type="mmr" activates Maximal Marginal Relevance
    # fetch_k: how many candidates to consider before MMR selection
    #          more candidates = better diversity, but slower
    #          rule of thumb: fetch_k = k * 3
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,                        # final number to return
            "fetch_k": min(k * 3, 20),     # candidates to consider
        },
    )

    # retriever.invoke() embeds the query and searches FAISS
    return retriever.invoke(query)


def retrieve_with_scores(
    doc_id: str, query: str, k: int = 4
) -> List[Tuple[Document, float]]:
    """
    Same as retrieve() but also returns the L2 distance score for each chunk.

    Used ONLY by the /evaluate endpoint to measure retrieval quality.
    Not used in the normal question-answering flow.

    Returns List of (Document, score) tuples.
    Score is L2 distance: lower = more similar = better retrieval.
    """
    vectorstore: FAISS = load_vectorstore(doc_id)
    return vectorstore.similarity_search_with_score(query, k=k)