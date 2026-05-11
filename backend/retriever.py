"""Retriever module for finding relevant document chunks via FAISS."""

from typing import List, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from ingest import load_vectorstore


def retrieve(doc_id: str, query: str, k: int = 4) -> List[Document]:
    """Finds the k most relevant and diverse chunks for a given query using MMR."""
    vectorstore: FAISS = load_vectorstore(doc_id)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(k * 4, 30),
        },
    )

    return retriever.invoke(query)


def retrieve_with_scores(
    doc_id: str, query: str, k: int = 4
) -> List[Tuple[Document, float]]:
    """Returns the top k relevant chunks along with their L2 distance scores."""
    vectorstore: FAISS = load_vectorstore(doc_id)
    return vectorstore.similarity_search_with_score(query, k=k)