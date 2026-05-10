"""LLM processing chain for generating answers from retrieved context."""

import os
import time
from typing import Any, Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser


def get_llm():
    """Returns the best available LLM based on configured API keys."""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")

    if openai_key and openai_key != "your_openai_key_here":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    if groq_key and groq_key != "your_groq_key_here":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    raise EnvironmentError(
        "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
    )


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a precise document assistant. Answer the question using ONLY the context below.

Rules:
- If the context contains the answer, give a clear, direct response.
- If the context does NOT contain the answer, respond exactly: "The document does not contain enough information to answer this question."
- Do NOT use outside knowledge.
- Be concise but complete.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_context(docs: List[Document]) -> str:
    """Formats retrieved document chunks into a structured context string."""
    parts = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", 0)
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[Excerpt {i + 1} | Document: {source} | Page: {page + 1}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n" + "\n\n---\n\n".join(parts) + "\n"


def _build_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """Builds deduplicated source citations from retrieved chunks."""
    sources = []
    seen = set()

    for doc in docs:
        page = int(doc.metadata.get("page", 0))
        source = doc.metadata.get("source", "unknown")
        key = (source, page)

        if key not in seen:
            seen.add(key)
            excerpt = doc.page_content.strip()
            sources.append({
                "document": source,
                "page": page + 1,
                "excerpt": excerpt[:300] + "…" if len(excerpt) > 300 else excerpt,
            })

    return sources


def answer_question(docs: List[Document], question: str) -> Dict[str, Any]:
    """Generates an answer to the question using the retrieved context."""
    llm = get_llm()
    t0 = time.perf_counter()

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": _format_context(docs),
        "question": question,
    })

    latency = round(time.perf_counter() - t0, 3)

    return {
        "answer": answer.strip(),
        "sources": _build_sources(docs),
        "latency_seconds": latency,
    }