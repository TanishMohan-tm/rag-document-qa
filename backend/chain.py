"""
chain.py

WHAT THIS FILE DOES:
Takes retrieved document chunks + a question → asks an LLM → returns answer + citations.

THE FLOW:
List[Document] + question
    → format context string (chunks with page numbers)
    → fill RAG prompt template
    → send to LLM (Groq FREE / OpenAI optional)
    → parse response text
    → extract source citations from metadata
    → return {"answer": "...", "sources": [...], "latency_seconds": 0.8}
"""

import os
import time
from typing import Any, Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser


def get_llm():
    """
    Return the best available LLM based on configured API keys.

    Priority order:
    1. OpenAI GPT-3.5-turbo — if OPENAI_API_KEY is set (paid: ~$0.002/1K tokens)
    2. Groq llama3-8b-8192   — if GROQ_API_KEY is set  (FREE, recommended)

    Groq is the default. It's fast (~1 second) and free.
    Get your free key at: https://console.groq.com
    """
    openai_key = os.getenv("OPENAI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")

    if openai_key and openai_key not in ("", "your_openai_key_here"):
        from langchain_openai import ChatOpenAI
        print("LLM: Using OpenAI gpt-3.5-turbo")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    if groq_key and groq_key not in ("", "your_groq_key_here"):
        from langchain_groq import ChatGroq
        print("LLM: Using Groq llama-3.3-70b-versatile (free)")
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        # temperature=0 → deterministic answers, same question = same answer
        # For QA systems this is what you want.
        # For creative writing you'd use 0.7+

    raise EnvironmentError(
        "No LLM API key found.\n"
        "Set GROQ_API_KEY in your .env file.\n"
        "Get a free key at: https://console.groq.com"
    )


# ── THE RAG PROMPT ────────────────────────────────────────────────────────────
#
# This is the most important design decision in the entire system.
# It tells the LLM exactly what rules to follow.
#
# Key design choices:
#   1. "ONLY the context below" — prevents using training data (hallucination)
#   2. Exact fallback phrase — makes "I don't know" responses parseable/consistent
#   3. "Do NOT use outside knowledge" — explicit, not ambiguous
#   4. {context} contains page numbers → LLM sees them → we extract them for citations
#
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
    """
    Convert a list of Document chunks into a single formatted context string.

    Before: [Document(page_content="...", metadata={"page": 2, "source": "paper_abc"}), ...]

    After:
    [Excerpt 1 | Document: paper_abc | Page: 3]
    ...chunk text...

    ---

    [Excerpt 2 | Document: paper_abc | Page: 5]
    ...chunk text...

    Why format this way?
    - The LLM can see which page each excerpt came from
    - This feeds directly into source citations
    - The "---" separators help the LLM distinguish between excerpts
    """
    parts = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", 0)      # 0-indexed from PyPDFLoader
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[Excerpt {i + 1} | Document: {source} | Page: {page + 1}]\n"
            # page + 1 converts to 1-indexed (page 0 becomes "Page 1")
            f"{doc.page_content.strip()}"
        )
    return "\n\n" + "\n\n---\n\n".join(parts) + "\n"


def _build_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Build deduplicated source citations from retrieved chunks.

    Returns a list like:
    [
    {
        "document": "attention_paper_abc123",
        "page": 3,              ← 1-indexed for display
        "excerpt": "The Transformer model..."  ← first 300 chars of chunk
    },
    ...
    ]

    Deduplication: if 2 chunks came from page 3, we only cite page 3 once.
    """
    sources = []
    seen: set = set()   # tracks (document, page) pairs we've already added

    for doc in docs:
        page = int(doc.metadata.get("page", 0))
        source = doc.metadata.get("source", "unknown")
        key = (source, page)

        if key not in seen:
            seen.add(key)
            excerpt = doc.page_content.strip()
            sources.append({
                "document": source,
                "page": page + 1,  # convert to 1-indexed for display
                "excerpt": excerpt[:300] + "…" if len(excerpt) > 300 else excerpt,
            })

    return sources


def answer_question(docs: List[Document], question: str) -> Dict[str, Any]:
    """
    The main function. Full RAG chain from chunks to answer.

    Arguments:
    docs:     List of retrieved Document chunks from retriever.py
    question: The user's question as a string

    Returns:
    {
        "answer": "The paper proposes the Transformer architecture...",
        "sources": [{"document": "paper_abc", "page": 3, "excerpt": "..."}],
        "latency_seconds": 0.847
    }
    """
    llm = get_llm()
    t0 = time.perf_counter()   # start timer

    # Build the LCEL chain: prompt | llm | parser
    #   RAG_PROMPT: fills {context} and {question} placeholders
    #   llm: sends filled prompt to Groq, returns a response object
    #   StrOutputParser(): extracts the text string from the response object
    chain = RAG_PROMPT | llm | StrOutputParser()

    # Invoke the chain with our variables
    answer = chain.invoke({
        "context": _format_context(docs),   # formatted chunks with page numbers
        "question": question,               # the user's raw question
    })

    latency = round(time.perf_counter() - t0, 3)   # seconds, rounded to milliseconds

    return {
        "answer": answer.strip(),
        "sources": _build_sources(docs),
        "latency_seconds": latency,
    }