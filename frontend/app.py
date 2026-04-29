"""
app.py

WHAT THIS FILE DOES:
The Streamlit web interface. Runs on localhost:8501.
Makes HTTP calls to the FastAPI backend on localhost:8000.

This file contains ONLY UI code.
No PDF processing, no embeddings, no LLM calls happen here.
Everything goes through the API.

HOW TO RUN:
cd frontend
streamlit run app.py
Then open http://localhost:8501
"""

import os
import json

import requests
import streamlit as st

# Where the backend API lives
# In local dev: http://localhost:8000
# In Docker: http://backend:8000 (Docker's internal network uses service names)
# On Railway: https://your-backend.up.railway.app (set via environment variable)
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# ── Page Configuration ────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script
st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📄",
    layout="wide",           # Use full browser width
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.source-card {
    background: #f8f9fa;
    border-left: 3px solid #4A90D9;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 4px;
    font-size: 0.85em;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
# The sidebar stays visible regardless of which tab is active
with st.sidebar:
    st.title("📄 RAG Document QA")
    st.caption("LangChain · FAISS · Groq (free) · HuggingFace")
    st.divider()

    # ── SECTION 1: Upload a new PDF ───────────────────────────────────
    st.subheader("1️⃣  Upload a PDF")

    # file_uploader returns None if no file is chosen, or a file-like object if one is
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed"  # hides the "Choose a PDF" label (we have the subheader)
    )

    if uploaded_file is not None:
        # Only show the ingest button if a file has been chosen
        if st.button("⚙️ Ingest Document", type="primary", use_container_width=True):
            # st.spinner shows a loading animation while the block runs
            with st.spinner("Processing PDF... (first run downloads ~90MB model)"):
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest",
                        files={
                            "file": (
                                uploaded_file.name,   # original filename
                                uploaded_file.getvalue(),  # file bytes
                                "application/pdf",    # MIME type
                            )
                        },
                        timeout=300,  # 5 minutes — large PDFs take time
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        # Save the doc_id to session_state for use in /ask requests
                        st.session_state["doc_id"] = data["doc_id"]
                        st.session_state["messages"] = []  # clear old chat
                        st.success(
                            f"✅ Ready!\n\n"
                            f"**{data['pages']} pages** split into "
                            f"**{data['chunks']} chunks**"
                        )
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot reach the backend.\n\n"
                        "Make sure it's running:\n"
                        "```\ncd backend\nuvicorn api:app --port 8000\n```"
                    )

    st.divider()

    # ── SECTION 2: Select an already-ingested document ────────────────
    st.subheader("2️⃣  Or select existing")
    try:
        docs_resp = requests.get(f"{API_URL}/documents", timeout=5)
        if docs_resp.status_code == 200:
            doc_list = docs_resp.json().get("documents", [])
            if doc_list:
                chosen = st.selectbox(
                    "Previously ingested documents",
                    ["— select a document —"] + doc_list,
                    label_visibility="collapsed"
                )
                if chosen != "— select a document —":
                    if st.button("📂 Load this document", use_container_width=True):
                        st.session_state["doc_id"] = chosen
                        st.session_state["messages"] = []
                        st.success(f"Loaded `{chosen}`")
            else:
                st.info("No documents yet. Upload one above.")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Backend offline — upload/select won't work.")

    # Show currently active document
    if "doc_id" in st.session_state:
        st.divider()
        st.caption(f"**Active document:**")
        st.code(st.session_state["doc_id"], language=None)


# ── Main Area ─────────────────────────────────────────────────────────────────

# Show welcome screen if no document is loaded
if "doc_id" not in st.session_state:
    st.markdown("""
    ## 👋 Welcome to RAG Document QA

    This system lets you have a **natural language conversation with any PDF document**.

    ### How it works

    **Upload phase (once per document):**
    1. Your PDF is split into small ~800-character chunks
    2. Each chunk is converted to a 384-dimensional vector (using HuggingFace)
    3. All vectors are stored in a FAISS index on disk

    **Question phase (every question):**
    1. Your question is converted to a vector using the same model
    2. FAISS finds the 4 most relevant chunks (using MMR for diversity)
    3. Those 4 chunks are given to Groq's LLaMA3 as context
    4. The LLM generates an answer using ONLY that context
    5. Page citations are extracted from chunk metadata

    ### Get started

    → **Upload a PDF** using the sidebar on the left

    ---

    *100% free · No data sent to third parties except the LLM call to Groq*
    """)
    st.stop()  # Stop rendering here if no doc is loaded

# ── Active Document Tabs ───────────────────────────────────────────────────────
doc_id = st.session_state["doc_id"]
st.subheader(f"💬 Document: `{doc_id}`")

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluate"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # Initialize message history if first time
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render all past messages
    # This runs on EVERY re-render (that's how Streamlit works)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show latency for assistant messages
            if msg.get("latency"):
                st.caption(f"⏱ Generated in {msg['latency']}s")

            # Show sources if available
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s)", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>{src["document"]}</strong> — Page {src["page"]}<br>'
                            f'<em>{src["excerpt"]}</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input at the bottom of the page
    # The walrus operator := assigns and tests in one expression
    # If user typed something and pressed Enter → question is that string → enter if block
    # If the input is empty → question is None → skip the if block
    if question := st.chat_input("Ask anything about the document…"):

        # Immediately show the user's message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Show the AI response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving & generating…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "doc_id": doc_id,
                            "question": question,
                            "k": 4,   # retrieve top-4 chunks
                        },
                        timeout=60,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                        latency = data.get("latency_seconds", "?")

                        # Render the answer
                        st.markdown(answer)
                        st.caption(f"⏱ Generated in {latency}s")

                        # Render sources in a collapsible section
                        if sources:
                            with st.expander(f"📎 {len(sources)} source(s)", expanded=False):
                                for src in sources:
                                    st.markdown(
                                        f'<div class="source-card">'
                                        f'<strong>{src["document"]}</strong> — Page {src["page"]}<br>'
                                        f'<em>{src["excerpt"]}</em>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                        # Save to message history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "latency": latency,
                        })

                    else:
                        err = f"⚠️ API Error {resp.status_code}: {resp.text}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach the backend. Is it running on port 8000?")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: EVALUATE
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("""
    ### 📊 Evaluate RAG Pipeline Quality

    Test your system by providing questions with known reference answers.
    The system compares its generated answers to your references using
    **semantic similarity** (sentence-transformers, runs locally — no API cost).

    **Metrics:**
    - **Answer Relevance (0-1):** How similar is the generated answer to your reference? 0.8+ is good.
    - **Retrieval Precision (0-1):** How relevant were the retrieved chunks? 0.7+ is good.
    - **Latency:** How long did the full pipeline take?
    """)

    # Default example to show users the format
    default_json = json.dumps([
        {
            "question": "What is the main contribution of this paper?",
            "reference_answer": "Replace this with your expected answer."
        },
        {
            "question": "What datasets were used for evaluation?",
            "reference_answer": "Replace this with your expected answer."
        }
    ], indent=2)

    raw_input = st.text_area(
        "Paste your question-answer pairs (JSON format):",
        value=default_json,
        height=200,
    )

    if st.button("▶ Run Evaluation", type="primary"):
        # Parse JSON input
        try:
            pairs = json.loads(raw_input)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        with st.spinner("Running evaluation… may take 30-60 seconds"):
            try:
                resp = requests.post(
                    f"{API_URL}/evaluate",
                    json={"doc_id": doc_id, "pairs": pairs},
                    timeout=180,
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend.")
                st.stop()

        if resp.status_code == 200:
            data = resp.json()
            summary = data["summary"]

            # Summary metrics row
            st.divider()
            st.subheader("Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Answer Relevance", f"{summary['avg_answer_relevance']:.0%}")
            col2.metric("Avg Retrieval Precision", f"{summary['avg_retrieval_precision']:.0%}")
            col3.metric("Avg Latency", f"{summary['avg_latency_seconds']}s")
            col4.metric("Questions Tested", summary["total_questions"])

            # Per-question breakdown
            st.divider()
            st.subheader("Per-Question Details")
            for item in data["per_question"]:
                with st.expander(f"❓ {item['question'][:80]}…"):
                    st.markdown(f"**Generated answer:** {item['generated_answer']}")
                    st.markdown(f"**Reference answer:** {item['reference_answer']}")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Relevance", f"{item['answer_relevance_score']:.0%}")
                    c2.metric("Precision", f"{item['retrieval_precision']:.0%}")
                    c3.metric("Latency", f"{item['latency_seconds']}s")

                    if item.get("sources"):
                        st.caption("Sources used:")
                        for src in item["sources"]:
                            st.caption(f"  • {src['document']}, page {src['page']}")
        else:
            st.error(f"Evaluation failed ({resp.status_code}): {resp.text}")