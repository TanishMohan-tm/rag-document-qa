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
st.set_page_config(
    page_title="RAG Document Analysis",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Case Study Theme CSS (Playfair Display, Source Serif 4, DM Sans, JetBrains Mono) ──
st.markdown("""
<style>
    /* Import fonts from the case study page */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global settings */
    html, body, [class*="st-"] {
        font-family: 'Source Serif 4', serif !important;
        color: #f0ede8 !important;      /* warm off-white */
        font-size: 15px !important;
        line-height: 1.6 !important;
        font-weight: 300 !important;    /* light weight like case study prose */
    }

    /* Block container */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }

    /* Captions / secondary text */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-family: 'DM Sans', sans-serif !important;
        color: #b0aaa0 !important;
        font-size: 13px !important;
        font-weight: 400 !important;
    }

    /* ===== HEADINGS – Playfair Display (serif, bold) ===== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        font-style: normal !important;
        color: #D97757 !important;        /* coral accent */
        letter-spacing: -0.02em !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Main title – add a subtle italic for flair (like the case study's <em>) */
    .stApp header + div h1:first-of-type {
        font-style: italic !important;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
    }

    /* Sidebar headings */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #D97757 !important;
        font-family: 'Playfair Display', serif !important;
    }

    /* Hide anchor links */
    [data-testid="stHeaderActionElements"], .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
        display: none !important;
    }

    /* Background */
    .stApp {
        background-color: #1a1a1a;
    }

    /* Hide Streamlit UI chrome */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    .stAppDeployButton {
        display: none;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #141414;
        border-right: 1px solid #2d2a28;
    }
    [data-testid="stSidebar"] * {
        color: #f0ede8 !important;
        font-family: 'Source Serif 4', serif !important;
    }

    /* Chat input area */
    .stChatInputContainer {
        border-top: none;
        padding-top: 0;
        background-color: #1a1a1a;
    }
    .stChatInputContainer textarea, .stChatInputContainer input {
        background-color: #252220 !important;
        color: #f0ede8 !important;
        border-color: #3a3632 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        border-radius: 0;
        border: none;
        border-bottom: 1px solid #2d2a28;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
        box-shadow: none;
    }
    [data-testid="stChatMessage"][aria-label="user"] {
        background-color: transparent;
        border: none;
        border-bottom: 1px solid #2d2a28;
    }

    /* Source cards – coral left border */
    .source-card {
        background: #252220;
        border-left: 2px solid #D97757;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9em;
        color: #f0ede8;
        border-top: 1px solid #3a3632;
        border-right: 1px solid #3a3632;
        border-bottom: 1px solid #3a3632;
        font-family: 'Source Serif 4', serif !important;
    }
    .source-card strong {
        color: #D97757;
        font-weight: 600;
    }
    .source-card em {
        color: #b0aaa0;
        font-style: italic;
        display: block;
        margin-top: 4px;
    }

    /* Buttons – DM Sans */
    .stButton>button {
        background-color: #252220 !important;
        color: #f0ede8 !important;
        border: 1px solid #3a3632 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #D97757 !important;
        color: #D97757 !important;
        background-color: #252220 !important;
    }
    .stButton>button[kind="primary"] {
        background-color: #D97757 !important;
        color: #FFFFFF !important;
        border-color: #D97757 !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #C6684B !important;
        color: #FFFFFF !important;
    }

    /* Metrics – Playfair for values, DM Sans for labels */
    [data-testid="stMetricValue"] {
        color: #D97757 !important;
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #b0aaa0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 400 !important;
    }

    hr {
        border-color: #2d2a28 !important;
    }

    /* Code – JetBrains Mono */
    code {
        color: #D97757 !important;
        background: #252220 !important;
        border-radius: 4px;
        padding: 2px 4px;
        font-family: 'JetBrains Mono', monospace !important;
    }
    pre code {
        color: #f0ede8 !important;
        background: #252220 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Form elements */
    label, .stSelectbox, .stFileUploader {
        font-family: 'DM Sans', sans-serif !important;
        color: #f0ede8 !important;
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #252220 !important;
        color: #f0ede8 !important;
        border-color: #3a3632 !important;
        font-family: 'Source Serif 4', serif !important;
    }
    [data-testid="stFileUploader"] {
        background-color: #252220 !important;
        border-color: #3a3632 !important;
    }
    [data-testid="stExpander"] {
        border-color: #3a3632 !important;
        background-color: #1a1a1a !important;
    }
    [data-testid="stExpander"] summary {
        color: #D97757 !important;
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom-color: #2d2a28 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #b0aaa0 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTabs [aria-selected="true"] {
        color: #D97757 !important;
        border-bottom-color: #D97757 !important;
    }

    /* Alerts */
    .stAlert {
        background-color: #252220 !important;
        color: #f0ede8 !important;
        border-color: #3a3632 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Dropdowns */
    [data-baseweb="select"] {
        background-color: #252220 !important;
    }
    [data-baseweb="select"] * {
        color: #f0ede8 !important;
        background-color: #252220 !important;
    }
    [data-baseweb="popover"] {
        background-color: #252220 !important;
    }
    [data-baseweb="popover"] li {
        background-color: #252220 !important;
        color: #f0ede8 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-baseweb="popover"] li:hover {
        background-color: #3a3632 !important;
    }

    /* Hide scrollbars */
    ::-webkit-scrollbar {
        width: 0px !important;
        height: 0px !important;
        background: transparent !important;
    }
    * {
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    .stApp {
        overflow: hidden !important;
    }
    .main {
        overflow: auto !important;
        scrollbar-width: none !important;
    }
    .main::-webkit-scrollbar {
        display: none !important;
    }

    /* Body text overrides */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown ol, .stMarkdown ul {
        color: #f0ede8 !important;
        font-size: 15px !important;
        line-height: 1.65 !important;
        font-family: 'Source Serif 4', serif !important;
        font-weight: 300 !important;
    }
    .stMarkdown strong {
        color: #D97757 !important;
        font-weight: 600 !important;
    }
    .stMarkdown em {
        color: #b0aaa0 !important;
        font-style: italic;
    }

    /* Section label styling (I. Ingestion, etc.) */
    .stMarkdown h3 + p strong:first-child {
        color: #D97757 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Document Analysis")
    st.caption("Retrieval-Augmented Generation Indexing")
    st.divider()

    # ── SECTION 1: Upload a new PDF ───────────────────────────────────
    st.markdown('<p style="color: #D97757; font-size: 12px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem;">I. Ingestion</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.button("Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing PDF... (initial run caches embeddings)"):
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=300,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["doc_id"] = data["doc_id"]
                        st.session_state["messages"] = []
                        st.success(
                            f"Processing Complete.\n\n"
                            f"Parsed **{data['pages']} pages** into "
                            f"**{data['chunks']} vector embeddings.**"
                        )
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Backend connection failed.\n\n"
                        "Verify service availability:\n"
                        "```\ncd backend\nuvicorn api:app --port 8000\n```"
                    )

    st.divider()

    # ── SECTION 2: Select an already-ingested document ────────────────
    st.markdown('<p style="color: #D97757; font-size: 12px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 1rem; margin-bottom: 0.5rem;">II. Active Workspace</p>', unsafe_allow_html=True)
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
                    if st.button("Load Workspace", use_container_width=True):
                        st.session_state["doc_id"] = chosen
                        st.session_state["messages"] = []
                        st.success(f"Loaded: {chosen}")
            else:
                st.info("No documents available in the index.")
    except requests.exceptions.ConnectionError:
        st.warning("Backend service unavailable.")

    if "doc_id" in st.session_state:
        st.divider()
        st.caption(f"**Active document:**")
        st.code(st.session_state["doc_id"], language=None)


# ── Main Area ─────────────────────────────────────────────────────────────────

if "doc_id" not in st.session_state:
    st.markdown("""
    <p style="color: #D97757; font-size: 12px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0;">RETRIEVAL-AUGMENTED GENERATION</p>
    """, unsafe_allow_html=True)
    st.markdown("""## Document Analysis Interface""")
    st.markdown("""This interface provides analytical capabilities over uploaded PDF documentation via Retrieval-Augmented Generation.""")

    st.markdown("""
    <p style="color: #D97757; font-size: 12px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2rem; margin-bottom: 0.5rem;">SYSTEM ARCHITECTURE</p>
    """, unsafe_allow_html=True)

    st.markdown("""
**Ingestion Pipeline**

1. Document parsing and segmentation into semantic chunks.
2. Vectorization mapping to a 384-dimensional embedding space.
3. Storage and indexing via FAISS for high-efficiency similarity search.

**Retrieval & Generation**

1. Query vectorization utilizing the identical embedding model.
2. Semantic search to isolate contextually relevant segments.
3. Synthesis of an evidence-based response using the retrieved context.
4. Automatic citation mapping to original document pages.
    """)

    st.markdown("""
    <p style="color: #D97757; font-size: 12px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2rem; margin-bottom: 0.5rem;">GETTING STARTED</p>
    """, unsafe_allow_html=True)

    st.markdown("""Initialize a session by uploading a document in the sidebar panel.""")

    st.divider()

    st.markdown("""<p style="color: #6b6662; font-size: 13px; font-style: italic;">Data handling is strictly local, excluding the generation phase API call.</p>""", unsafe_allow_html=True)
    st.stop()

doc_id = st.session_state["doc_id"]

st.markdown("""
<p style="color: #D97757; font-size: 12px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0;">ANALYSIS SESSION</p>
""", unsafe_allow_html=True)
st.markdown(f"## {doc_id}")

tab_chat, tab_eval = st.tabs(["Query Interface", "Quantitative Evaluation"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    question = st.chat_input("Ask anything about the document…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={
                    "doc_id": doc_id,
                    "question": question,
                    "k": 4,
                },
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                    "latency": data.get("latency_seconds", "?"),
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"[ API Error {resp.status_code} ] {resp.text}",
                })

        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Connection failed. Service unavailable.",
            })

        st.rerun()

    messages = st.session_state.messages
    pairs = []
    i = 0
    while i < len(messages):
        if i + 1 < len(messages) and messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            pairs.append((messages[i], messages[i + 1]))
            i += 2
        else:
            pairs.append((messages[i],))
            i += 1

    for pair in reversed(pairs):
        for msg in pair:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg.get("latency"):
                    st.caption(f"[ Generated in {msg['latency']}s ]")

                if msg.get("sources"):
                    with st.expander(f"[{len(msg['sources'])} Reference(s)]", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>{src["document"]}</strong> — Page {src["page"]}<br>'
                                f'<em>{src["excerpt"]}</em>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: EVALUATE
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("""
    ### Pipeline Evaluation Metrics

    Quantitatively measure system performance using reference pair validation.
    The system computes semantic similarity locally (zero API overhead).

    **Metrics:**
    - **Answer Relevance (0-1):** Generated answer semantic similarity to reference. (Target > 0.8)
    - **Retrieval Precision (0-1):** Context relevance of retrieved semantic chunks. (Target > 0.7)
    - **Latency:** End-to-end pipeline execution time.
    """)

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

    if st.button("Execute Evaluation", type="primary"):
        try:
            pairs = json.loads(raw_input)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {e}")
            st.stop()

        with st.spinner("Executing semantic evaluation..."):
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

            st.divider()
            st.subheader("Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Answer Relevance", f"{summary['avg_answer_relevance']:.0%}")
            col2.metric("Avg Retrieval Precision", f"{summary['avg_retrieval_precision']:.0%}")
            col3.metric("Avg Latency", f"{summary['avg_latency_seconds']}s")
            col4.metric("Questions Tested", summary["total_questions"])

            st.divider()
            st.subheader("Detailed Analysis")
            for item in data["per_question"]:
                with st.expander(f"Query: {item['question'][:80]}..."):
                    st.markdown(f"**Generated response:** {item['generated_answer']}")
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