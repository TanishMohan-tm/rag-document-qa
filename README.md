# RAG Document QA System

> Upload any PDF. Ask questions in plain English. Get answers with page citations.

**[🚀 Live Demo](https://your-frontend-url.up.railway.app)**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![Free](https://img.shields.io/badge/Cost-$0-brightgreen)

---

## What It Does

Upload any PDF — research paper, legal document, technical manual — and have a 
natural language conversation with it. Every answer includes page-level citations 
so you can verify the source.

**Works on:** Research papers · Legal contracts · Technical manuals · Any PDF

---

## Architecture
```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  Frontend   │    │   Backend   │    │   Vector DB    │
│  (Next.js)  │    │  (FastAPI)  │    │   (FAISS)    │
│  client.tsx │    │   main.py   │    │ data/vectors/  │
└──────┬──────┘    └──────┬──────┘    └───────┬──────┘
       │ GET /docs          │ load PDF        │
       │                   │ split into chunks │
       │                   │ create FAISS    │
       │                   └───────────────┬─┘
       │
       │ POST /ask
       │    │
       │    ▼
       │   ┌─────────────────────────────┐
       │   │        RAG Chain            │
       │   │  (chain.py + retriever.py)  │
       │   └───────────────┬───────────────┘
       │                   │
       │           Retrieves chunks from FAISS
       │                   │
       │            Calls LLM (Groq free)
       │                   │
       │        Returns: answer + sources
       │                   │
       └───────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LangChain 0.2 (LCEL) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (free, local) |
| Vector Store | FAISS (local, no external DB) |
| LLM | Groq LLaMA3 70B (free tier) |
| Frontend | Streamlit |
| Deployment | Docker + Railway |

---

## Run Locally

**Prerequisites:** Python 3.11, Docker

```bash
# 1. Clone
git clone https://github.com/TanishMohan-tm/rag-document-qa.git
cd rag-document-qa

# 2. Set up environment
cp .env.example .env
# Edit .env — add your free Groq key from https://console.groq.com

# 3. Run with Docker
docker compose up --build

# Frontend: http://localhost:8501
# Backend API: http://localhost:8000/docs
```

**Without Docker:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.backend.txt
pip install -r requirements.frontend.txt

# Terminal 1
cd backend && uvicorn api:app --reload --port 8000

# Terminal 2  
cd frontend && streamlit run app.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/documents` | List ingested documents |
| POST | `/ingest` | Upload PDF → returns doc_id |
| POST | `/ask` | Question → answer + citations |
| POST | `/evaluate` | Batch QA evaluation |

Full docs at `/docs` (Swagger UI).

---

## Cost

Everything runs free:
- **Embeddings:** HuggingFace model runs locally on CPU
- **LLM:** Groq free tier (6,000 req/day, no credit card)
- **Vector DB:** FAISS saves to disk (2 files)
- **Hosting:** Railway free tier

Total: **$0**