# Argus — Defense Intelligence Platform

A full-stack RAG-based defense intelligence dashboard that ingests live geopolitical and federal contract data, enables natural-language querying, and visualizes results on maps and charts.

## Stack

| Layer | Technology |
|---|---|
| ETL | Python, Pandas, requests |
| Embeddings | Cohere Embed v3 |
| Vector DB | Pinecone |
| LLM | Groq (Llama 3.3 70B) |
| Backend | FastAPI (Railway) |
| Frontend | React + Tailwind + Recharts + Mapbox (Vercel) |
| Database | Supabase (Phase 5+) |
| Automation | GitHub Actions (nightly ingestion) |

## Data Sources

- **GDELT 2.0** — real-time global geopolitical events
- **USASpending.gov** — DoD federal contracts

## Project Structure

```
defense-rag/
├── .env.example          # Safe template — copy to .env and add keys
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/              # Raw downloaded data (gitignored)
│   └── processed/        # Cleaned/chunked data (gitignored)
├── ingestion/            # ETL scripts
│   ├── gdelt_ingest.py
│   └── usaspending_ingest.py
├── rag/                  # Embedding + retrieval pipeline
├── backend/              # FastAPI app
├── frontend/             # React dashboard
└── logs/
```

## Setup

### 1. Clone & create virtual environment

```bash
py -3.11 -m venv venv
venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure secrets

```bash
copy .env.example .env
# Open .env and paste your real API keys — this file is gitignored
```

### 4. Run ingestion

```bash
python ingestion/gdelt_ingest.py
python ingestion/usaspending_ingest.py
```

## Phases

- [x] **Phase 1** — Project setup + data ingestion pipeline
- [ ] **Phase 2** — RAG pipeline (Cohere + Pinecone + Groq)
- [ ] **Phase 3** — FastAPI backend
- [ ] **Phase 4** — React dashboard (Map, Trends, Query views)
- [ ] **Phase 5** — Deployment (Vercel + Railway)
- [ ] **Phase 6** — Additional data sources
- [ ] **Phase 7** — Alert system + user accounts

## Security

- `.env` is gitignored and **must never be committed**
- All secrets go in `.env` locally, and in the Railway/Vercel dashboard for deployment
- `data/raw/` and `data/processed/` are gitignored to avoid committing large or sensitive datasets
