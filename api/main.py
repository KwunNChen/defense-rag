"""
Argus Intelligence API — FastAPI entry point.

Run from project root:
    uvicorn api.main:app --reload --port 8000

Routes:
    GET  /                      health check
    GET  /api/v1/health         health + UTC timestamp
    GET  /api/v1/events         GDELT geopolitical events
    GET  /api/v1/contracts      DoD contracts
    POST /api/v1/query          RAG natural-language query
    POST /api/v1/ingest         trigger full ingestion pipeline
"""

from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import data, ingest, query

load_dotenv()

app = FastAPI(
    title="Argus Intelligence API",
    version="0.1.0",
    description="Defense intelligence RAG platform — GDELT + USASpending + Groq",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"status": "online", "service": "Argus"}


@app.get("/api/v1/health")
def health():
    return {
        "status": "online",
        "service": "Argus",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
