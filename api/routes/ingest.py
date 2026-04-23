"""
POST /api/v1/ingest — trigger the full ingestion + embedding pipeline.

Runs sequentially:
  1. ingestion/gdelt_ingest.py  (GDELT events)
  2. ingestion/usaspending_ingest.py  (DoD contracts)
  3. rag/embed_and_upsert.py  (Cohere embed + Pinecone upsert)

Protected by Bearer token auth when INGEST_SECRET is set in .env.
Expected runtime: 30-120 seconds depending on data size.
"""

import os

from fastapi import APIRouter, Header, HTTPException

import ingestion.gdelt_ingest as gdelt_ingest
import ingestion.usaspending_ingest as usaspending_ingest
import rag.embed_and_upsert as embed_and_upsert
from api.models import IngestResponse

router = APIRouter()


def _verify_token(authorization: str | None):
    secret = os.getenv("INGEST_SECRET")
    if not secret:
        return  # dev mode — no auth required
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    if authorization[7:] != secret:
        raise HTTPException(status_code=401, detail="Invalid token.")


@router.post("/ingest", response_model=IngestResponse)
def ingest(authorization: str | None = Header(default=None)):
    _verify_token(authorization)

    try:
        gdelt_count = gdelt_ingest.main() or 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GDELT ingestion failed: {e}")

    try:
        usaspending_count = usaspending_ingest.main() or 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"USASpending ingestion failed: {e}")

    try:
        vectors_upserted = embed_and_upsert.main() or 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding/upsert failed: {e}")

    return IngestResponse(
        status="success",
        gdelt_records=gdelt_count,
        usaspending_records=usaspending_count,
        vectors_upserted=vectors_upserted,
    )
