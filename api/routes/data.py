"""
GET /api/v1/events   — return processed GDELT events
GET /api/v1/contracts — return processed DoD contracts

Both endpoints read from data/processed/ and return up to `limit` records.
Returns 404 if the data files haven't been ingested yet.
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.models import ContractsResponse, EventsResponse

router = APIRouter()

PROCESSED_DIR = Path("data/processed")


@router.get("/events", response_model=EventsResponse)
def get_events(limit: int = Query(100, ge=1, le=10_000)):
    path = PROCESSED_DIR / "gdelt_events.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="GDELT events data not found. Run POST /api/v1/ingest first.",
        )

    try:
        df = pd.read_csv(path, dtype=str).fillna("").head(limit)
        events = df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load events: {e}")

    return EventsResponse(count=len(events), events=events)


@router.get("/contracts", response_model=ContractsResponse)
def get_contracts(limit: int = Query(100, ge=1, le=10_000)):
    # Primary: dod_contracts.csv (produced by usaspending_ingest.py)
    csv_path = PROCESSED_DIR / "dod_contracts.csv"

    # Fallback: usaspending_*.json glob, pick most recent
    json_paths = sorted(PROCESSED_DIR.glob("usaspending_*.json"))

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, dtype=str).fillna("").head(limit)
            contracts = df.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load contracts: {e}")
    elif json_paths:
        latest = json_paths[-1]
        try:
            with open(latest) as f:
                data = json.load(f)
            contracts = data if isinstance(data, list) else [data]
            contracts = contracts[:limit]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load contracts: {e}")
    else:
        raise HTTPException(
            status_code=404,
            detail="Contract data not found. Run POST /api/v1/ingest first.",
        )

    return ContractsResponse(count=len(contracts), contracts=contracts)
