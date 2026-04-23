"""
USASpending.gov DoD contract ingestion script.

Queries the USASpending Awards API for recent Department of Defense contracts,
normalizes the response, and saves to data/processed/dod_contracts.csv.

API docs: https://api.usaspending.gov/
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.usaspending.gov/api/v2"
AWARDS_SEARCH = f"{API_BASE}/search/spending_by_award/"

# DoD awarding agency CGAC codes
DOD_AGENCY_CODES = ["017"]  # Department of Defense

# Pull contracts from last N days
LOOKBACK_DAYS = 30

# Max contracts per run (pagination via pages)
MAX_RECORDS = 500
PAGE_SIZE = 100  # API max per page

# Fields to extract from each award
FIELDS = [
    "Award ID",
    "Recipient Name",
    "Start Date",
    "End Date",
    "Award Amount",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Award Type",
    "Description",
    "Place of Performance State Code",
    "Place of Performance Country Code",
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def build_payload(page: int, date_range: dict) -> dict:
    return {
        "filters": {
            "award_type_codes": ["A", "B", "C", "D"],  # contract types
            "agencies": [
                {
                    "type": "awarding",
                    "tier": "toptier",
                    "name": "Department of Defense",
                }
            ],
            "time_period": [date_range],
        },
        "fields": FIELDS,
        "page": page,
        "limit": PAGE_SIZE,
        "sort": "Award Amount",
        "order": "desc",
        "subawards": False,
    }


def fetch_contracts(lookback_days: int = LOOKBACK_DAYS) -> list[dict]:
    """Page through USASpending API and return raw award records."""
    from datetime import date, timedelta

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    date_range = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    print(f"Fetching DoD contracts: {date_range['start_date']} → {date_range['end_date']}")

    records = []
    page = 1

    while len(records) < MAX_RECORDS:
        payload = build_payload(page, date_range)
        resp = requests.post(AWARDS_SEARCH, json=payload, timeout=30)

        if resp.status_code != 200:
            print(f"  API error {resp.status_code}: {resp.text[:200]}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        records.extend(results)
        print(f"  Page {page}: +{len(results)} records (total: {len(records)})")

        if not data.get("page_metadata", {}).get("hasNext", False):
            break

        page += 1
        time.sleep(0.5)  # polite rate limiting

    return records[:MAX_RECORDS]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    # Rename to snake_case
    rename_map = {
        "Award ID": "award_id",
        "Recipient Name": "recipient_name",
        "Start Date": "start_date",
        "End Date": "end_date",
        "Award Amount": "award_amount",
        "Awarding Agency": "awarding_agency",
        "Awarding Sub Agency": "awarding_sub_agency",
        "Award Type": "award_type",
        "Description": "description",
        "Place of Performance State Code": "perf_state",
        "Place of Performance Country Code": "perf_country",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Cast types
    df["award_amount"] = pd.to_numeric(df.get("award_amount", pd.Series(dtype=float)), errors="coerce")
    for date_col in ["start_date", "end_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.sort_values("award_amount", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== USASpending DoD Contract Ingestion ===")

    records = fetch_contracts()
    if not records:
        print("No records returned. Check API availability.")
        return 0

    df = normalize(records)

    out_path = PROCESSED_DIR / "dod_contracts.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} contracts → {out_path}")
    print(f"Total obligated: ${df['award_amount'].sum():,.0f}")
    return len(df)


if __name__ == "__main__":
    main()
