"""
GDELT 2.0 ingestion script.

Downloads the latest 15-minute event file from GDELT, parses it into a
structured DataFrame, filters for high-impact events, and saves to
data/processed/gdelt_events.csv.

GDELT GKG (Global Knowledge Graph) format docs:
http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf
"""

import io
import os
import zipfile
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

# GDELT 2.0 last-update master list
GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# Columns we care about (GDELT 2.0 events have 61 columns total)
COLUMNS = {
    0: "GlobalEventID",
    1: "Day",            # YYYYMMDD
    5: "Actor1Name",
    11: "Actor2Name",
    15: "IsRootEvent",
    26: "EventCode",     # CAMEO code
    27: "EventBaseCode",
    28: "EventRootCode",
    30: "GoldsteinScale",   # conflict/cooperation score (-10 to +10)
    31: "NumMentions",
    33: "AvgTone",
    37: "Actor1Geo_FullName",
    39: "Actor1Geo_Lat",
    40: "Actor1Geo_Long",
    41: "Actor2Geo_FullName",
    43: "Actor2Geo_Lat",
    44: "Actor2Geo_Long",
    53: "SOURCEURL",
}

# Only keep events with significant conflict signal (negative Goldstein)
GOLDSTEIN_THRESHOLD = -3.0
MIN_MENTIONS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_latest_gdelt_url() -> str:
    """Parse the GDELT master update file and return the latest events CSV URL."""
    resp = requests.get(GDELT_MASTER_URL, timeout=30)
    resp.raise_for_status()
    for line in resp.text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3 and "export.CSV.zip" in parts[2]:
            return parts[2]
    raise RuntimeError("Could not find export CSV URL in GDELT master list")


def download_gdelt_zip(url: str) -> pd.DataFrame:
    """Download and parse the zipped GDELT events CSV."""
    print(f"Downloading: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".CSV")][0]
        with z.open(csv_name) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                dtype=str,  # read all as str first, cast selectively
                low_memory=False,
            )

    return df


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns, cast types, and filter for high-impact events."""
    # Keep only the columns we defined
    col_indices = list(COLUMNS.keys())
    df = df.iloc[:, col_indices].copy()
    df.columns = list(COLUMNS.values())

    # Cast numeric columns
    for col in ["GoldsteinScale", "AvgTone", "Actor1Geo_Lat", "Actor1Geo_Long",
                "Actor2Geo_Lat", "Actor2Geo_Long"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(0).astype(int)

    # Parse date
    df["Date"] = pd.to_datetime(df["Day"], format="%Y%m%d", errors="coerce")

    # Filter: significant conflict events with enough media coverage
    mask = (
        df["GoldsteinScale"].notna()
        & (df["GoldsteinScale"] <= GOLDSTEIN_THRESHOLD)
        & (df["NumMentions"] >= MIN_MENTIONS)
    )
    filtered = df[mask].copy()

    # Drop rows with no location data
    filtered = filtered.dropna(subset=["Actor1Geo_Lat", "Actor1Geo_Long"])

    filtered = filtered.reset_index(drop=True)
    print(f"  Raw rows: {len(df):,}  |  After filter: {len(filtered):,}")
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== GDELT 2.0 Ingestion ===")

    url = get_latest_gdelt_url()
    raw_df = download_gdelt_zip(url)

    events_df = filter_and_clean(raw_df)

    out_path = PROCESSED_DIR / "gdelt_events.csv"
    events_df.to_csv(out_path, index=False)
    print(f"Saved {len(events_df):,} events → {out_path}")


if __name__ == "__main__":
    main()
