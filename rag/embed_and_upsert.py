"""
Embed processed records with Cohere and upsert them into Pinecone.

Loads all *.json files from data/processed/ plus gdelt_events.csv if present.
Each record must have a 'text' field to embed.

Run from project root:
    python rag/embed_and_upsert.py
"""

import json
import os
import sys
from pathlib import Path

import cohere
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

load_dotenv()

PROCESSED_DIR = Path("data/processed")
COHERE_BATCH = 96   # Cohere embed-v3 max
PINECONE_BATCH = 100


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_json_records(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_csv_records(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str).fillna("")
    return df.to_dict(orient="records")


def load_all_records() -> list[tuple[str, list[dict]]]:
    """Returns list of (source_name, records) tuples."""
    sources = []

    for json_path in sorted(PROCESSED_DIR.glob("*.json")):
        try:
            records = load_json_records(json_path)
            sources.append((json_path.stem, records))
            print(f"  Loaded {len(records):,} records from {json_path.name}")
        except Exception as e:
            print(f"  WARN: could not load {json_path.name}: {e}")

    csv_path = PROCESSED_DIR / "gdelt_events.csv"
    if csv_path.exists():
        try:
            records = load_csv_records(csv_path)
            sources.append(("gdelt_events", records))
            print(f"  Loaded {len(records):,} records from gdelt_events.csv")
        except Exception as e:
            print(f"  WARN: could not load gdelt_events.csv: {e}")

    return sources


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

def embed_texts(co: cohere.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Cohere search_document input type."""
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return response.embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cohere_key = os.getenv("COHERE_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "argus-index")

    missing = [n for n, v in [("COHERE_API_KEY", cohere_key), ("PINECONE_API_KEY", pinecone_key)] if not v]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    co = cohere.Client(cohere_key)
    pc = Pinecone(api_key=pinecone_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        raise RuntimeError(f"Index '{index_name}' does not exist. Run scripts/create_index.py first.")

    index = pc.Index(index_name)

    print("=== Loading records ===")
    sources = load_all_records()

    if not sources:
        print("No records found in data/processed/. Run the ingestion scripts first.")
        return 0

    total_upserted = 0

    for source_name, records in sources:
        print(f"\n=== Embedding: {source_name} ({len(records):,} records) ===")

        # Filter out records without a text field
        valid = [r for r in records if r.get("text", "").strip()]
        if len(valid) < len(records):
            print(f"  Skipped {len(records) - len(valid)} records with empty 'text'")

        if not valid:
            print(f"  No embeddable records in {source_name}, skipping.")
            continue

        vectors = []

        for batch_start in tqdm(range(0, len(valid), COHERE_BATCH), desc="  Embedding", unit="batch"):
            batch = valid[batch_start : batch_start + COHERE_BATCH]
            texts = [r["text"] for r in batch]

            try:
                embeddings = embed_texts(co, texts)
            except Exception as e:
                print(f"\n  ERROR embedding batch at {batch_start}: {e}")
                continue

            for i, (record, embedding) in enumerate(zip(batch, embeddings)):
                vec_id = f"{source_name}_{batch_start + i}"
                metadata = {k: v for k, v in record.items()}
                # Pinecone metadata values must be str/int/float/bool/list
                # Coerce anything else to string
                safe_meta = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        safe_meta[k] = v
                    elif isinstance(v, list):
                        safe_meta[k] = [str(x) for x in v]
                    elif v is None:
                        safe_meta[k] = ""
                    else:
                        safe_meta[k] = str(v)

                vectors.append({"id": vec_id, "values": embedding, "metadata": safe_meta})

        # Upsert to Pinecone in batches
        print(f"  Upserting {len(vectors):,} vectors to Pinecone ...")
        for upsert_start in tqdm(range(0, len(vectors), PINECONE_BATCH), desc="  Upserting", unit="batch"):
            batch = vectors[upsert_start : upsert_start + PINECONE_BATCH]
            try:
                index.upsert(vectors=batch)
                total_upserted += len(batch)
            except Exception as e:
                print(f"\n  ERROR upserting batch at {upsert_start}: {e}")

    print(f"\n=== Done. Total vectors upserted: {total_upserted:,} ===")
    return total_upserted


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
