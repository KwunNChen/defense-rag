"""
Create the Pinecone vector index for Argus.

Run from project root:
    python scripts/create_index.py

Creates a serverless index (AWS us-east-1, cosine, 1024 dims) if it doesn't
already exist. Safe to re-run — idempotent.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "argus-index")

    if not api_key:
        print("ERROR: PINECONE_API_KEY not set in .env")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        print(f"Index '{index_name}' already exists — nothing to do.")
        info = pc.describe_index(index_name)
        print(f"  Dimension : {info.dimension}")
        print(f"  Metric    : {info.metric}")
        print(f"  Status    : {info.status.state}")
        return

    print(f"Creating index '{index_name}' ...")
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Wait until ready
    import time
    for _ in range(30):
        info = pc.describe_index(index_name)
        if info.status.state == "Ready":
            break
        print("  Waiting for index to become ready ...")
        time.sleep(5)

    print(f"Index '{index_name}' is ready.")
    print(f"  Dimension : 1024")
    print(f"  Metric    : cosine")
    print(f"  Cloud     : aws / us-east-1")


if __name__ == "__main__":
    main()
