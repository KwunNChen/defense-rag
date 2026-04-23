"""
End-to-end test of the Argus RAG pipeline.

Runs 3 sample queries and prints answers with separators.

Run from project root:
    python scripts/test_rag.py

Prerequisites:
    1. python scripts/create_index.py
    2. python rag/embed_and_upsert.py
"""

import sys
from pathlib import Path

# Allow imports from project root when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.query import ask

QUERIES = [
    "What are the most violent geopolitical events in the last 24 hours?",
    "Which defense contractors received the largest DoD contracts recently?",
    "Are there any conflicts in Africa right now?",
]

SEPARATOR = "\n" + "=" * 70 + "\n"


def main():
    print("=== Argus RAG Pipeline — End-to-End Test ===\n")

    for i, question in enumerate(QUERIES, 1):
        print(f"Query {i}: {question}")
        print("-" * 50)
        answer = ask(question)
        print(answer)
        if i < len(QUERIES):
            print(SEPARATOR)

    print("\n=== Test complete ===")


if __name__ == "__main__":
    main()
