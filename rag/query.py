"""
RAG query engine for Argus.

Embeds a question with Cohere, retrieves top-5 contexts from Pinecone,
and generates an answer with Groq (Llama 3.3 70B).

Run from project root:
    python rag/query.py "What conflicts are happening in the Middle East?"
"""

import os
import sys

import cohere
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone

load_dotenv()

SYSTEM_PROMPT = (
    "You are Argus, a defense intelligence analyst. "
    "Answer questions using only the provided context from GDELT geopolitical event data "
    "and USASpending DoD contract data. Be precise and cite specific records when possible. "
    "If the context doesn't contain enough information, say so."
)

TOP_K = 5


def get_clients():
    cohere_key = os.getenv("COHERE_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "argus-index")

    missing = [n for n, v in [
        ("COHERE_API_KEY", cohere_key),
        ("PINECONE_API_KEY", pinecone_key),
        ("GROQ_API_KEY", groq_key),
    ] if not v]

    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")

    co = cohere.Client(cohere_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
    groq_client = Groq(api_key=groq_key)

    return co, index, groq_client


def embed_query(co: cohere.Client, question: str) -> list[float]:
    response = co.embed(
        texts=[question],
        model="embed-english-v3.0",
        input_type="search_query",
    )
    return response.embeddings[0]


def retrieve(index, query_embedding: list[float]) -> list[dict]:
    result = index.query(vector=query_embedding, top_k=TOP_K, include_metadata=True)
    return result.matches


def build_context(matches: list[dict]) -> str:
    blocks = []
    for i, match in enumerate(matches, 1):
        meta = match.metadata or {}
        text = meta.get("text", "(no text)")
        source = match.id.split("_")[0]
        score = round(match.score, 3)
        blocks.append(f"[{i}] Source: {source} | Score: {score}\n{text}")
    return "\n\n".join(blocks)


def run_query(question: str) -> dict:
    """Callable by FastAPI. Raises on error; never calls sys.exit."""
    co, index, groq_client = get_clients()

    query_embedding = embed_query(co, question)
    matches = retrieve(index, query_embedding)

    sources = [
        {"_id": m.id, "_score": round(m.score, 3), **dict(m.metadata or {})}
        for m in matches
    ]

    if not matches:
        return {
            "answer": "No relevant records found in the vector database. Try running the ingestion and embed scripts first.",
            "sources": [],
        }

    context = build_context(matches)
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return {"answer": completion.choices[0].message.content, "sources": sources}


def ask(question: str) -> str:
    """CLI wrapper — catches all errors and returns them as strings."""
    try:
        return run_query(question)["answer"]
    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag/query.py \"Your question here\"")
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}\n")
    answer = ask(question)
    print(f"Answer:\n{answer}")
