"""
POST /api/v1/query — natural-language RAG query endpoint.

Accepts a question, runs it through the Argus RAG pipeline
(Cohere embed → Pinecone retrieve → Groq answer), and returns
the answer plus source metadata.
"""

from fastapi import APIRouter, HTTPException

from api.models import QueryRequest, QueryResponse
from rag.query import run_query

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = run_query(body.question)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=f"Configuration error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(
        question=body.question,
        answer=result["answer"],
        sources=result["sources"],
    )
