"""
Pydantic request/response models for the Argus API.
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


class IngestResponse(BaseModel):
    status: str
    gdelt_records: int
    usaspending_records: int
    vectors_upserted: int


class EventsResponse(BaseModel):
    count: int
    events: list[dict]


class ContractsResponse(BaseModel):
    count: int
    contracts: list[dict]
