import pytest
from pydantic import ValidationError

from schemas import ChunkResult, QueryRequest, QueryResponse


# ---------------------------------------------------------------------------
# QueryRequest — top_k validation
# ---------------------------------------------------------------------------

def test_query_request_default_top_k():
    req = QueryRequest(question="test")
    assert req.top_k == 5


def test_query_request_top_k_minimum_valid():
    req = QueryRequest(question="test", top_k=1)
    assert req.top_k == 1


def test_query_request_top_k_maximum_valid():
    req = QueryRequest(question="test", top_k=20)
    assert req.top_k == 20


def test_query_request_top_k_zero_raises():
    with pytest.raises(ValidationError):
        QueryRequest(question="test", top_k=0)


def test_query_request_top_k_negative_raises():
    with pytest.raises(ValidationError):
        QueryRequest(question="test", top_k=-1)


def test_query_request_top_k_above_max_raises():
    with pytest.raises(ValidationError):
        QueryRequest(question="test", top_k=21)


# ---------------------------------------------------------------------------
# ChunkResult
# ---------------------------------------------------------------------------

def test_chunk_result_stores_fields():
    chunk = ChunkResult(text="hello", source="doc.txt", chunk_index=3, score=0.87)
    assert chunk.text == "hello"
    assert chunk.source == "doc.txt"
    assert chunk.chunk_index == 3
    assert chunk.score == 0.87


# ---------------------------------------------------------------------------
# QueryResponse
# ---------------------------------------------------------------------------

def test_query_response_with_results():
    chunk = ChunkResult(text="hello", source="doc.txt", chunk_index=0, score=0.9)
    resp = QueryResponse(question="What?", results=[chunk])
    assert resp.question == "What?"
    assert len(resp.results) == 1


def test_query_response_empty_results():
    resp = QueryResponse(question="What?", results=[])
    assert resp.results == []
