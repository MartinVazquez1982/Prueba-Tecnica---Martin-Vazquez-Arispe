import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from unittest.mock import MagicMock

from api import health, query
from schemas import QueryRequest


def make_request(vs):
    """Build a minimal mock FastAPI Request with the given vectorstore in app.state."""
    req = MagicMock()
    req.app.state.vectorstore = vs
    return req


@pytest.fixture
def default_vs():
    docs_with_scores = [
        (Document(page_content="First chunk.", metadata={"source": "doc.txt", "chunk_index": 0}), 0.95),
        (Document(page_content="Second chunk.", metadata={"source": "doc.txt", "chunk_index": 1}), 0.80),
    ]
    vs = MagicMock()
    vs.index.ntotal = 2
    vs.similarity_search_with_relevance_scores.return_value = docs_with_scores
    return vs


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_ok(default_vs):
    assert health(make_request(default_vs)) == {
        "status": "ok",
        "index_loaded": True,
        "total_chunks": 2,
    }


def test_health_not_loaded():
    result = health(make_request(None))
    assert result["index_loaded"] is False
    assert result["total_chunks"] == 0


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

async def test_query_ok(default_vs):
    result = await query(make_request(default_vs), QueryRequest(question="What is the policy?"))
    assert result.question == "What is the policy?"
    assert len(result.results) == 2


async def test_query_result_fields(default_vs):
    r = (await query(make_request(default_vs), QueryRequest(question="test"))).results[0]
    assert r.text == "First chunk."
    assert r.source == "doc.txt"
    assert r.score == 0.95


async def test_query_empty_question_raises_400(default_vs):
    with pytest.raises(HTTPException) as exc:
        await query(make_request(default_vs), QueryRequest(question="   "))
    assert exc.value.status_code == 400


async def test_query_not_loaded_raises_503():
    with pytest.raises(HTTPException) as exc:
        await query(make_request(None), QueryRequest(question="hello"))
    assert exc.value.status_code == 503


async def test_query_top_k():
    vs = MagicMock()
    vs.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="Only chunk.", metadata={"source": "a.txt", "chunk_index": 0}), 0.9),
    ]
    result = await query(make_request(vs), QueryRequest(question="test", top_k=1))
    vs.similarity_search_with_relevance_scores.assert_called_once_with("test", k=1)
    assert len(result.results) == 1
