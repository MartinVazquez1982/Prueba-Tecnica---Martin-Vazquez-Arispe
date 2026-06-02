import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from unittest.mock import MagicMock

import api as api_module
from api import health, query
from schemas import QueryRequest


@pytest.fixture(autouse=True)
def default_state(monkeypatch):
    """Inject a fake FAISS vectorstore before every test, restored after."""
    docs_with_scores = [
        (Document(page_content="First chunk.", metadata={"source": "doc.txt", "chunk_index": 0}), 0.95),
        (Document(page_content="Second chunk.", metadata={"source": "doc.txt", "chunk_index": 1}), 0.80),
    ]

    vs = MagicMock()
    vs.index.ntotal = 2
    vs.similarity_search_with_relevance_scores.return_value = docs_with_scores
    vs.as_retriever.return_value = MagicMock()

    monkeypatch.setattr(api_module, "_vectorstore", vs)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_ok():
    assert health() == {"status": "ok", "index_loaded": True, "total_chunks": 2}


def test_health_not_loaded(monkeypatch):
    monkeypatch.setattr(api_module, "_vectorstore", None)
    result = health()
    assert result["index_loaded"] is False
    assert result["total_chunks"] == 0


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

def test_query_ok():
    result = query(QueryRequest(question="What is the policy?"))
    assert result.question == "What is the policy?"
    assert len(result.results) == 2


def test_query_result_fields():
    r = query(QueryRequest(question="test")).results[0]
    assert r.text == "First chunk."
    assert r.source == "doc.txt"
    assert r.score == 0.95


def test_query_empty_question_raises_400():
    with pytest.raises(HTTPException) as exc:
        query(QueryRequest(question="   "))
    assert exc.value.status_code == 400


def test_query_not_loaded_raises_503(monkeypatch):
    monkeypatch.setattr(api_module, "_vectorstore", None)
    with pytest.raises(HTTPException) as exc:
        query(QueryRequest(question="hello"))
    assert exc.value.status_code == 503


def test_query_top_k(monkeypatch):
    vs = MagicMock()
    vs.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="Only chunk.", metadata={"source": "a.txt", "chunk_index": 0}), 0.9),
    ]
    vs.as_retriever.return_value = MagicMock()
    monkeypatch.setattr(api_module, "_vectorstore", vs)

    result = query(QueryRequest(question="test", top_k=1))
    vs.similarity_search_with_relevance_scores.assert_called_once_with("test", k=1)
    assert len(result.results) == 1
