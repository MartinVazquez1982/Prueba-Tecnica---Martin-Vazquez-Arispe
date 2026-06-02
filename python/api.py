import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import DATA_DIR, EMBEDDING_MODEL
from schemas import ChunkResult, QueryRequest, QueryResponse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_vectorstore: FAISS | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vectorstore

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    if not DATA_DIR.exists():
        raise RuntimeError("Vector store not found. Run ingest.py first.")

    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    _vectorstore = FAISS.load_local(
        str(DATA_DIR), embedder, allow_dangerous_deserialization=True
    )
    logger.info(f"Vector store loaded: {_vectorstore.index.ntotal} vectors")
    yield


app = FastAPI(title="Support Assistant — Retrieval API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": _vectorstore is not None,
        "total_chunks": _vectorstore.index.ntotal if _vectorstore else 0,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Retrieve the most relevant chunks for a question via the LangChain retriever."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if _vectorstore is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    docs_with_scores = _vectorstore.similarity_search_with_relevance_scores(
        request.question, k=request.top_k
    )

    results = [
        ChunkResult(
            text=doc.page_content,
            source=doc.metadata.get("source", ""),
            chunk_index=doc.metadata.get("chunk_index", 0),
            score=round(float(score), 4),
        )
        for doc, score in docs_with_scores
    ]

    return QueryResponse(question=request.question, results=results)
