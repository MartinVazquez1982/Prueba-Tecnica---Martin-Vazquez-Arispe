import json
import logging
import os
from contextlib import asynccontextmanager

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL, INDEX_PATH, METADATA_PATH
from schemas import ChunkResult, QueryRequest, QueryResponse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Global state loaded once at startup
_index: faiss.IndexFlatIP | None = None
_metadata: list[dict] = []
_embedder: OpenAIEmbeddings | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index, chunk metadata, and embedder on startup."""
    global _index, _metadata, _embedder

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Index not found. Run ingest.py first to generate python/data/faiss.index."
        )

    _index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, encoding="utf-8") as f:
        _metadata = json.load(f)
    _embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    logger.info(f"Index loaded: {_index.ntotal} vectors / {len(_metadata)} chunks")
    yield


app = FastAPI(title="Support Assistant — Retrieval API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Return API status and basic index statistics."""
    return {
        "status": "ok",
        "index_loaded": _index is not None,
        "total_chunks": len(_metadata),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Retrieve the most relevant documentation chunks for a given question.

    Embeds the question using the same model used during ingestion, then
    performs a cosine-similarity search against the FAISS index. Returns
    the top-k chunks sorted by relevance score (highest first).

    Raises 400 for empty questions, 503 if the index is not loaded.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if _index is None or _embedder is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    vector = np.array(
        _embedder.embed_query(request.question), dtype=np.float32
    ).reshape(1, -1)
    faiss.normalize_L2(vector)

    k = min(request.top_k, _index.ntotal)
    scores, indices = _index.search(vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _metadata[idx]
        results.append(
            ChunkResult(
                text=chunk["text"],
                source=chunk["source"],
                chunk_index=chunk["chunk_index"],
                score=round(float(score), 4),
            )
        )

    return QueryResponse(question=request.question, results=results)
