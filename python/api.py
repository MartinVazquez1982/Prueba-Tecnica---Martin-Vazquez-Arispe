import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_community.vectorstores import FAISS
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from config import DATA_DIR, MIN_RELEVANCE_SCORE
from embedder import get_embedder
from schemas import ChunkResult, QueryRequest, QueryResponse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    if not DATA_DIR.exists():
        raise RuntimeError("Vector store not found. Run ingest.py first.")

    embedder = get_embedder()
    app.state.vectorstore = FAISS.load_local(
        str(DATA_DIR), embedder, allow_dangerous_deserialization=True
    )
    logger.info(f"Vector store loaded: {app.state.vectorstore.index.ntotal} vectors")
    yield
    app.state.vectorstore = None


app = FastAPI(title="Support Assistant — Retrieval API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health(request: Request):
    vs = getattr(request.app.state, "vectorstore", None)
    return {
        "status": "ok",
        "index_loaded": vs is not None,
        "total_chunks": vs.index.ntotal if vs else 0,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: Request, body: QueryRequest):
    """Retrieve the most relevant chunks for a question via the LangChain retriever."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    vs = getattr(req.app.state, "vectorstore", None)
    if vs is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    try:
        docs_with_scores = await asyncio.to_thread(
            vs.similarity_search_with_relevance_scores,
            body.question,
            k=body.top_k,
        )
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="OpenAI API request timed out.")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached. Try again later.")
    except APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not connect to OpenAI API.")
    except APIStatusError as exc:
        logger.error(f"OpenAI API error: {exc.status_code} — {exc.message}")
        raise HTTPException(status_code=502, detail="OpenAI API returned an error.")
    except Exception as exc:
        logger.error(f"FAISS search error: {exc}")
        raise HTTPException(status_code=500, detail="Internal error during vector search.")

    results = [
        ChunkResult(
            text=doc.page_content,
            source=doc.metadata.get("source", ""),
            chunk_index=doc.metadata.get("chunk_index", 0),
            score=round(float(score), 4),
        )
        for doc, score in docs_with_scores
        if score >= MIN_RELEVANCE_SCORE
    ]

    return QueryResponse(question=body.question, results=results)
