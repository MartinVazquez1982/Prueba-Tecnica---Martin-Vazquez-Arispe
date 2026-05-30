from pydantic import BaseModel

from config import DEFAULT_TOP_K


class QueryRequest(BaseModel):
    """Incoming query from the user or n8n webhook."""
    question: str
    top_k: int = DEFAULT_TOP_K


class ChunkResult(BaseModel):
    """A single retrieved document chunk with its similarity score."""
    text: str
    source: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    """Response returned to the caller with the ranked relevant chunks."""
    question: str
    results: list[ChunkResult]
