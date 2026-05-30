import os
import re
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DOCS_DIR,
    EMBEDDING_MODEL,
    INDEX_PATH,
    METADATA_PATH,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_txt(path: Path) -> str:
    """Read a plain text file and return its content as a string."""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_md(path: Path) -> str:
    """Read a Markdown file and strip syntax markers, returning clean prose.

    Removes headers (##), bold/italic (**), inline code (`), links ([text](url))
    and horizontal rules so the LLM receives plain text without noise.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text


def read_pdf(path: Path) -> str:
    """Extract text from all pages of a PDF file and join them with newlines."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            pages.append(extracted)
    return "\n".join(pages)


def _flatten(obj: Any) -> str:
    """Recursively convert a JSON structure into human-readable text.

    Dicts become labeled lines (key: value), lists become bullet lines (- item),
    and nested objects recurse. Items are separated by blank lines so the
    text splitter can treat each entry as its own paragraph.
    """
    parts = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            label = key.replace("_", " ").capitalize()
            if isinstance(value, (dict, list)):
                parts.append(f"{label}:")
                parts.append(_flatten(value))
            else:
                parts.append(f"{label}: {value}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                parts.append(_flatten(item))
                parts.append("")
            else:
                parts.append(f"- {item}")
    else:
        parts.append(str(obj))
    return "\n".join(parts)


def read_json(path: Path) -> str:
    """Parse a JSON file and flatten it into human-readable plain text."""
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return _flatten(data)


READERS = {
    ".txt": read_txt,
    ".md": read_md,
    ".pdf": read_pdf,
    ".json": read_json,
}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize and clean raw text extracted from any document format.

    Steps applied in order:
    - NFKC Unicode normalization (decomposes ligatures like ﬁ → fi)
    - Normalize line endings to LF
    - Replace tabs with spaces
    - Remove control characters (except newlines)
    - Collapse multiple consecutive spaces into one
    - Collapse 3+ blank lines into 2
    - Strip trailing whitespace from each line
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
)


def chunk_text(text: str, source: str) -> list[dict]:
    """Split a document into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

    The splitter tries separators in order: paragraph breaks, newlines,
    sentences, words, and finally individual characters. Each chunk is
    returned as a dict with the text content, originating filename, and
    its position index within that document.
    """
    texts = _splitter.split_text(text)
    return [{"text": t, "source": source, "chunk_index": i} for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Embeddings + FAISS
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of text chunks using LangChain's OpenAIEmbeddings.

    Uses the text-embedding-3-small model. Batching and retries are handled
    internally by LangChain. Returns a float32 numpy array of shape (N, 1536).
    """
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    logger.info(f"Embedding {len(texts)} chunks via LangChain OpenAIEmbeddings ...")
    vectors = embedder.embed_documents(texts)
    logger.info("Embedding complete")
    return np.array(vectors, dtype=np.float32)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from the given embedding matrix.

    Vectors are L2-normalized before insertion so that inner product
    is equivalent to cosine similarity during search.
    """
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def load_documents(docs_dir: Path) -> list[dict]:
    """Read and clean all supported documents from a directory.

    Iterates the directory in alphabetical order. Files with unsupported
    extensions are skipped with a warning. Read errors are logged and
    skipped so a single bad file does not abort the entire pipeline.

    Returns a list of dicts with keys 'source' (filename) and 'text' (cleaned content).
    """
    documents = []
    for path in sorted(docs_dir.iterdir()):
        ext = path.suffix.lower()
        if ext not in READERS:
            logger.warning(f"Skipping unsupported file: {path.name}")
            continue
        logger.info(f"Reading {path.name} ...")
        try:
            raw = READERS[ext](path)
            cleaned = clean_text(raw)
            documents.append({"source": path.name, "text": cleaned})
            logger.info(f"  {len(cleaned)} chars after cleaning")
        except Exception as exc:
            logger.error(f"Failed to read {path.name}: {exc}")
    return documents


def ingest(docs_dir: Path = DOCS_DIR, data_dir: Path = DATA_DIR) -> None:
    """Run the full ingestion pipeline: read → clean → chunk → embed → index → persist.

    Reads all supported documents from docs_dir, splits them into overlapping
    chunks, generates OpenAI embeddings for each chunk, builds a FAISS index,
    and saves the index plus chunk metadata to data_dir.

    Requires OPENAI_API_KEY to be set in the environment.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

    data_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(docs_dir)
    if not documents:
        raise RuntimeError(f"No readable documents found in {docs_dir}")

    all_chunks: list[dict] = []
    for doc in documents:
        chunks = chunk_text(doc["text"], source=doc["source"])
        logger.info(f"{doc['source']}: {len(chunks)} chunk(s)")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks to embed: {len(all_chunks)}")

    embeddings = get_embeddings([c["text"] for c in all_chunks])

    index = build_index(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"FAISS index saved → {INDEX_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Metadata saved → {METADATA_PATH} ({len(all_chunks)} chunks)")


if __name__ == "__main__":
    ingest()
