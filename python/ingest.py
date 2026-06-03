import os
import re
import logging
import unicodedata
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, DOCS_DIR
from embedder import get_embedder
from readers.txt_reader import read_txt
from readers.md_reader import read_md
from readers.pdf_reader import read_pdf
from readers.json_reader import read_json

load_dotenv()

logger = logging.getLogger(__name__)

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
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents(docs_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(p for p in docs_dir.rglob("*") if p.is_file()):
        ext = path.suffix.lower()
        if ext not in READERS:
            logger.warning(f"Skipping unsupported file: {path.name}")
            continue
        logger.info(f"Reading {path.name} ...")
        try:
            raw = READERS[ext](path)
            cleaned = clean_text(raw)
            if not cleaned:
                logger.warning(f"Skipping empty document: {path.name}")
                continue
            documents.append(Document(page_content=cleaned, metadata={"source": path.relative_to(docs_dir).as_posix()}))
            logger.info(f"  {len(cleaned)} chars after cleaning")
        except Exception as exc:
            logger.error(f"Failed to read {path.name}: {exc}")
    return documents


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def ingest(docs_dir: Path = DOCS_DIR, data_dir: Path = DATA_DIR) -> None:
    """Read → clean → chunk → embed → build FAISS vector store → persist."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

    data_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(docs_dir)
    if not documents:
        raise RuntimeError(f"No readable documents found in {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    logger.info(f"Total chunks to embed: {len(chunks)}")

    embedder = get_embedder()
    try:
        vectorstore = FAISS.from_documents(chunks, embedder)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate embeddings via OpenAI: {exc}") from exc
    vectorstore.save_local(str(data_dir))
    logger.info(f"Vector store saved → {data_dir} ({len(chunks)} chunks)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ingest()
