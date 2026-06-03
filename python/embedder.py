from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL, EMBEDDING_TIMEOUT


def get_embedder() -> Embeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, timeout=EMBEDDING_TIMEOUT)
