import os
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OLLAMA_CONFIG = {
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "model": os.getenv("OLLAMA_MODEL", "gemma3:12b"),
    "temperature": 0,
    "keep_alive": 10000,
    "num_gpu": -1,
}

EMBEDDING_MODEL_CONFIG = {"model": os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")}
EMBEDDING_PATH = "data/processed/"

LLM_CACHE_PATH = "data/cache/llm_cache.db"
RAW_DATA_PATH = "docs/"


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "420"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
CHUNK_METHOD = os.getenv("CHUNK_METHOD", "recursive")


CHUNK_TABLE_SIZE = int(os.getenv("CHUNK_TABLE_SIZE", "2000"))
CHUNK_TABLE_OVERLAP = int(os.getenv("CHUNK_TABLE_OVERLAP", "200"))


RERANKER_TYPE = os.getenv("RERANKER_TYPE", "bi").lower()

BIENCODER_RERANKER_MODEL = os.getenv(
    "BIENCODER_RERANKER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
CROSSENCODER_RERANKER_MODEL = os.getenv(
    "CROSSENCODER_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "10"))



API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_VERSION = "1.0.0"
