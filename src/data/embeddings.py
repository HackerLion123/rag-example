import os
from typing import Iterable, Set

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src import config


def _ensure_persist_path(persist_path: str) -> None:
    os.makedirs(persist_path, exist_ok=True)


def _persisted_faiss_exists(persist_path: str) -> bool:
    index_file = os.path.join(persist_path, "index.faiss")
    pkl_file = os.path.join(persist_path, "index.pkl")
    return os.path.exists(index_file) and os.path.exists(pkl_file)


def generate_embeddings(docs: Iterable[Document], *, persist_path: str | None = None) -> FAISS:
    persist_path = persist_path or config.EMBEDDING_PATH
    _ensure_persist_path(persist_path)

    embeddings = OllamaEmbeddings(**config.EMBEDDING_MODEL_CONFIG)
    db = FAISS.from_documents(list(docs), embeddings)
    db.save_local(persist_path)
    return db


def get_vector_store(*, persist_path: str | None = None) -> FAISS:
    persist_path = persist_path or config.EMBEDDING_PATH
    embeddings = OllamaEmbeddings(**config.EMBEDDING_MODEL_CONFIG)
    if not _persisted_faiss_exists(persist_path):
        raise FileNotFoundError(
            f"No persisted FAISS index found at '{persist_path}'. Run ingestion/embedding creation first."
        )
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


def add_documents_to_store(docs: Iterable[Document], *, persist_path: str | None = None) -> int:
    persist_path = persist_path or config.EMBEDDING_PATH
    _ensure_persist_path(persist_path)

    docs_list = list(docs)

    embeddings = OllamaEmbeddings(**config.EMBEDDING_MODEL_CONFIG)
    if _persisted_faiss_exists(persist_path):
        db = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs_list)
    else:
        db = FAISS.from_documents(docs_list, embeddings)
    db.save_local(persist_path)
    return len(docs_list)


def get_existing_doc_hashes(*, persist_path: str | None = None) -> Set[str]:
    """ Retrieve existing document hashes from the vector store."""
    persist_path = persist_path or config.EMBEDDING_PATH
    if not _persisted_faiss_exists(persist_path):
        return set()

    db = get_vector_store(persist_path=persist_path)
    hashes: Set[str] = set()
    for doc_id in db.index_to_docstore_id.values():
        doc = db.docstore.search(doc_id)
        if isinstance(doc, Document):
            doc_hash = doc.metadata.get("doc_hash")
            if doc_hash:
                hashes.add(doc_hash)
    return hashes
