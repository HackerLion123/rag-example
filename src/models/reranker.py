from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

from src import config
from src.models.schemas import RerankResult




@lru_cache(maxsize=2)
def _get_st_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class SentenceTransformerReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.RERANKER_MODEL

    def rerank(self, query: str, documents: Sequence[Document]) -> List[RerankResult]:
        if not query.strip() or not documents:
            return []

        model = _get_st_model(self.model_name)

        doc_texts = [d.page_content or "" for d in documents]
        query_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        doc_embs = model.encode(doc_texts, convert_to_tensor=True, normalize_embeddings=True)

        scores = util.cos_sim(query_emb, doc_embs)[0]
        ranked_indices = scores.argsort(descending=True).tolist()

        return [
            RerankResult(document=documents[i], score=float(scores[i].item()))
            for i in ranked_indices
        ]


def rerank_documents(query: str, documents: Iterable[Document], *, top_k: int = 3) -> List[Document]:
    docs_list = list(documents)
    if not docs_list:
        return []

    reranker = SentenceTransformerReranker()
    ranked = reranker.rerank(query, docs_list)
    if not ranked:
        return docs_list[:top_k]

    return [r.document for r in ranked[: max(1, top_k)]]
