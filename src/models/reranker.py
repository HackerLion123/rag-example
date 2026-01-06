from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from src import config
from src.models.schemas import RerankResult

import math


@lru_cache(maxsize=2)
def _get_biencoder_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _get_crossencoder_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class SentenceTransformerReranker:
    """
    Bi-encoder reranker: embeds query and docs separately, then cosine-sim.
    """

    def __init__(self) -> None:
        default_model = config.BIENCODER_RERANKER_MODEL
        self.model_name = default_model

    def rerank(self, query: str, documents: Sequence[Document]) -> List[RerankResult]:
        if not query.strip() or not documents:
            return []

        model = _get_biencoder_model(self.model_name)

        doc_texts = [d.page_content or "" for d in documents]
        query_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        doc_embs = model.encode(doc_texts, convert_to_tensor=True, normalize_embeddings=True)

        scores = util.cos_sim(query_emb, doc_embs)[0]
        ranked_indices = scores.argsort(descending=True).tolist()

        return [
            RerankResult(document=documents[i], score=float(scores[i].item()))
            for i in ranked_indices
        ]


class CrossEncoderReranker:
    """
    Cross-encoder reranker: scores (query, doc) pairs directly.
    """

    def __init__(self, model_name: str | None = None) -> None:
        default_model = config.CROSSENCODER_RERANKER_MODEL
        self.model_name = model_name or default_model

    def rerank(self, query: str, documents: Sequence[Document]) -> List[RerankResult]:
        if not query.strip() or not documents:
            return []

        model = _get_crossencoder_model(self.model_name)

        doc_texts = [d.page_content or "" for d in documents]
        pairs = [(query, text) for text in doc_texts]

        scores = model.predict(pairs)
        
        # applying sigmoid to convert logits to [0,1] scores
        scores_list = scores_list = [1.0 / (1.0 + math.exp(-score)) for score in scores]

        ranked_indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)

        return [
            RerankResult(document=documents[i], score=scores_list[i])
            for i in ranked_indices
        ]


def create_reranker():
    ranker_type = getattr(config, "RERANKER_TYPE", "bi").lower().strip()

    if ranker_type in {"cross", "cross-encoder", "cross_encoder", "crossencoder"}:
        return CrossEncoderReranker()

    return SentenceTransformerReranker()


def rerank_documents(query: str, documents: Iterable[Document], *, top_k: int = 3) -> List[Document]:
    docs_list = list(documents)
    if not docs_list:
        return []

    reranker = create_reranker()
    ranked = reranker.rerank(query, docs_list)
    
    if not ranked:
        return docs_list[: max(1, top_k)]
    
    return [r.document for r in ranked[: max(1, top_k)]]