
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    ALL_COLLECTIONS,
    CHROMA_DIR,
    EMBED_MODEL,
    K_PER_COLLECTION,
    RERANKER_MODEL,
    SCORE_THRESHOLD,
    TOP_N_RESULTS,
    USE_RERANKER,
)

logger = logging.getLogger(__name__)


class MergedRetriever:
    def __init__(
        self,
        k_per_collection: int = K_PER_COLLECTION,
        top_n: int = TOP_N_RESULTS,
        score_threshold: float = SCORE_THRESHOLD,
        use_reranker: bool = USE_RERANKER,
    ):
        self.k = k_per_collection
        self.top_n = top_n
        self.score_threshold = score_threshold
        self.use_reranker = use_reranker
        self._reranker = None  # lazy-loaded on first use

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.stores = {
            name: Chroma(
                collection_name=name,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            )
            for name in ALL_COLLECTIONS
        }

    # ── internals ────
    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model %s ...", RERANKER_MODEL)
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    def _gather_candidates(self, query: str) -> list[Document]:
        candidates: list[Document] = []
        seen_content: set[str] = set()
        for name, store in self.stores.items():
            try:
                results = store.similarity_search_with_relevance_scores(
                    query, k=self.k
                )
            except Exception as exc:  # collection missing / empty
                logger.warning("Retrieval from '%s' failed: %s", name, exc)
                continue
            for doc, score in results:
                if score < self.score_threshold:
                    continue
                key = doc.page_content.strip()
                if key in seen_content:
                    continue
                seen_content.add(key)
                doc.metadata["retrieval_score"] = round(float(score), 3)
                candidates.append(doc)
        return candidates

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._get_reranker().predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        for doc, score in ranked:
            doc.metadata["rerank_score"] = round(float(score), 3)
        return [doc for doc, _ in ranked]

    # ── public API ─────
    def retrieve(self, query: str) -> list[Document]:
        candidates = self._gather_candidates(query)
        if not candidates:
            return []
        if self.use_reranker:
            candidates = self._rerank(query, candidates)
        else:
            candidates.sort(
                key=lambda d: d.metadata.get("retrieval_score", 0.0),
                reverse=True,
            )
        return candidates[: self.top_n]


def build_retriever(**kwargs) -> MergedRetriever:
    return MergedRetriever(**kwargs)
