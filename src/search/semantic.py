from __future__ import annotations

from functools import lru_cache
from typing import Iterable

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_embedder(model_name: str = EMBED_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = RERANK_MODEL_NAME):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def build_expanded_query(query: str, matched_phrases: Iterable[str] | None = None) -> str:
    query = (query or "").strip()
    if not matched_phrases:
        return query
    seen = set()
    ordered = []
    for phrase in matched_phrases:
        p = (phrase or "").strip()
        if not p:
            continue
        key = p.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(p)
    return query if not ordered else f"{query} | {' | '.join(ordered)}"


def encode_query(query: str) -> list[float]:
    model = get_embedder()
    vec = model.encode([query], normalize_embeddings=True)
    return vec[0].tolist()


def blend_score(vector_distance: float | None, rerank_score: float | None, tag_overlap: int = 0) -> float:
    # Chroma returns cosine distance for hnsw cosine space (smaller is better).
    # Convert to similarity-like component in [roughly 0,1+].
    vector_similarity = 0.0 if vector_distance is None else max(0.0, 1.0 - float(vector_distance))
    rerank = 0.0 if rerank_score is None else float(rerank_score)
    return (0.60 * rerank) + (0.35 * vector_similarity) + (0.05 * float(tag_overlap))
