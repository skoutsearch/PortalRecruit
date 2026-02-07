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


def semantic_search(
    collection,
    query: str,
    n_results: int = 15,
    extra_query_terms: Iterable[str] | None = None,
    required_tags: Iterable[str] | None = None,
) -> list[str]:
    """Run semantic search with normalized embeddings + optional rerank blend.

    Returns a list of play_ids ranked best-first.
    """
    expanded_query = build_expanded_query(query, extra_query_terms)
    results = collection.query(
        query_embeddings=[encode_query(expanded_query)],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        return []

    required_tag_set = {t.lower() for t in (required_tags or [])}

    try:
        if docs:
            cross = get_cross_encoder()
            pairs = [[expanded_query, d] for d in docs]
            rerank_scores = cross.predict(pairs)
            ranked = []
            for pid, dist, meta, rerank_score in zip(ids, distances, metadatas, rerank_scores):
                tag_overlap = 0
                if isinstance(meta, dict) and required_tag_set:
                    meta_tags = set(str(meta.get("tags", "")).replace("|", ",").split(","))
                    meta_tags = {t.strip().lower() for t in meta_tags if t and t.strip()}
                    tag_overlap = len(meta_tags.intersection(required_tag_set))
                ranked.append((pid, blend_score(dist, rerank_score, tag_overlap)))
            ranked.sort(key=lambda x: x[1], reverse=True)
            return [r[0] for r in ranked]
    except Exception:
        return ids

    return ids
