from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Iterable

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Increase HF Hub timeouts for Streamlit Cloud cold starts
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


@lru_cache(maxsize=1)
def get_embedder(model_name: str = EMBED_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name)
    except Exception:
        # Retry once with a longer timeout window
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "240")
        return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = RERANK_MODEL_NAME):
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(model_name)
    except Exception:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "240")
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
    return _encode_query_cached((query or "").strip())


@lru_cache(maxsize=512)
def _encode_query_cached(query: str) -> list[float]:
    model = get_embedder()
    vec = model.encode([query], normalize_embeddings=True)
    return vec[0].tolist()


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1}


def _parse_tags(meta: dict | None) -> set[str]:
    if not isinstance(meta, dict):
        return set()
    raw_tags = str(meta.get("tags", "")).replace("|", ",")
    return {t.strip().lower() for t in raw_tags.split(",") if t and t.strip()}


def _meta_player_id(meta: dict | None) -> str:
    if not isinstance(meta, dict):
        return ""
    for key in ("player_id", "player", "pid"):
        val = meta.get(key)
        if val is None:
            continue
        sval = str(val).strip()
        if sval:
            return sval
    return ""


def _filter_candidates_by_tags(
    candidates: list[tuple[str, str | None, float | None, dict | None, float]],
    required_tag_set: set[str],
    requested_n: int,
) -> tuple[list[tuple[str, str | None, float | None, dict | None, float]], bool]:
    """Prefer strict tag matches, but degrade gracefully when too sparse."""
    if not required_tag_set:
        return candidates, False

    strict = []
    for row in candidates:
        meta_tags = _parse_tags(row[3])
        if required_tag_set.issubset(meta_tags):
            strict.append(row)

    if len(strict) >= max(3, min(requested_n, 8)):
        return strict, False

    # Fallback: keep plays with partial overlap and prioritize by overlap during scoring.
    partial = []
    for row in candidates:
        meta_tags = _parse_tags(row[3])
        if meta_tags.intersection(required_tag_set):
            partial.append(row)

    if partial:
        return partial, True

    return candidates, True


def _lexical_overlap_score(query_tokens: set[str], doc: str | None, meta: dict | None) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(doc or "")
    if not doc_tokens:
        return 0.0
    tag_tokens = _parse_tags(meta)
    overlap = len(query_tokens.intersection(doc_tokens))
    tag_overlap = len(query_tokens.intersection(tag_tokens))
    return float(overlap) + (0.5 * float(tag_overlap))


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
    diversify_by_player: bool = True,
) -> list[str]:
    """Run semantic search with normalized embeddings + optional rerank blend.

    Returns a list of play_ids ranked best-first.
    """
    expanded_query = build_expanded_query(query, extra_query_terms)
    requested_n = max(int(n_results), 1)
    fetch_n = min(max(requested_n * 3, requested_n), 100)

    results = collection.query(
        query_embeddings=[encode_query(expanded_query)],
        n_results=fetch_n,
        include=["documents", "distances", "metadatas"],
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        return []

    required_tag_set = {str(t).strip().lower() for t in (required_tags or []) if str(t).strip()}
    query_tokens = _tokenize(expanded_query)

    candidates: list[tuple[str, str | None, float | None, dict | None, float]] = []
    for pid, doc, dist, meta in zip(ids, docs, distances, metadatas):
        lexical = _lexical_overlap_score(query_tokens, doc, meta)
        candidates.append((pid, doc, dist, meta, lexical))

    candidates, used_tag_fallback = _filter_candidates_by_tags(candidates, required_tag_set, requested_n)

    # Fast pre-ranking improves precision and reduces cross-encoder workload.
    candidates.sort(
        key=lambda row: ((1.0 - float(row[2])) if row[2] is not None else 0.0) + (0.15 * row[4]),
        reverse=True,
    )
    rerank_pool = candidates[: min(len(candidates), max(requested_n * 2, requested_n))]

    try:
        if rerank_pool:
            cross = get_cross_encoder()
            pairs = [[expanded_query, (doc or "")] for _, doc, _, _, _ in rerank_pool]
            rerank_scores = cross.predict(pairs, batch_size=16)
            ranked = []
            for (pid, _doc, dist, meta, lexical), rerank_score in zip(rerank_pool, rerank_scores):
                tag_overlap = 0
                if required_tag_set:
                    meta_tags = _parse_tags(meta)
                    tag_overlap = len(meta_tags.intersection(required_tag_set))
                score = blend_score(dist, float(rerank_score), tag_overlap) + (0.10 * lexical)
                if required_tag_set and used_tag_fallback:
                    score += 0.08 * float(tag_overlap)
                ranked.append((pid, score))
            ranked.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [r[0] for r in ranked]
            if not diversify_by_player:
                return ranked_ids[:requested_n]
            # Light diversity pass: avoid flooding top results with one player.
            max_per_player = 2 if requested_n >= 12 else 1
            counts: dict[str, int] = {}
            selected: list[str] = []
            by_id_meta = {pid: meta for pid, _doc, _dist, meta, _lex in rerank_pool}
            for pid in ranked_ids:
                pkey = _meta_player_id(by_id_meta.get(pid)) or "__unknown__"
                if counts.get(pkey, 0) >= max_per_player:
                    continue
                selected.append(pid)
                counts[pkey] = counts.get(pkey, 0) + 1
                if len(selected) >= requested_n:
                    break
            if len(selected) < requested_n:
                for pid in ranked_ids:
                    if pid in selected:
                        continue
                    selected.append(pid)
                    if len(selected) >= requested_n:
                        break
            return selected
    except Exception:
        return [row[0] for row in candidates[:requested_n]]

    return [row[0] for row in candidates[:requested_n]]
