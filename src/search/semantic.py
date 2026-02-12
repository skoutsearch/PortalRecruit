from __future__ import annotations

import os
import re
import sqlite3
from functools import lru_cache
from typing import Iterable

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Increase HF Hub timeouts for Streamlit Cloud cold starts
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


@lru_cache(maxsize=1)
def _load_position_weights() -> tuple[float, float]:
    try:
        from src.position_calibration import load_model_bundle

        weights = load_model_bundle("position_model.json")
        alpha = float(weights.get("alpha_semantic", 1.0))
        beta = float(weights.get("beta_size", 1.0))
        print(f"Loaded Position Model (alpha={alpha:.3f}, beta={beta:.3f})")
        return alpha, beta
    except Exception:
        return 1.0, 1.0


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


def expand_query_terms(query: str) -> list[str]:
    q = (query or "").lower()
    terms = []
    synonym_map = {
        "athletic": ["explosive", "quick twitch", "springy", "vertical pop"],
        "big man": ["big", "center", "frontcourt", "rim protector", "paint"],
        "stretch": ["pick and pop", "spacing", "trail three"],
        "shoot": ["3pt", "catch and shoot", "shot making", "spacing"],
        "rim protect": ["shot block", "anchor", "backline"],
        "rebound": ["glass", "board", "clean the glass"],
        "downhill": ["rim pressure", "paint touch", "drive"],
        "playmaker": ["creator", "facilitator", "unselfish"],
        "defend": ["stopper", "menace", "lockdown"],
        "switchable": ["versatile", "multi-positional"],
        "huge": ["heavy", "strong"],
        "massive": ["heavy", "strong"],
        "tiny": ["undersized", "skinny"],
        "small": ["undersized", "skinny"],
        "lanky": ["lanky", "skinny"],
        "athletic build": ["athletic", "vertical"],
    }
    for key, vals in synonym_map.items():
        if key in q:
            terms.extend(vals)
    # phrase-based expansions
    if "big" in q and "shoot" in q:
        terms.extend(["stretch 5", "pick and pop", "trail 3", "shooting big"])
    if "athletic" in q and "big" in q:
        terms.extend(["rim run", "lob threat", "vertical spacer"])
    if "wing" in q and ("defend" in q or "stopper" in q):
        terms.extend(["point of attack", "chase over", "screen navigation"])
    if "high iq" in q or "smart" in q:
        terms.extend(["decision making", "reads", "processing speed"])
    return list(dict.fromkeys([t for t in terms if t]))


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "for", "of", "in", "on", "with", "at", "by",
    "from", "into", "up", "down", "over", "under", "is", "are", "was", "were", "be", "been",
    "can", "could", "should", "would", "i", "we", "they", "you", "he", "she", "it", "that",
}


def _normalize_query(query: str) -> str:
    q = (query or "").strip().lower()
    if not q:
        return ""
    tokens = [t for t in re.findall(r"[a-z0-9']+", q) if t and t not in _STOPWORDS]
    return " ".join(tokens) if tokens else q


def encode_query(query: str) -> list[float]:
    return _encode_query_cached(_normalize_query(query))


@lru_cache(maxsize=512)
def _encode_query_cached(query: str) -> list[float]:
    model = get_embedder()
    vec = model.encode([query], normalize_embeddings=True)
    return vec[0].tolist()


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1}


def _expand_synonyms_for_embedding(query: str) -> list[str]:
    q = (query or "").lower()
    synonyms = {
        "attack": ["drive", "downhill"],
        "rim": ["paint", "basket"],
        "shoot": ["shot", "jumper"],
        "rebound": ["glass", "boards"],
        "defend": ["stop", "contain"],
    }
    expanded = []
    for key, vals in synonyms.items():
        if key in q:
            expanded.extend(vals[:2])
    return expanded


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
    action_boost = _action_keyword_boost(doc or "")
    return float(overlap) + (0.5 * float(tag_overlap)) + action_boost


def _action_keyword_boost(doc: str) -> float:
    text = (doc or "").lower()
    positive = ["made 3", "make 3", "dunk", "assist", "layup", "rim", "block", "steal"]
    negative = ["kicked ball", "foul", "turnover", "violation"]
    boost = 0.0
    for term in positive:
        if term in text:
            boost += 0.3
    for term in negative:
        if term in text:
            boost -= 0.3
    return boost


def _phrase_boost(doc: str | None, phrases: Iterable[str]) -> float:
    if not doc:
        return 0.0
    lower = doc.lower()
    boost = 0.0
    for phrase in phrases:
        if phrase and phrase.lower() in lower:
            boost += 0.15
    return boost


def _adjective_boost(query: str) -> float:
    q = (query or "").lower()
    adjectives = ["unselfish", "clutch", "athletic", "tough", "physical", "explosive", "lengthy", "rangy"]
    return 0.10 if any(a in q for a in adjectives) else 0.0


def blend_score(vector_distance: float | None, rerank_score: float | None, tag_overlap: int = 0) -> float:
    # Chroma returns cosine distance for hnsw cosine space (smaller is better).
    # Normalize distance into similarity in [0,1].
    if vector_distance is None:
        vector_similarity = 0.0
    else:
        dist = float(vector_distance)
        dist = min(max(dist, 0.0), 2.0)
        vector_similarity = 1.0 - (dist / 2.0)
    rerank = 0.0 if rerank_score is None else float(rerank_score)
    return (0.55 * rerank) + (0.35 * vector_similarity) + (0.10 * float(tag_overlap))


@lru_cache(maxsize=1)
def _load_position_lookup() -> dict[str, str]:
    db_path = os.path.join(os.getcwd(), "data/skout.db")
    if not os.path.exists(db_path):
        return {}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT player_id, full_name, position FROM players")
        rows = cur.fetchall()
        lookup = {}
        for pid, name, pos in rows:
            if pid:
                lookup[str(pid)] = str(pos or "")
            if name:
                lookup[str(name).lower()] = str(pos or "")
        # add play_player_id mapping if table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='player_id_map'")
        if cur.fetchone():
            cur.execute(
                """
                SELECT m.play_player_id, p.position
                FROM player_id_map m
                JOIN players p ON p.player_id = m.player_id
                WHERE m.play_player_id IS NOT NULL
                """
            )
            for play_pid, pos in cur.fetchall():
                if play_pid:
                    lookup[str(play_pid)] = str(pos or "")
        conn.close()
    except Exception:
        return {}
    return lookup


def _position_match_boost(query_terms: set[str], meta: dict | None) -> float:
    if not meta or not query_terms:
        return 0.0
    q_terms = {t.upper() for t in query_terms}
    pos = str(meta.get("position") or "").upper()
    if not pos:
        lookup = _load_position_lookup()
        pid = str(meta.get("player_id") or "")
        pname = str(meta.get("player_name") or "").lower()
        pos = lookup.get(pid) or lookup.get(pname) or ""
        pos = pos.upper()
    if not pos and meta.get("player_name"):
        try:
            db_path = os.path.join(os.getcwd(), "data/skout.db")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            last = str(meta.get("player_name") or "").split(" ")[-1]
            cur.execute("SELECT position FROM players WHERE full_name LIKE ? LIMIT 1", (f"%{last}%",))
            row = cur.fetchone()
            conn.close()
            if row and row[0]:
                pos = str(row[0]).upper()
        except Exception:
            pos = pos or ""
    if not pos:
        guard_terms = {"GUARD", "PG", "SG", "POINT", "1"}
        forward_terms = {"FORWARD", "SF", "PF", "WING", "3", "4"}
        center_terms = {"CENTER", "C", "5"}
        if q_terms.intersection(guard_terms | forward_terms | center_terms):
            return 0.05
        return 0.0
    try:
        from src.position_calibration import map_db_to_canonical
        canonical = set(map_db_to_canonical(pos))
    except Exception:
        canonical = set()
    if not canonical:
        return 0.0

    guard_terms = {"GUARD", "PG", "SG", "POINT", "1"}
    forward_terms = {"FORWARD", "SF", "PF", "WING", "3", "4"}
    center_terms = {"CENTER", "C", "5"}

    if q_terms.intersection(center_terms) and "CENTER" in canonical:
        return 0.10
    if q_terms.intersection(guard_terms) and canonical.intersection({"GUARD", "POINT_GUARD", "SHOOTING_GUARD"}):
        return 0.10
    if q_terms.intersection(forward_terms) and canonical.intersection({"FORWARD", "SMALL_FORWARD", "POWER_FORWARD"}):
        return 0.10
    return 0.0


def hybrid_score(
    vector_distance: float | None,
    rerank_score: float | None,
    tag_overlap: int,
    lexical: float,
    phrase_boost: float,
    adj_boost: float,
    position_boost: float = 0.0,
    biometric_boost: float = 0.0,
    tag_fallback_boost: float = 0.0,
) -> tuple[float, dict]:
    vector = blend_score(vector_distance, rerank_score, tag_overlap)
    keyword = (0.10 * float(lexical)) + float(phrase_boost) + float(adj_boost) + float(tag_fallback_boost)
    total = vector + keyword + float(position_boost) + float(biometric_boost)
    total = max(0.0, min(1.0, total))
    breakdown = {
        "total": total,
        "vector": vector,
        "keyword": keyword,
        "position_boost": float(position_boost),
        "biometric_boost": float(biometric_boost),
    }
    return total, breakdown


def semantic_search(
    collection,
    query: str,
    n_results: int = 15,
    extra_query_terms: Iterable[str] | None = None,
    required_tags: Iterable[str] | None = None,
    boost_tags: Iterable[str] | None = None,
    diversify_by_player: bool = True,
    meta_filters: dict[str, set[str]] | None = None,
    biometric_tags: dict[str, set[str]] | None = None,
    strict_positions: set[str] | None = None,
    return_breakdowns: bool = False,
    alpha_override: float | None = None,
    beta_override: float | None = None,
) -> list[str] | tuple[list[str], dict[str, dict]]:
    """Run semantic search with normalized embeddings + optional rerank blend.

    Returns a list of play_ids ranked best-first.
    """
    from src.position_calibration import score_positions, topk
    synonym_terms = _expand_synonyms_for_embedding(query)
    expanded_query = build_expanded_query(query, list(extra_query_terms or []) + synonym_terms)
    requested_n = max(int(n_results), 1)
    fetch_n = min(max(requested_n * 4, requested_n), 150)

    where_filter = None
    try:
        alpha, beta = _load_position_weights()
        if alpha_override is not None:
            alpha = float(alpha_override)
        if beta_override is not None:
            beta = float(beta_override)
        scores = score_positions(query, alpha_semantic=alpha, beta_size=beta)
        top = topk(scores, k=1)
        if top:
            canon, score = top[0]
            max_score = max(scores.values()) or 1.0
            conf = float(score) / float(max_score)
            if conf > 0.8:
                if canon == "CENTER":
                    where_filter = {"position": {"$in": ["C", "F/C"]}}
                    print("Detected Canonical Position: CENTER")
                elif canon == "POINT_GUARD":
                    where_filter = {"position": {"$in": ["PG", "G"]}}
                    print("Detected Canonical Position: POINT_GUARD")
                elif canon == "SHOOTING_GUARD":
                    where_filter = {"position": {"$in": ["SG", "G"]}}
                    print("Detected Canonical Position: SHOOTING_GUARD")
                elif canon == "POWER_FORWARD":
                    where_filter = {"position": {"$in": ["PF", "F", "F/C"]}}
                    print("Detected Canonical Position: POWER_FORWARD")
                elif canon == "SMALL_FORWARD":
                    where_filter = {"position": {"$in": ["SF", "F", "F/G"]}}
                    print("Detected Canonical Position: SMALL_FORWARD")
    except Exception:
        pass

    try:
        query_vec = encode_query(expanded_query)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=fetch_n,
            include=["documents", "distances", "metadatas"],
            where=where_filter,
        )
        ids = results.get("ids", [[]])[0]
        if where_filter and not ids:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=fetch_n,
                include=["documents", "distances", "metadatas"],
            )
    except Exception:
        return ([], {}) if return_breakdowns else []

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        return ([], {}) if return_breakdowns else []

    required_tag_set = {str(t).strip().lower() for t in (required_tags or []) if str(t).strip()}
    boost_tag_set = {str(t).strip().lower() for t in (boost_tags or []) if str(t).strip()}
    query_tokens = _tokenize(expanded_query)
    query_terms = {t.lower() for t in query_tokens}
    meta_filters = meta_filters or {}

    strict_candidates: list[tuple[str, str | None, float | None, dict | None, float]] = []
    candidates: list[tuple[str, str | None, float | None, dict | None, float]] = []
    strict_pos_terms = {"CENTER", "POINT", "PG", "SG", "GUARD", "FORWARD", "SF", "PF", "WING", "5", "1"}
    has_strict = any(t in query_terms for t in strict_pos_terms)

    for pid, doc, dist, meta in zip(ids, docs, distances, metadatas):
        if meta_filters and isinstance(meta, dict):
            skip = False
            for key, allowed in meta_filters.items():
                if allowed and str(meta.get(key, "")).lower() not in {a.lower() for a in allowed}:
                    skip = True
                    break
            if skip:
                continue
        lexical = _lexical_overlap_score(query_tokens, doc, meta)
        if isinstance(meta, dict):
            pos = str(meta.get("position") or "").upper()
            is_center = "C" in pos or "F/C" in pos
            is_guard = "PG" in pos or "SG" in pos or pos == "G"
            is_forward = "SF" in pos or "PF" in pos or pos == "F"
            if ("CENTER" in query_terms or "5" in query_terms or "C" in query_terms) and is_center:
                strict_candidates.append((pid, doc, dist, meta, lexical))
            if ("POINT" in query_terms or "PG" in query_terms or "1" in query_terms) and is_guard:
                strict_candidates.append((pid, doc, dist, meta, lexical))
            if ("FORWARD" in query_terms or "SF" in query_terms or "PF" in query_terms or "WING" in query_terms) and is_forward:
                strict_candidates.append((pid, doc, dist, meta, lexical))

        candidates.append((pid, doc, dist, meta, lexical))

    if has_strict and strict_candidates:
        candidates = strict_candidates

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
            breakdowns: dict[str, dict] = {}
            phrase_terms = ["point guards", "clutch"]
            adj_boost = _adjective_boost(query)
            for (pid, doc, dist, meta, lexical), rerank_score in zip(rerank_pool, rerank_scores):
                meta_tags = _parse_tags(meta)
                tag_overlap = 0
                if required_tag_set:
                    tag_overlap = len(meta_tags.intersection(required_tag_set))
                elif boost_tag_set:
                    tag_overlap = len(meta_tags.intersection(boost_tag_set))
                phrase_boost = _phrase_boost(doc, phrase_terms)
                position_boost = _position_match_boost(query_terms, meta if isinstance(meta, dict) else None)
                tag_fallback_boost = 0.0
                if required_tag_set and used_tag_fallback:
                    tag_fallback_boost += 0.08 * float(tag_overlap)
                if boost_tag_set and tag_overlap:
                    tag_fallback_boost += 0.05 * float(tag_overlap)
                bio_boost = 0.0
                if isinstance(meta, dict) and biometric_tags:
                    bio = str(meta.get("bio_tags") or "").lower()
                    for _tag, allowed in biometric_tags.items():
                        if allowed and any(a in bio for a in allowed):
                            bio_boost += 0.2
                score, breakdown = hybrid_score(
                    dist,
                    float(rerank_score),
                    tag_overlap,
                    lexical,
                    phrase_boost,
                    adj_boost,
                    position_boost=position_boost,
                    biometric_boost=bio_boost,
                    tag_fallback_boost=tag_fallback_boost,
                )
                ranked.append((pid, score))
                breakdowns[pid] = breakdown
            ranked.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [r[0] for r in ranked]
            if not diversify_by_player:
                if return_breakdowns:
                    return ranked_ids[:requested_n], {pid: breakdowns.get(pid, {}) for pid in ranked_ids[:requested_n]}
                return ranked_ids[:requested_n]
            # Light diversity pass: avoid flooding top results with one player.
            max_per_player = 2 if requested_n >= 12 else 1
            counts: dict[str, int] = {}
            selected: list[str] = []
            seen_snippets: set[str] = set()
            by_id_meta = {pid: meta for pid, _doc, _dist, meta, _lex in rerank_pool}
            by_id_doc = {pid: doc for pid, doc, _dist, _meta, _lex in rerank_pool}
            for pid in ranked_ids:
                pkey = _meta_player_id(by_id_meta.get(pid)) or "__unknown__"
                if counts.get(pkey, 0) >= max_per_player:
                    continue
                doc = (by_id_doc.get(pid) or "").lower()
                signature = " ".join(re.findall(r"[a-z0-9]+", doc)[:12])
                if signature and signature in seen_snippets:
                    continue
                if signature:
                    seen_snippets.add(signature)
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
            if return_breakdowns:
                return selected, {pid: breakdowns.get(pid, {}) for pid in selected}
            return selected
    except Exception:
        ids = [row[0] for row in candidates[:requested_n]]
        return (ids, {}) if return_breakdowns else ids

    ids = [row[0] for row in candidates[:requested_n]]
    return (ids, {}) if return_breakdowns else ids
