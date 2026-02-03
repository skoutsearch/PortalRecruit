"""Autocomplete helpers for coach-speak queries."""
from __future__ import annotations

from src.search.coach_dictionary import PHRASES


def all_phrases() -> list[str]:
    phrases = []
    for items in PHRASES.values():
        phrases.extend(items)
    # de-dupe + sort
    return sorted(set(phrases))


def suggest(prefix: str, limit: int = 8) -> list[str]:
    p = (prefix or "").lower().strip()
    if not p or len(p) < 2:
        return []
    out = [s for s in all_phrases() if p in s]
    return out[:limit]


def suggest_rich(prefix: str, limit: int = 25) -> list[str]:
    """Richer autocomplete: includes related phrases from matched buckets."""
    p = (prefix or "").lower().strip()
    if not p or len(p) < 2:
        return []

    from src.search.coach_dictionary import PHRASES

    # direct matches
    direct = [s for s in all_phrases() if p in s]

    # related bucket phrases: any bucket containing a direct match
    related = []
    for bucket, phrases in PHRASES.items():
        if any(p in phrase for phrase in phrases) or any(d in phrase for d in direct for phrase in phrases):
            related.extend(phrases)

    # rank: startswith > contains
    def score(s: str) -> int:
        return 2 if s.startswith(p) else 1 if p in s else 0

    merged = list(dict.fromkeys(direct + related))
    merged.sort(key=lambda s: (-score(s), s))
    return merged[:limit]
