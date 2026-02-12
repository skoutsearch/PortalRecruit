import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Core synonym mapping
# ----------------------------

# Each entry:
#  - primary: canonical position
#  - secondary: optional list of secondary canonical positions
#  - confidence: semantic confidence (0..1)
#  - type: numeric | slang | role | tactical | defense | size
POSITION_SYNONYMS: Dict[str, Dict] = {
    # Guard general
    "guard": {"primary": "GUARD", "confidence": 0.95, "type": "numeric"},
    "backcourt": {"primary": "GUARD", "confidence": 0.90, "type": "slang"},
    "ballhandler": {"primary": "GUARD", "secondary": ["POINT_GUARD", "SMALL_FORWARD"], "confidence": 0.80, "type": "role"},
    "handler": {"primary": "GUARD", "secondary": ["POINT_GUARD", "SMALL_FORWARD"], "confidence": 0.75, "type": "role"},
    "lead guard": {"primary": "POINT_GUARD", "confidence": 0.85, "type": "role"},
    "combo guard": {"primary": "GUARD", "secondary": ["POINT_GUARD", "SHOOTING_GUARD"], "confidence": 0.90, "type": "role"},
    "initiator": {"primary": "GUARD", "secondary": ["POINT_GUARD", "SMALL_FORWARD"], "confidence": 0.75, "type": "role"},
    "playmaker": {"primary": "GUARD", "secondary": ["POINT_GUARD", "SMALL_FORWARD"], "confidence": 0.75, "type": "role"},
    "primary handler": {"primary": "POINT_GUARD", "confidence": 0.85, "type": "role"},
    "secondary handler": {"primary": "GUARD", "secondary": ["POINT_GUARD", "SHOOTING_GUARD", "SMALL_FORWARD"], "confidence": 0.70, "type": "role"},
    "floor general": {"primary": "POINT_GUARD", "confidence": 0.95, "type": "slang"},
    "point of attack": {"primary": "GUARD", "secondary": ["POINT_GUARD", "SHOOTING_GUARD"], "confidence": 0.80, "type": "defense"},
    "poa": {"primary": "GUARD", "secondary": ["POINT_GUARD", "SHOOTING_GUARD"], "confidence": 0.80, "type": "defense"},

    # Point guard
    "point guard": {"primary": "POINT_GUARD", "confidence": 1.00, "type": "numeric"},
    "pg": {"primary": "POINT_GUARD", "confidence": 1.00, "type": "numeric"},
    "the 1": {"primary": "POINT_GUARD", "confidence": 0.95, "type": "slang"},
    "1": {"primary": "POINT_GUARD", "confidence": 0.95, "type": "numeric"},
    "table setter": {"primary": "POINT_GUARD", "confidence": 0.90, "type": "slang"},
    "orchestrator": {"primary": "POINT_GUARD", "confidence": 0.85, "type": "role"},
    "engine": {"primary": "POINT_GUARD", "confidence": 0.80, "type": "role"},
    "pace controller": {"primary": "POINT_GUARD", "confidence": 0.90, "type": "role"},
    "pick and roll guard": {"primary": "POINT_GUARD", "confidence": 0.95, "type": "tactical"},
    "pnr operator": {"primary": "POINT_GUARD", "confidence": 0.90, "type": "tactical"},
    "pure point": {"primary": "POINT_GUARD", "confidence": 0.95, "type": "slang"},

    # Shooting guard
    "shooting guard": {"primary": "SHOOTING_GUARD", "confidence": 1.00, "type": "numeric"},
    "sg": {"primary": "SHOOTING_GUARD", "confidence": 1.00, "type": "numeric"},
    "the 2": {"primary": "SHOOTING_GUARD", "confidence": 0.95, "type": "slang"},
    "2": {"primary": "SHOOTING_GUARD", "confidence": 0.95, "type": "numeric"},
    "two guard": {"primary": "SHOOTING_GUARD", "confidence": 0.95, "type": "slang"},
    "off guard": {"primary": "SHOOTING_GUARD", "confidence": 0.90, "type": "role"},
    "scoring guard": {"primary": "SHOOTING_GUARD", "confidence": 0.85, "type": "role"},
    "shooter": {"primary": "SHOOTING_GUARD", "secondary": ["SMALL_FORWARD"], "confidence": 0.75, "type": "role"},
    "marksman": {"primary": "SHOOTING_GUARD", "secondary": ["SMALL_FORWARD"], "confidence": 0.80, "type": "slang"},
    "sniper": {"primary": "SHOOTING_GUARD", "secondary": ["SMALL_FORWARD"], "confidence": 0.80, "type": "slang"},
    "microwave": {"primary": "SHOOTING_GUARD", "secondary": ["POINT_GUARD"], "confidence": 0.80, "type": "slang"},
    "catch and shoot": {"primary": "SHOOTING_GUARD", "secondary": ["SMALL_FORWARD"], "confidence": 0.80, "type": "tactical"},
    "movement shooter": {"primary": "SHOOTING_GUARD", "secondary": ["SMALL_FORWARD"], "confidence": 0.85, "type": "tactical"},

    # Forward general
    "forward": {"primary": "FORWARD", "confidence": 0.95, "type": "numeric"},
    "wing": {"primary": "SMALL_FORWARD", "secondary": ["SHOOTING_GUARD"], "confidence": 0.90, "type": "slang"},
    "swingman": {"primary": "SMALL_FORWARD", "secondary": ["SHOOTING_GUARD"], "confidence": 0.90, "type": "slang"},
    "utility forward": {"primary": "FORWARD", "secondary": ["SMALL_FORWARD", "POWER_FORWARD"], "confidence": 0.80, "type": "role"},
    "two way wing": {"primary": "SMALL_FORWARD", "secondary": ["SHOOTING_GUARD"], "confidence": 0.85, "type": "role"},
    "slasher": {"primary": "SMALL_FORWARD", "secondary": ["SHOOTING_GUARD"], "confidence": 0.75, "type": "role"},
    "glue guy": {"primary": "FORWARD", "secondary": ["SMALL_FORWARD"], "confidence": 0.70, "type": "slang"},

    # Small forward
    "small forward": {"primary": "SMALL_FORWARD", "confidence": 1.00, "type": "numeric"},
    "sf": {"primary": "SMALL_FORWARD", "confidence": 1.00, "type": "numeric"},
    "the 3": {"primary": "SMALL_FORWARD", "confidence": 0.95, "type": "slang"},
    "3": {"primary": "SMALL_FORWARD", "confidence": 0.95, "type": "numeric"},
    "3 and d wing": {"primary": "SMALL_FORWARD", "secondary": ["SHOOTING_GUARD"], "confidence": 0.95, "type": "role"},
    "scoring wing": {"primary": "SMALL_FORWARD", "secondary": ["SHOOTING_GUARD"], "confidence": 0.90, "type": "role"},
    "playmaking wing": {"primary": "SMALL_FORWARD", "secondary": ["POINT_GUARD"], "confidence": 0.85, "type": "role"},
    "point forward": {"primary": "SMALL_FORWARD", "secondary": ["POINT_GUARD"], "confidence": 0.90, "type": "role"},
    "isolation wing": {"primary": "SMALL_FORWARD", "secondary": ["SHOOTING_GUARD"], "confidence": 0.85, "type": "tactical"},

    # Power forward
    "power forward": {"primary": "POWER_FORWARD", "confidence": 1.00, "type": "numeric"},
    "pf": {"primary": "POWER_FORWARD", "confidence": 1.00, "type": "numeric"},
    "the 4": {"primary": "POWER_FORWARD", "confidence": 0.95, "type": "slang"},
    "4": {"primary": "POWER_FORWARD", "confidence": 0.95, "type": "numeric"},
    "big forward": {"primary": "POWER_FORWARD", "secondary": ["CENTER"], "confidence": 0.85, "type": "size"},
    "stretch four": {"primary": "POWER_FORWARD", "confidence": 0.95, "type": "role"},
    "pick and pop four": {"primary": "POWER_FORWARD", "confidence": 0.90, "type": "tactical"},
    "face up four": {"primary": "POWER_FORWARD", "confidence": 0.85, "type": "role"},
    "interior forward": {"primary": "POWER_FORWARD", "secondary": ["CENTER"], "confidence": 0.80, "type": "role"},
    "banger": {"primary": "POWER_FORWARD", "secondary": ["CENTER"], "confidence": 0.70, "type": "slang"},

    # Center
    "center": {"primary": "CENTER", "confidence": 1.00, "type": "numeric"},
    "c": {"primary": "CENTER", "confidence": 1.00, "type": "numeric"},
    "the 5": {"primary": "CENTER", "confidence": 0.95, "type": "slang"},
    "5": {"primary": "CENTER", "confidence": 0.95, "type": "numeric"},
    "big": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.85, "type": "size"},
    "big man": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.90, "type": "slang"},
    "post": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.80, "type": "slang"},
    "post player": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.80, "type": "slang"},
    "rim protector": {"primary": "CENTER", "confidence": 0.95, "type": "role"},
    "anchor": {"primary": "CENTER", "confidence": 0.90, "type": "defense"},
    "rim runner": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.90, "type": "role"},
    "roll man": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.90, "type": "tactical"},
    "dive man": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.90, "type": "tactical"},
    "vertical spacer": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.90, "type": "role"},
    "stretch five": {"primary": "CENTER", "confidence": 0.95, "type": "role"},
    "small ball five": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.85, "type": "role"},
    "point center": {"primary": "CENTER", "confidence": 0.80, "type": "role"},

    # Useful composite phrases (common in scouting)
    "big guard": {"primary": "GUARD", "secondary": ["SHOOTING_GUARD", "POINT_GUARD"], "confidence": 0.85, "type": "size"},
    "small guard": {"primary": "GUARD", "secondary": ["POINT_GUARD"], "confidence": 0.80, "type": "size"},
}

# ----------------------------
# Size priors and term evidence
# ----------------------------

POSITION_SIZE_PRIORS: Dict[str, Dict[str, float]] = {
    "POINT_GUARD": {"h_mu": 73.0, "h_sigma": 2.2, "w_mu": 180.0, "w_sigma": 18.0},
    "SHOOTING_GUARD": {"h_mu": 75.0, "h_sigma": 2.3, "w_mu": 195.0, "w_sigma": 20.0},
    "SMALL_FORWARD": {"h_mu": 78.0, "h_sigma": 2.4, "w_mu": 215.0, "w_sigma": 22.0},
    "POWER_FORWARD": {"h_mu": 80.0, "h_sigma": 2.3, "w_mu": 235.0, "w_sigma": 25.0},
    "CENTER": {"h_mu": 82.5, "h_sigma": 2.4, "w_mu": 250.0, "w_sigma": 28.0},
    "GUARD": {"h_mu": 74.0, "h_sigma": 2.8, "w_mu": 188.0, "w_sigma": 24.0},
    "FORWARD": {"h_mu": 79.0, "h_sigma": 2.8, "w_mu": 225.0, "w_sigma": 28.0},
}

TERM_SIZE_EVIDENCE: Dict[str, Dict[str, float]] = {
    "big": {"mode": "absolute", "h_mu": 81.5, "h_sigma": 2.6, "w_mu": 240.0, "w_sigma": 30.0},
    "big man": {"mode": "absolute", "h_mu": 82.0, "h_sigma": 2.6, "w_mu": 245.0, "w_sigma": 30.0},
    "post": {"mode": "absolute", "h_mu": 81.0, "h_sigma": 3.0, "w_mu": 235.0, "w_sigma": 35.0},
    "post player": {"mode": "absolute", "h_mu": 81.0, "h_sigma": 3.0, "w_mu": 235.0, "w_sigma": 35.0},
    "interior": {"mode": "absolute", "h_mu": 80.5, "h_sigma": 3.0, "w_mu": 230.0, "w_sigma": 35.0},
    "wing": {"mode": "absolute", "h_mu": 77.5, "h_sigma": 2.6, "w_mu": 210.0, "w_sigma": 25.0},
    "swingman": {"mode": "absolute", "h_mu": 77.0, "h_sigma": 2.8, "w_mu": 205.0, "w_sigma": 28.0},
    "perimeter": {"mode": "absolute", "h_mu": 76.0, "h_sigma": 3.0, "w_mu": 200.0, "w_sigma": 30.0},

    "big guard": {"mode": "delta", "h_delta_mu": 2.0, "h_delta_sigma": 1.4, "w_delta_mu": 18.0, "w_delta_sigma": 10.0},
    "small guard": {"mode": "delta", "h_delta_mu": -2.0, "h_delta_sigma": 1.4, "w_delta_mu": -12.0, "w_delta_sigma": 10.0},
    "long": {"mode": "delta", "h_delta_mu": 1.5, "h_delta_sigma": 1.6, "w_delta_mu": 0.0, "w_delta_sigma": 20.0},
    "rangy": {"mode": "delta", "h_delta_mu": 1.5, "h_delta_sigma": 1.6, "w_delta_mu": -5.0, "w_delta_sigma": 18.0},

    "rim protector": {"mode": "absolute", "h_mu": 82.5, "h_sigma": 2.4, "w_mu": 245.0, "w_sigma": 30.0},
    "rim runner": {"mode": "absolute", "h_mu": 81.5, "h_sigma": 2.7, "w_mu": 235.0, "w_sigma": 30.0},
    "roll man": {"mode": "absolute", "h_mu": 81.5, "h_sigma": 2.7, "w_mu": 235.0, "w_sigma": 30.0},
    "dive man": {"mode": "absolute", "h_mu": 81.5, "h_sigma": 2.7, "w_mu": 235.0, "w_sigma": 30.0},

    "stretch four": {"mode": "absolute", "h_mu": 79.5, "h_sigma": 2.4, "w_mu": 225.0, "w_sigma": 25.0},
    "stretch five": {"mode": "absolute", "h_mu": 82.0, "h_sigma": 2.6, "w_mu": 240.0, "w_sigma": 28.0},
}

# ----------------------------
# Utilities
# ----------------------------

CANONICAL_POSITIONS: Tuple[str, ...] = (
    "POINT_GUARD",
    "SHOOTING_GUARD",
    "SMALL_FORWARD",
    "POWER_FORWARD",
    "CENTER",
    "GUARD",
    "FORWARD",
)

@dataclass
class Gaussian:
    mu: float
    sigma: float


def _norm_logpdf(x: float, g: Gaussian) -> float:
    if g.sigma <= 0:
        return float("-inf")
    z = (x - g.mu) / g.sigma
    return -0.5 * (math.log(2.0 * math.pi) + 2.0 * math.log(g.sigma) + z * z)


def _gaussian_posterior(prior: Gaussian, like: Gaussian) -> Gaussian:
    v0 = prior.sigma * prior.sigma
    v1 = like.sigma * like.sigma
    if v0 <= 0 or v1 <= 0:
        return prior
    v_post = 1.0 / (1.0 / v0 + 1.0 / v1)
    mu_post = v_post * (prior.mu / v0 + like.mu / v1)
    return Gaussian(mu=mu_post, sigma=math.sqrt(v_post))


def _tokenize(text: str) -> List[str]:
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split(" ") if t else []


def _extract_phrases(text: str) -> List[str]:
    # Phrase matching first (e.g., "big man", "pick and roll guard")
    t = text.lower()
    phrases = []
    for k in sorted(POSITION_SYNONYMS.keys(), key=lambda s: -len(s)):
        if " " in k and k in t:
            phrases.append(k)
    for k in sorted(TERM_SIZE_EVIDENCE.keys(), key=lambda s: -len(s)):
        if " " in k and k in t:
            phrases.append(k)
    # Also include single tokens
    phrases.extend(_tokenize(text))
    # Dedup while preserving order
    out = []
    seen = set()
    for p in phrases:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out

# ----------------------------
# Scoring
# ----------------------------


def score_positions_from_terms(
    query: str,
    height_in: Optional[float] = None,
    weight_lb: Optional[float] = None,
    alpha_semantic: float = 1.0,
    beta_size: float = 1.0,
) -> Dict[str, float]:
    """
    Returns a score per canonical position.
    - alpha_semantic scales synonym/term semantic confidence contribution
    - beta_size scales size log-likelihood contribution
    If height/weight are None, size scoring is skipped.
    """

    terms = _extract_phrases(query)
    scores = {p: 0.0 for p in CANONICAL_POSITIONS}

    # 1) Semantic evidence from synonyms
    for term in terms:
        m = POSITION_SYNONYMS.get(term)
        if not m:
            continue
        primary = m["primary"]
        conf = float(m.get("confidence", 0.7))
        scores[primary] += alpha_semantic * conf
        for sec in m.get("secondary", []) or []:
            scores[sec] += alpha_semantic * conf * 0.6  # secondary discount

    # 2) Size evidence (Bayesian update + likelihood) if size known
    if height_in is not None or weight_lb is not None:
        for pos in CANONICAL_POSITIONS:
            prior_cfg = POSITION_SIZE_PRIORS.get(pos)
            if not prior_cfg:
                continue

            h_prior = Gaussian(prior_cfg["h_mu"], prior_cfg["h_sigma"])
            w_prior = Gaussian(prior_cfg["w_mu"], prior_cfg["w_sigma"])
            h_post = h_prior
            w_post = w_prior

            # Apply term-conditioned evidence
            for term in terms:
                ev = TERM_SIZE_EVIDENCE.get(term)
                if not ev:
                    continue
                mode = ev.get("mode", "absolute")

                if mode == "absolute":
                    if height_in is not None:
                        h_like = Gaussian(ev["h_mu"], ev["h_sigma"])
                        h_post = _gaussian_posterior(h_post, h_like)
                    if weight_lb is not None:
                        w_like = Gaussian(ev["w_mu"], ev["w_sigma"])
                        w_post = _gaussian_posterior(w_post, w_like)

                elif mode == "delta":
                    if height_in is not None and "h_delta_mu" in ev:
                        h_like = Gaussian(h_post.mu + ev["h_delta_mu"], ev["h_delta_sigma"])
                        h_post = _gaussian_posterior(h_post, h_like)
                    if weight_lb is not None and "w_delta_mu" in ev:
                        w_like = Gaussian(w_post.mu + ev["w_delta_mu"], ev["w_delta_sigma"])
                        w_post = _gaussian_posterior(w_post, w_like)

            ll = 0.0
            if height_in is not None:
                ll += _norm_logpdf(height_in, h_post)
            if weight_lb is not None:
                ll += _norm_logpdf(weight_lb, w_post)

            # Add scaled size likelihood. Center it slightly so it doesn’t swamp semantics.
            scores[pos] += beta_size * (ll * 0.15)

    return scores


def best_positions(scores: Dict[str, float], top_k: int = 3) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max(1, top_k)]

# ----------------------------
# Export helpers (JSON)
# ----------------------------


def export_mapping_json(path: str) -> None:
    blob = {
        "position_synonyms": POSITION_SYNONYMS,
        "position_size_priors": POSITION_SIZE_PRIORS,
        "term_size_evidence": TERM_SIZE_EVIDENCE,
        "canonical_positions": list(CANONICAL_POSITIONS),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, sort_keys=True)

# ----------------------------
# Example usage
# ----------------------------


def _demo() -> None:
    q = "big wing who can guard 4s and hit threes"
    scores = score_positions_from_terms(q, height_in=79.0, weight_lb=220.0)
    print("Query:", q)
    print("Top:", best_positions(scores, top_k=5))

if __name__ == "__main__":
    _demo()
