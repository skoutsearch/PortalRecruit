import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# ----------------------------
# Canonical positions
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


# ----------------------------
# Base synonyms (kept small; term groups do most of the work)
# ----------------------------

POSITION_SYNONYMS: Dict[str, Dict[str, Any]] = {
    "point guard": {"primary": "POINT_GUARD", "confidence": 1.00, "type": "numeric"},
    "pg": {"primary": "POINT_GUARD", "confidence": 1.00, "type": "numeric"},
    "shooting guard": {"primary": "SHOOTING_GUARD", "confidence": 1.00, "type": "numeric"},
    "sg": {"primary": "SHOOTING_GUARD", "confidence": 1.00, "type": "numeric"},
    "small forward": {"primary": "SMALL_FORWARD", "confidence": 1.00, "type": "numeric"},
    "sf": {"primary": "SMALL_FORWARD", "confidence": 1.00, "type": "numeric"},
    "power forward": {"primary": "POWER_FORWARD", "confidence": 1.00, "type": "numeric"},
    "pf": {"primary": "POWER_FORWARD", "confidence": 1.00, "type": "numeric"},
    "center": {"primary": "CENTER", "confidence": 1.00, "type": "numeric"},
    "c": {"primary": "CENTER", "confidence": 1.00, "type": "numeric"},
    "wing": {"primary": "SMALL_FORWARD", "secondary": ["SHOOTING_GUARD"], "confidence": 0.90, "type": "slang"},
    "big": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.85, "type": "size"},
    "big man": {"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.90, "type": "slang"},
    "combo guard": {"primary": "GUARD", "secondary": ["POINT_GUARD", "SHOOTING_GUARD"], "confidence": 0.90, "type": "role"},
    "point forward": {"primary": "SMALL_FORWARD", "secondary": ["POINT_GUARD"], "confidence": 0.90, "type": "role"},
    "stretch four": {"primary": "POWER_FORWARD", "confidence": 0.95, "type": "role"},
    "stretch five": {"primary": "CENTER", "confidence": 0.95, "type": "role"},
}


# ----------------------------
# Default priors (overwritten by calibration if you provide data)
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


# ----------------------------
# Term groups
# - patterns: regex patterns, case-insensitive
# - semantic_votes: list of label evidence items
# - size_evidence: None or dict with mode absolute/delta and target distributions
# ----------------------------

def _rx(p: str) -> re.Pattern:
    return re.compile(p, flags=re.IGNORECASE)


TERM_GROUPS: Dict[str, Dict[str, Any]] = {
    # Numeric positions
    "NUM_ONE": {
        "patterns": [_rx(r"\bthe\s*1\b"), _rx(r"\b1\b")],
        "semantic_votes": [{"primary": "POINT_GUARD", "confidence": 0.95}],
        "size_evidence": None,
    },
    "NUM_TWO": {
        "patterns": [_rx(r"\bthe\s*2\b"), _rx(r"\b2\b"), _rx(r"\btwo[-\s]*guard\b")],
        "semantic_votes": [{"primary": "SHOOTING_GUARD", "confidence": 0.95}],
        "size_evidence": None,
    },
    "NUM_THREE": {
        "patterns": [_rx(r"\bthe\s*3\b"), _rx(r"\b3\b")],
        "semantic_votes": [{"primary": "SMALL_FORWARD", "confidence": 0.95}],
        "size_evidence": None,
    },
    "NUM_FOUR": {
        "patterns": [_rx(r"\bthe\s*4\b"), _rx(r"\b4\b")],
        "semantic_votes": [{"primary": "POWER_FORWARD", "confidence": 0.95}],
        "size_evidence": None,
    },
    "NUM_FIVE": {
        "patterns": [_rx(r"\bthe\s*5\b"), _rx(r"\b5\b")],
        "semantic_votes": [{"primary": "CENTER", "confidence": 0.95}],
        "size_evidence": None,
    },

    # Wings
    "WING_GENERAL": {
        "patterns": [
            _rx(r"\bwing\b"),
            _rx(r"\bswingman\b"),
            _rx(r"\b(perimeter\s+forward)\b"),
        ],
        "semantic_votes": [{"primary": "SMALL_FORWARD", "secondary": ["SHOOTING_GUARD"], "confidence": 0.90}],
        "size_evidence": {"mode": "absolute", "h_mu": 77.5, "h_sigma": 2.6, "w_mu": 210.0, "w_sigma": 25.0},
    },

    # Bigs
    "BIG_GENERAL": {
        "patterns": [
            _rx(r"\bbig\b"),
            _rx(r"\bbig\s+man\b"),
            _rx(r"\bpost\b"),
            _rx(r"\bpost\s+player\b"),
            _rx(r"\binterior\b"),
        ],
        "semantic_votes": [{"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.85}],
        "size_evidence": {"mode": "absolute", "h_mu": 81.5, "h_sigma": 2.8, "w_mu": 240.0, "w_sigma": 32.0},
    },

    # Rim roles
    "RIM_PROTECTION": {
        "patterns": [
            _rx(r"\brim\s+protector\b"),
            _rx(r"\bshot\s+block(er|ing)\b"),
            _rx(r"\banchor\b"),
        ],
        "semantic_votes": [{"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.90}],
        "size_evidence": {"mode": "absolute", "h_mu": 82.5, "h_sigma": 2.4, "w_mu": 245.0, "w_sigma": 30.0},
    },

    # PnR big actions
    "ROLL_DIVE": {
        "patterns": [
            _rx(r"\broll\s+man\b"),
            _rx(r"\broller\b"),
            _rx(r"\bdive\s+man\b"),
            _rx(r"\brim\s+runner\b"),
            _rx(r"\bvertical\s+spacer\b"),
        ],
        "semantic_votes": [{"primary": "CENTER", "secondary": ["POWER_FORWARD"], "confidence": 0.85}],
        "size_evidence": {"mode": "absolute", "h_mu": 81.5, "h_sigma": 2.7, "w_mu": 235.0, "w_sigma": 30.0},
    },

    # Stretch bigs
    "STRETCH_BIG": {
        "patterns": [
            _rx(r"\bstretch\s*4\b"),
            _rx(r"\bstretch\s*five\b"),
            _rx(r"\bstretch\s*5\b"),
            _rx(r"\bpick[-\s]*and[-\s]*pop\b"),
        ],
        "semantic_votes": [
            {"primary": "POWER_FORWARD", "secondary": ["CENTER"], "confidence": 0.80},
        ],
        "size_evidence": {"mode": "absolute", "h_mu": 80.5, "h_sigma": 2.6, "w_mu": 232.0, "w_sigma": 28.0},
    },

    # Big guard modifier (delta evidence)
    "BIG_GUARD": {
        "patterns": [_rx(r"\bbig[-\s]+guard\b")],
        "semantic_votes": [{"primary": "GUARD", "secondary": ["SHOOTING_GUARD", "POINT_GUARD"], "confidence": 0.85}],
        "size_evidence": {"mode": "delta", "h_delta_mu": 2.0, "h_delta_sigma": 1.4, "w_delta_mu": 18.0, "w_delta_sigma": 10.0},
    },

    # Small guard modifier (delta evidence)
    "SMALL_GUARD": {
        "patterns": [_rx(r"\bsmall\s+guard\b"), _rx(r"\bquick\s+guard\b")],
        "semantic_votes": [{"primary": "POINT_GUARD", "secondary": ["GUARD"], "confidence": 0.75}],
        "size_evidence": {"mode": "delta", "h_delta_mu": -2.0, "h_delta_sigma": 1.4, "w_delta_mu": -12.0, "w_delta_sigma": 10.0},
    },
}


# ----------------------------
# Probabilistic helpers
# ----------------------------

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


# ----------------------------
# Text extraction
# ----------------------------

def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_group_hits(text: str) -> Dict[str, int]:
    t = normalize_text(text)
    hits: Dict[str, int] = {}
    for gid, g in TERM_GROUPS.items():
        count = 0
        for pat in g["patterns"]:
            if pat.search(t):
                count += 1
        if count > 0:
            hits[gid] = count
    return hits


def extract_base_terms(text: str) -> List[str]:
    t = normalize_text(text)
    # phrase-first for base synonyms
    found: List[str] = []
    for k in sorted(POSITION_SYNONYMS.keys(), key=lambda s: -len(s)):
        if " " in k and k in t:
            found.append(k)
    found.extend(t.split(" ") if t else [])
    # dedup
    out: List[str] = []
    seen = set()
    for x in found:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ----------------------------
# Scoring
# ----------------------------

def score_positions(
    query: str,
    height_in: Optional[float] = None,
    weight_lb: Optional[float] = None,
    alpha_semantic: float = 1.0,
    beta_size: float = 1.0,
    group_semantic_multiplier: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    scores = {p: 0.0 for p in CANONICAL_POSITIONS}
    t = normalize_text(query)

    # 1) Base synonym evidence
    for term in extract_base_terms(t):
        m = POSITION_SYNONYMS.get(term)
        if not m:
            continue
        conf = float(m.get("confidence", 0.7))
        scores[m["primary"]] += alpha_semantic * conf
        for sec in m.get("secondary", []) or []:
            scores[sec] += alpha_semantic * conf * 0.6

    # 2) Term-group semantic votes
    hits = extract_group_hits(t)
    for gid, hitcount in hits.items():
        g = TERM_GROUPS[gid]
        mult = 1.0
        if group_semantic_multiplier and gid in group_semantic_multiplier:
            mult = float(group_semantic_multiplier[gid])

        for vote in g.get("semantic_votes", []) or []:
            conf = float(vote.get("confidence", 0.7))
            primary = vote["primary"]
            scores[primary] += alpha_semantic * conf * mult * min(2.0, 1.0 + 0.15 * (hitcount - 1))
            for sec in vote.get("secondary", []) or []:
                scores[sec] += alpha_semantic * conf * 0.6 * mult * min(2.0, 1.0 + 0.15 * (hitcount - 1))

    # 3) Size evidence (Bayesian posterior + log-likelihood)
    if height_in is not None or weight_lb is not None:
        for pos in CANONICAL_POSITIONS:
            prior_cfg = POSITION_SIZE_PRIORS.get(pos)
            if not prior_cfg:
                continue

            h_post = Gaussian(prior_cfg["h_mu"], prior_cfg["h_sigma"])
            w_post = Gaussian(prior_cfg["w_mu"], prior_cfg["w_sigma"])

            for gid in hits.keys():
                ev = TERM_GROUPS[gid].get("size_evidence")
                if not ev:
                    continue
                mode = ev.get("mode", "absolute")

                if mode == "absolute":
                    if height_in is not None and "h_mu" in ev:
                        h_post = _gaussian_posterior(h_post, Gaussian(float(ev["h_mu"]), float(ev["h_sigma"])))
                    if weight_lb is not None and "w_mu" in ev:
                        w_post = _gaussian_posterior(w_post, Gaussian(float(ev["w_mu"]), float(ev["w_sigma"])))

                elif mode == "delta":
                    if height_in is not None and "h_delta_mu" in ev:
                        h_post = _gaussian_posterior(
                            h_post,
                            Gaussian(h_post.mu + float(ev["h_delta_mu"]), float(ev["h_delta_sigma"]))
                        )
                    if weight_lb is not None and "w_delta_mu" in ev:
                        w_post = _gaussian_posterior(
                            w_post,
                            Gaussian(w_post.mu + float(ev["w_delta_mu"]), float(ev["w_delta_sigma"]))
                        )

            ll = 0.0
            if height_in is not None:
                ll += _norm_logpdf(height_in, h_post)
            if weight_lb is not None:
                ll += _norm_logpdf(weight_lb, w_post)

            # scaling so LL doesn't dominate semantic votes
            scores[pos] += beta_size * (ll * 0.15)

    return scores


def topk(scores: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max(1, k)]


# ----------------------------
# Calibration
# ----------------------------

def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 1.0
    mu = sum(xs) / float(len(xs))
    var = sum((x - mu) * (x - mu) for x in xs) / float(max(1, len(xs) - 1))
    sigma = math.sqrt(max(1e-6, var))
    return mu, sigma


def calibrate_position_priors(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_pos_h: Dict[str, List[float]] = {}
    by_pos_w: Dict[str, List[float]] = {}
    for s in samples:
        pos = s.get("true_position")
        if not pos:
            continue
        h = s.get("height_in")
        w = s.get("weight_lb")
        if h is not None:
            by_pos_h.setdefault(pos, []).append(float(h))
        if w is not None:
            by_pos_w.setdefault(pos, []).append(float(w))

    priors: Dict[str, Dict[str, float]] = {}
    for pos in CANONICAL_POSITIONS:
        # fallback: keep existing defaults if no data
        base = POSITION_SIZE_PRIORS.get(pos, {"h_mu": 0, "h_sigma": 1, "w_mu": 0, "w_sigma": 1})
        h_mu, h_sig = _mean_std(by_pos_h.get(pos, []))
        w_mu, w_sig = _mean_std(by_pos_w.get(pos, []))

        priors[pos] = {
            "h_mu": h_mu if by_pos_h.get(pos) else float(base["h_mu"]),
            "h_sigma": h_sig if by_pos_h.get(pos) else float(base["h_sigma"]),
            "w_mu": w_mu if by_pos_w.get(pos) else float(base["w_mu"]),
            "w_sigma": w_sig if by_pos_w.get(pos) else float(base["w_sigma"]),
        }

    return priors


def calibrate_group_size_evidence(
    samples: List[Dict[str, Any]],
    min_hits: int = 40,
    shrink_k: float = 60.0,
) -> Dict[str, Dict[str, Any]]:
    # Fit evidence based on samples where group hit is present in text.
    # Uses shrinkage: posterior_mu = (n * mu_hat + k * mu_prior) / (n + k)
    # prior is global mean/std across all samples with size present.
    all_h = [float(s["height_in"]) for s in samples if s.get("height_in") is not None]
    all_w = [float(s["weight_lb"]) for s in samples if s.get("weight_lb") is not None]
    global_h_mu, global_h_sig = _mean_std(all_h)
    global_w_mu, global_w_sig = _mean_std(all_w)

    # Precompute per-position means for delta modeling
    pos_means = calibrate_position_priors(samples)

    group_stats: Dict[str, Dict[str, Any]] = {}
    for gid in TERM_GROUPS.keys():
        hs: List[float] = []
        ws: List[float] = []
        deltas_h: List[float] = []
        deltas_w: List[float] = []
        n = 0

        for s in samples:
            text = s.get("text", "")
            if not text:
                continue
            hits = extract_group_hits(text)
            if gid not in hits:
                continue

            n += 1
            pos = s.get("true_position")
            h = s.get("height_in")
            w = s.get("weight_lb")
            if h is not None:
                hs.append(float(h))
                if pos in pos_means:
                    deltas_h.append(float(h) - float(pos_means[pos]["h_mu"]))
            if w is not None:
                ws.append(float(w))
                if pos in pos_means:
                    deltas_w.append(float(w) - float(pos_means[pos]["w_mu"]))

        if n < min_hits:
            continue

        # Decide mode: keep original group mode if defined, else infer.
        original = TERM_GROUPS[gid].get("size_evidence")
        mode = "absolute"
        if original and "mode" in original:
            mode = str(original["mode"])

        if mode == "absolute":
            h_mu, h_sig = _mean_std(hs) if hs else (global_h_mu, global_h_sig)
            w_mu, w_sig = _mean_std(ws) if ws else (global_w_mu, global_w_sig)

            # shrinkage toward global
            h_mu = (len(hs) * h_mu + shrink_k * global_h_mu) / float(len(hs) + shrink_k) if hs else global_h_mu
            w_mu = (len(ws) * w_mu + shrink_k * global_w_mu) / float(len(ws) + shrink_k) if ws else global_w_mu
            h_sig = max(1.0, h_sig)
            w_sig = max(10.0, w_sig)

            group_stats[gid] = {
                "mode": "absolute",
                "h_mu": h_mu,
                "h_sigma": h_sig,
                "w_mu": w_mu,
                "w_sigma": w_sig,
                "n": n,
            }

        elif mode == "delta":
            dh_mu, dh_sig = _mean_std(deltas_h) if deltas_h else (0.0, 1.6)
            dw_mu, dw_sig = _mean_std(deltas_w) if deltas_w else (0.0, 18.0)

            # shrink deltas toward 0
            dh_mu = (len(deltas_h) * dh_mu) / float(len(deltas_h) + shrink_k) if deltas_h else 0.0
            dw_mu = (len(deltas_w) * dw_mu) / float(len(deltas_w) + shrink_k) if deltas_w else 0.0
            dh_sig = max(0.8, dh_sig)
            dw_sig = max(8.0, dw_sig)

            group_stats[gid] = {
                "mode": "delta",
                "h_delta_mu": dh_mu,
                "h_delta_sigma": dh_sig,
                "w_delta_mu": dw_mu,
                "w_delta_sigma": dw_sig,
                "n": n,
            }

    return group_stats


def learn_global_weights_logreg(
    samples: List[Dict[str, Any]],
    candidate_positions: Optional[List[str]] = None,
    max_iter: int = 300,
) -> Dict[str, Any]:
    # Learns alpha_semantic and beta_size via logistic regression over two features:
    #  f1 = semantic_score_only(pos)
    #  f2 = size_ll_only(pos)
    #
    # This is stable and interpretable.
    #
    # Requires scikit-learn. If not available, returns defaults.

    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return {"alpha_semantic": 1.0, "beta_size": 1.0, "note": "scikit-learn not available"}

    pos_list = candidate_positions or [p for p in CANONICAL_POSITIONS if p not in ("GUARD", "FORWARD")]

    X: List[List[float]] = []
    y: List[int] = []
    pos_to_idx = {p: i for i, p in enumerate(pos_list)}

    # Build training rows: one row per (sample, pos) with label 1 if pos == true_position else 0
    for s in samples:
        true_pos = s.get("true_position")
        if true_pos not in pos_to_idx:
            continue
        text = s.get("text", "")
        h = s.get("height_in")
        w = s.get("weight_lb")

        # Compute per-position decomposition
        for p in pos_list:
            sem = score_positions(text, None, None, alpha_semantic=1.0, beta_size=0.0).get(p, 0.0)
            size = score_positions(text, h, w, alpha_semantic=0.0, beta_size=1.0).get(p, 0.0)
            X.append([sem, size])
            y.append(1 if p == true_pos else 0)

    if not X:
        return {"alpha_semantic": 1.0, "beta_size": 1.0, "note": "no training rows"}

    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    clf.fit(X, y)

    # Coefs correspond to multipliers on semantic and size features
    w_sem = float(clf.coef_[0][0])
    w_size = float(clf.coef_[0][1])

    # Convert to positive-friendly scales; clamp to avoid extreme ratios.
    alpha = max(0.1, min(5.0, abs(w_sem)))
    beta = max(0.1, min(5.0, abs(w_size)))

    return {
        "alpha_semantic": alpha,
        "beta_size": beta,
        "raw_coef_sem": w_sem,
        "raw_coef_size": w_size,
        "note": "logreg fit on [semantic, size] features; magnitudes used as scales",
    }


# ----------------------------
# Model bundle I/O
# ----------------------------

def export_model_bundle(
    path: str,
    priors: Dict[str, Dict[str, float]],
    group_size_updates: Dict[str, Dict[str, Any]],
    weights: Dict[str, Any],
) -> None:
    bundle = {
        "position_size_priors": priors,
        "term_groups": {
            gid: {
                "patterns": [p.pattern for p in g["patterns"]],
                "semantic_votes": g.get("semantic_votes", []),
                "size_evidence": group_size_updates.get(gid, g.get("size_evidence")),
            }
            for gid, g in TERM_GROUPS.items()
        },
        "weights": weights,
        "canonical_positions": list(CANONICAL_POSITIONS),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)


def apply_group_size_updates(group_size_updates: Dict[str, Dict[str, Any]]) -> None:
    # Mutates TERM_GROUPS in-place
    for gid, upd in group_size_updates.items():
        if gid in TERM_GROUPS:
            TERM_GROUPS[gid]["size_evidence"] = upd


def apply_position_priors(priors: Dict[str, Dict[str, float]]) -> None:
    POSITION_SIZE_PRIORS.clear()
    for k, v in priors.items():
        POSITION_SIZE_PRIORS[k] = dict(v)


# ----------------------------
# Expected training sample format
# ----------------------------

# Each sample dict:
# {
#   "true_position": "POINT_GUARD" | ...,
#   "height_in": 76.0,
#   "weight_lb": 205.0,
#   "text": "big wing 3-and-d ...",
# }


def calibrate_all(
    samples: List[Dict[str, Any]],
    min_group_hits: int = 40,
) -> Dict[str, Any]:
    priors = calibrate_position_priors(samples)
    apply_position_priors(priors)

    group_updates = calibrate_group_size_evidence(samples, min_hits=min_group_hits)
    apply_group_size_updates(group_updates)

    weights = learn_global_weights_logreg(samples)

    return {
        "priors": priors,
        "group_size_updates": group_updates,
        "weights": weights,
    }


def load_model_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    priors = bundle.get("position_size_priors") or {}
    if priors:
        apply_position_priors(priors)

    term_groups = bundle.get("term_groups") or {}
    group_updates: Dict[str, Dict[str, Any]] = {}
    for gid, group in term_groups.items():
        if isinstance(group, dict) and "size_evidence" in group:
            group_updates[gid] = group.get("size_evidence")

    if group_updates:
        apply_group_size_updates(group_updates)

    return bundle.get("weights") or {}
