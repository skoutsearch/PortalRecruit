import argparse
import json
import math
import os
import sqlite3
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.processing.play_tagger import tag_play

DOG_TAGS = ["oreb", "loose_ball", "charge_taken", "deflection", "steal", "block"]
DEFAULT_WEIGHTS = {
    "oreb": 1.5,
    "steal": 2.0,
    "block": 1.5,
    "charge_taken": 3.0,
    "loose_ball": 1.0,
    "deflection": 1.0,
}
DEFAULT_OUT = REPO_ROOT / "models" / "dog_index_weights.json"


def compute_weights(total_events: int, tag_counts: dict, strategy: str = "idf", min_count: int = 50) -> dict:
    raw = {}
    for tag in DOG_TAGS:
        count = tag_counts.get(tag, 0)
        if count < min_count:
            raw_val = DEFAULT_WEIGHTS.get(tag, 1.0)
        else:
            if strategy == "idf":
                # Higher weight for rarer events
                raw_val = math.log((total_events + 1) / (count + 1)) + 1.0
            elif strategy == "sqrt_inv":
                raw_val = math.sqrt((total_events + 1) / (count + 1))
            else:
                raw_val = 1.0
        raw[tag] = raw_val

    # Normalize to a readable scale (mean ~1.5)
    mean_raw = sum(raw.values()) / max(1, len(raw))
    scale = 1.5 / mean_raw if mean_raw else 1.0
    weights = {k: round(v * scale, 3) for k, v in raw.items()}
    return weights


def main():
    parser = argparse.ArgumentParser(description="Train/tune Dog Index weights from play tags.")
    parser.add_argument("--db", default=os.path.join(os.getcwd(), "data/skout.db"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--strategy", default="idf", choices=["idf", "sqrt_inv", "uniform"])
    parser.add_argument("--min-count", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        raise FileNotFoundError(f"DB not found: {args.db}")

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT description
        FROM plays
        WHERE player_id IS NOT NULL
        """
    )
    rows = cur.fetchall()

    total_events = 0
    tag_counts = {k: 0 for k in DOG_TAGS}

    for (desc,) in rows:
        tags = set(tag_play(desc))
        if "non_possession" in tags:
            continue
        total_events += 1
        for tag in DOG_TAGS:
            if tag in tags:
                tag_counts[tag] += 1

    conn.close()

    weights = compute_weights(total_events, tag_counts, args.strategy, args.min_count)

    payload = {
        "strategy": args.strategy,
        "total_events": total_events,
        "tag_counts": tag_counts,
        "weights": weights,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Wrote weights to {out_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
