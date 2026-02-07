from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.translatability import build_training_frame, predict_latest, save_artifacts, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Translatability XGBoost model (EFF/40 next season).")
    parser.add_argument("--db", type=str, default=None, help="Path to sqlite db (default: data/skout.db)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/translatability_xgb.json",
        help="Where to save the xgboost model",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="models/translatability_meta.json",
        help="Where to save model metadata",
    )
    parser.add_argument(
        "--predict-latest",
        action="store_true",
        help="Generate predictions for the latest season and write CSV",
    )
    parser.add_argument(
        "--pred-path",
        type=str,
        default="data/translatability_predictions.csv",
        help="Path to save predictions when --predict-latest is set",
    )
    args = parser.parse_args()

    df = build_training_frame(args.db)
    if df.empty:
        raise RuntimeError("Training frame is empty. Ensure multiple seasons are ingested.")

    result = train_model(df)
    save_artifacts(result, args.model_path, args.meta_path)

    print("✅ Translatability model trained")
    print(f"  Train rows: {result.train_rows}")
    print(f"  Test rows: {result.test_rows}")
    print(f"  Metrics: {result.metrics}")
    print(f"  Saved model: {args.model_path}")
    print(f"  Saved meta: {args.meta_path}")

    if args.predict_latest:
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(args.model_path)
        pred_df = predict_latest(df, model, result.features)
        if pred_df.empty:
            print("  Skipping predictions: no valid latest-season rows")
            return
        os.makedirs(os.path.dirname(args.pred_path), exist_ok=True)
        pred_df.to_csv(args.pred_path, index=False)
        print(f"  Predictions saved: {args.pred_path}")


if __name__ == "__main__":
    main()
