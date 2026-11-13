#!/usr/bin/env python
"""
Train Candidate Constrained Section Fusion Model (Phase 2).

Fits ConstrainedSectionFusion on project-disjoint training data enforcing:
- Non-negative section weights (w_i >= 0)
- Sum of weights equals 4.0 (intercept = 0.0)
- Bounded prediction in [0.0, 4.0]

Saves artifact to models/candidate_constrained_fusion_model.joblib
WITHOUT touching models/ridge_fusion_model.joblib or any hybrid artifacts.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.fusion_model import ConstrainedSectionFusion
from src.features.section_tfidf import FEATURE_NAMES as TFIDF_FEATURE_NAMES
from src.utils.metrics import evaluate_regression_metrics, project_disjoint_split


def main():
    pairs_path = ROOT_DIR / "data" / "processed" / "similarity_pairs_clean.csv"
    features_path = ROOT_DIR / "results" / "feature_label_analysis" / "section_wise_tfidf_features.csv"
    candidate_model_path = ROOT_DIR / "models" / "candidate_constrained_fusion_model.joblib"

    print("=== Training Candidate Constrained Fusion Model (Phase 2) ===")
    print(f"Loading pairs from: {pairs_path}")
    print(f"Loading features from: {features_path}")

    pairs_df = pd.read_csv(pairs_path)
    features_df = pd.read_csv(features_path)

    # Merge by pair_id to ensure clean pair alignment
    merged_df = pd.merge(
        pairs_df[["pair_id", "project_A_id", "project_B_id", "expert_similarity_label"]],
        features_df[["pair_id"] + TFIDF_FEATURE_NAMES],
        on="pair_id"
    )
    print(f"Total merged pairs: {len(merged_df)}")

    # Project-disjoint split (80/20)
    train_df, test_df = project_disjoint_split(merged_df, test_ratio=0.2, random_state=42)
    print(f"Train split pairs: {len(train_df)}")
    print(f"Test split pairs: {len(test_df)}")

    X_train = train_df[TFIDF_FEATURE_NAMES].values
    y_train = train_df["expert_similarity_label"].values
    X_test = test_df[TFIDF_FEATURE_NAMES].values
    y_test = test_df["expert_similarity_label"].values

    # Fit ConstrainedSectionFusion
    model = ConstrainedSectionFusion()
    model.fit(X_train, y_train, feature_names=TFIDF_FEATURE_NAMES)

    # Predictions
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    # Metrics
    train_metrics = evaluate_regression_metrics(y_train, preds_train)
    test_metrics = evaluate_regression_metrics(y_test, preds_test)

    print("\n--- Candidate Model Fitted Parameters ---")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Sum of coefficients: {np.sum(model.coef_):.4f}")
    print("Section Weights (sum to 4.0):")
    for feat, coef, w in zip(TFIDF_FEATURE_NAMES, model.coef_, model.weights):
        print(f"  {feat:25s}: {coef:8.4f} (normalized: {w*100:5.1f}%)")

    print("\n--- Performance Metrics (on Synthetic Benchmark Labels) ---")
    print(f"Train RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}, Pearson r: {train_metrics['pearson_r']:.4f}")
    print(f"Test  RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}, Pearson r: {test_metrics['pearson_r']:.4f}")

    # Save artifact
    model.save(candidate_model_path)
    print(f"\nSaved candidate artifact to: {candidate_model_path}")
    print("Verification: Production Ridge model was NOT modified.")

    # Save evaluation summary to results/
    summary_path = ROOT_DIR / "results" / "model_optimization" / "candidate_model_evaluation.json"
    summary_data = {
        "model_architecture": "Constrained Section Fusion (SLSQP)",
        "constraints": {
            "non_negative_weights": True,
            "weights_sum": 4.0,
            "intercept": 0.0,
            "bounds": [0.0, 4.0]
        },
        "learned_weights": {feat: float(coef) for feat, coef in zip(TFIDF_FEATURE_NAMES, model.coef_)},
        "normalized_weights_pct": {feat: float(w * 100) for feat, w in zip(TFIDF_FEATURE_NAMES, model.weights)},
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "sample_counts": {
            "total_pairs": len(merged_df),
            "train_pairs": len(train_df),
            "test_pairs": len(test_df)
        },
        "synthetic_benchmark_note": "Evaluated against synthetic benchmark labels; not real-world faculty validated."
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved evaluation metrics to: {summary_path}")


if __name__ == "__main__":
    main()
