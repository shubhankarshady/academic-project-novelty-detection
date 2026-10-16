
#!/usr/bin/env python
"""Error analysis for the saved hybrid novelty model."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_projects, load_similarity_pairs, load_feature_csv
from src.features.dense_embeddings import DENSE_FEATURE_NAMES
from src.features.section_tfidf import FEATURE_NAMES as TFIDF_FEATURE_NAMES
from src.models.hybrid_fusion import HYBRID_FEATURE_NAMES
from src.utils.metrics import project_disjoint_split


def main():
    data_path = (
        ROOT / "data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx"
    )
    tfidf_path = (
        ROOT / "results/feature_label_analysis/section_wise_tfidf_features.csv"
    )
    model_path = ROOT / "models/hybrid_fusion_model.joblib"
    output_dir = ROOT / "results/error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data.
    projects = load_projects(data_path)
    pairs = load_similarity_pairs(data_path)
    tfidf = load_feature_csv(tfidf_path)

    # Load the saved dense extractor and calculate pair features.
    dense_path = ROOT / "models/section_dense_extractor.joblib"
    dense_extractor = joblib.load(dense_path)
    dense_pairs = dense_extractor.extract_pair_features(pairs)

    # Validate pair identity before merging.
    keys = ["pair_id", "project_A_id", "project_B_id",
            "expert_similarity_label"]

    for name, df in [("TF-IDF", tfidf), ("Dense", dense_pairs)]:
        if df["pair_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate pair IDs.")

    for col in ["pair_id", "project_A_id", "project_B_id"]:
        tfidf[col] = tfidf[col].astype(str).str.strip()
        dense_pairs[col] = dense_pairs[col].astype(str).str.strip()

    # Confirm pair IDs and endpoints match.
    left = tfidf[keys].copy().sort_values("pair_id").reset_index(drop=True)
    right = dense_pairs[keys].copy().sort_values("pair_id").reset_index(drop=True)

    if not left["pair_id"].equals(right["pair_id"]):
        raise ValueError("TF-IDF and dense pair IDs differ.")

    for col in ["project_A_id", "project_B_id"]:
        if not left[col].equals(right[col]):
            raise ValueError(f"Pair endpoints differ: {col}")

    if not np.allclose(
        pd.to_numeric(left["expert_similarity_label"]).to_numpy(),
        pd.to_numeric(right["expert_similarity_label"]).to_numpy(),
    ):
        raise ValueError("Target labels differ between feature sources.")

    dense_cols = ["pair_id", *DENSE_FEATURE_NAMES]
    hybrid = tfidf.merge(
        dense_pairs[dense_cols],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )

    if hybrid[DENSE_FEATURE_NAMES].isna().any().any():
        raise ValueError("Missing dense features after merge.")

    # Use exactly the same deterministic project-disjoint split.
    train_df, test_df = project_disjoint_split(
        hybrid, test_ratio=0.2, random_state=42
    )

    if test_df.empty:
        raise ValueError("Test split is empty.")

    # Load model checkpoint.
    model = joblib.load(model_path)

    # Support checkpoints saved either as a model object or a dictionary.
    if isinstance(model, dict):
        model_type = model.get("model_type", "ridge")
        alpha = model.get("alpha", 1.0)
        estimator = model.get("model")

        if estimator is None:
            raise ValueError("Checkpoint dictionary has no 'model' entry.")

        predictions = estimator.predict(
            test_df[HYBRID_FEATURE_NAMES].to_numpy()
        )
    else:
        predictions = model.predict(
            test_df[HYBRID_FEATURE_NAMES].to_numpy()
        )

    actual = test_df["expert_similarity_label"].to_numpy(dtype=float)

    # Keep both raw predictions and clipped predictions for diagnosis.
    errors = test_df[
        [
            "pair_id", "project_A_id", "project_B_id",
            "expert_similarity_label", "similarity_category",
        ]
    ].copy()

    errors["actual_similarity"] = actual
    errors["predicted_raw"] = predictions
    errors["predicted_similarity"] = np.clip(predictions, 0, 4)
    errors["absolute_error"] = np.abs(
        errors["actual_similarity"] - errors["predicted_similarity"]
    )
    errors["squared_error"] = (
        errors["actual_similarity"] - errors["predicted_similarity"]
    ) ** 2

    errors = errors.sort_values("absolute_error", ascending=False)
    errors.to_csv(output_dir / "test_pair_errors.csv", index=False)

    # Error summary by expert-provided similarity category.
    category_summary = (
        errors.groupby("similarity_category", dropna=False)
        .agg(
            pair_count=("pair_id", "count"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda s: np.sqrt(s.mean())),
            mean_actual=("actual_similarity", "mean"),
            mean_predicted=("predicted_similarity", "mean"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    category_summary.to_csv(
        output_dir / "errors_by_category.csv", index=False
    )

    print("\nError analysis: held-out project-disjoint split")
    print(f"Test pairs: {len(errors)}")
    print(f"Train pairs: {len(train_df)}")

    print("\nErrors by similarity category:")
    print(category_summary.to_string(index=False))

    print("\n10 largest prediction errors:")
    print(
        errors[
            [
                "pair_id", "project_A_id", "project_B_id",
                "actual_similarity", "predicted_similarity",
                "absolute_error",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nSaved:")
    print("  results/error_analysis/test_pair_errors.csv")
    print("  results/error_analysis/errors_by_category.csv")


if __name__ == "__main__":
    main()
