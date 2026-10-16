
#!/usr/bin/env python
"""
Train section TF-IDF vectorizers, dense sentence transformers, and
hybrid section fusion models.

Saves checkpoints into models/ and logs project-disjoint metrics.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.loader import (
    load_projects,
    load_similarity_pairs,
    load_feature_csv,
)
from src.features.section_tfidf import (
    SectionTFIDFExtractor,
    FEATURE_NAMES as TFIDF_FEATURE_NAMES,
)
from src.features.dense_embeddings import (
    SectionDenseEmbeddingExtractor,
    DENSE_FEATURE_NAMES,
)
from src.models.hybrid_fusion import (
    HybridSectionFusionModel,
    HYBRID_FEATURE_NAMES,
)
from src.utils.metrics import (
    evaluate_regression_metrics,
    project_disjoint_split,
    project_disjoint_cv,
)


def align_and_merge_features(tfidf_df, dense_pairs_df):
    """
    Validate pair identity and merge dense features onto TF-IDF rows
    using pair_id rather than assuming both dataframes have the same order.
    """

    pair_keys = [
        "pair_id",
        "project_A_id",
        "project_B_id",
        "expert_similarity_label",
    ]

    # Validate required columns, unique IDs, and missing identifiers.
    for name, df in [
        ("TF-IDF", tfidf_df),
        ("Dense", dense_pairs_df),
    ]:
        missing = [col for col in pair_keys if col not in df.columns]

        if missing:
            raise ValueError(
                f"{name} dataframe is missing required columns: {missing}"
            )

        if df["pair_id"].duplicated().any():
            raise ValueError(
                f"{name} dataframe contains duplicate pair_id values."
            )

        if df[pair_keys].isna().any().any():
            raise ValueError(
                f"{name} dataframe has missing pair IDs, endpoints, or labels."
            )

    # Normalize identifiers before comparing.
    tfidf_keys = tfidf_df[pair_keys].copy()
    dense_keys = dense_pairs_df[pair_keys].copy()

    for col in ["pair_id", "project_A_id", "project_B_id"]:
        tfidf_keys[col] = tfidf_keys[col].astype(str).str.strip()
        dense_keys[col] = dense_keys[col].astype(str).str.strip()

    tfidf_keys["expert_similarity_label"] = pd.to_numeric(
        tfidf_keys["expert_similarity_label"], errors="raise"
    )
    dense_keys["expert_similarity_label"] = pd.to_numeric(
        dense_keys["expert_similarity_label"], errors="raise"
    )

    tfidf_keys = tfidf_keys.sort_values("pair_id").reset_index(drop=True)
    dense_keys = dense_keys.sort_values("pair_id").reset_index(drop=True)

    # Confirm both feature sources contain the same pairs.
    if not tfidf_keys["pair_id"].equals(dense_keys["pair_id"]):
        raise ValueError(
            "TF-IDF and dense dataframes contain different pair IDs."
        )

    # Confirm pair endpoints match.
    for col in ["project_A_id", "project_B_id"]:
        if not tfidf_keys[col].equals(dense_keys[col]):
            raise ValueError(
                f"TF-IDF and dense pair endpoints disagree in {col}."
            )

    # Confirm target labels match for each pair.
    labels_match = (
        tfidf_keys["expert_similarity_label"].to_numpy()
        == dense_keys["expert_similarity_label"].to_numpy()
    ).all()

    if not labels_match:
        raise ValueError(
            "TF-IDF and dense similarity labels do not match."
        )

    # Preserve TF-IDF row order and attach dense features by pair_id.
    hybrid_df = tfidf_df.copy()
    hybrid_df["pair_id"] = (
        hybrid_df["pair_id"].astype(str).str.strip()
    )

    dense_features = dense_pairs_df[
        ["pair_id", *DENSE_FEATURE_NAMES]
    ].copy()

    dense_features["pair_id"] = (
        dense_features["pair_id"].astype(str).str.strip()
    )

    hybrid_df = hybrid_df.merge(
        dense_features,
        on="pair_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    # Verify the merge did not lose any features or rows.
    if len(hybrid_df) != len(tfidf_df):
        raise ValueError("Row count changed unexpectedly after feature merge.")

    if hybrid_df[DENSE_FEATURE_NAMES].isna().any().any():
        raise ValueError(
            "Missing dense feature values detected after merging."
        )

    # Validate expected feature columns.
    missing_tfidf = [
        col for col in TFIDF_FEATURE_NAMES
        if col not in hybrid_df.columns
    ]
    missing_dense = [
        col for col in DENSE_FEATURE_NAMES
        if col not in hybrid_df.columns
    ]

    if missing_tfidf or missing_dense:
        raise ValueError(
            f"Missing TF-IDF features: {missing_tfidf}; "
            f"missing dense features: {missing_dense}"
        )

    if len(HYBRID_FEATURE_NAMES) != (
        len(TFIDF_FEATURE_NAMES) + len(DENSE_FEATURE_NAMES)
    ):
        raise ValueError(
            "Unexpected hybrid feature configuration: expected "
            "TF-IDF and dense feature names to form the hybrid set."
        )

    print(
        f"   Verified {len(hybrid_df)} pair-aligned hybrid rows."
    )

    return hybrid_df


def main():
    data_path = (
        ROOT_DIR
        / "data"
        / "raw"
        / "Academic_Project_Similarity_Synthetic_Dataset.xlsx"
    )

    feature_path = (
        ROOT_DIR
        / "results"
        / "feature_label_analysis"
        / "section_wise_tfidf_features.csv"
    )

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load projects, similarity pairs, and TF-IDF features
    # ---------------------------------------------------------
    print("1. Loading projects and similarity pairs...")

    projects_df = load_projects(data_path)
    pairs_df = load_similarity_pairs(data_path)
    tfidf_df = load_feature_csv(feature_path)

    print(f"   Projects: {len(projects_df)}")
    print(f"   Similarity pairs: {len(pairs_df)}")
    print(f"   TF-IDF feature rows: {len(tfidf_df)}")

    # ---------------------------------------------------------
    # 2. Fit and save the TF-IDF extractor
    # ---------------------------------------------------------
    print("\n2. Fitting SectionTFIDFExtractor...")

    tfidf_extractor = SectionTFIDFExtractor()
    tfidf_extractor.fit_transform_projects(projects_df)

    tfidf_path = models_dir / "section_tfidf_extractor.joblib"
    joblib.dump(tfidf_extractor, tfidf_path)

    print(f"   Saved TF-IDF extractor: {tfidf_path}")

    # ---------------------------------------------------------
    # 3. Fit and save the dense embedding extractor
    # ---------------------------------------------------------
    print("\n3. Fitting SectionDenseEmbeddingExtractor...")

    dense_extractor = SectionDenseEmbeddingExtractor(
        model_name="all-MiniLM-L6-v2"
    )

    dense_extractor.fit_transform_projects(projects_df)
    dense_pairs_df = dense_extractor.extract_pair_features(pairs_df)

    dense_path = models_dir / "section_dense_extractor.joblib"
    joblib.dump(dense_extractor, dense_path)

    print(f"   Saved dense extractor: {dense_path}")

    # ---------------------------------------------------------
    # 4. Align and merge feature sets by pair_id
    # ---------------------------------------------------------
    print("\n4. Building aligned hybrid feature matrix...")

    hybrid_df = align_and_merge_features(
        tfidf_df,
        dense_pairs_df,
    )

    print(
        f"   TF-IDF features: {len(TFIDF_FEATURE_NAMES)}"
    )
    print(
        f"   Dense features: {len(DENSE_FEATURE_NAMES)}"
    )
    print(
        f"   Hybrid features: {len(HYBRID_FEATURE_NAMES)}"
    )

    # ---------------------------------------------------------
    # 5. Project-disjoint cross-validation
    # ---------------------------------------------------------
    print("\n5. Evaluating project-disjoint 5-fold cross-validation...")

    cv_tfidf = project_disjoint_cv(
        hybrid_df,
        TFIDF_FEATURE_NAMES,
        n_splits=5,
    )

    cv_dense = project_disjoint_cv(
        hybrid_df,
        DENSE_FEATURE_NAMES,
        n_splits=5,
    )

    cv_hybrid = project_disjoint_cv(
        hybrid_df,
        HYBRID_FEATURE_NAMES,
        n_splits=5,
    )

    print(
        f"   TF-IDF only -> RMSE: "
        f"{cv_tfidf['mean_rmse']:.4f} +/- "
        f"{cv_tfidf['std_rmse']:.4f} | "
        f"R2: {cv_tfidf['mean_r2']:.4f}"
    )

    print(
        f"   Dense only  -> RMSE: "
        f"{cv_dense['mean_rmse']:.4f} +/- "
        f"{cv_dense['std_rmse']:.4f} | "
        f"R2: {cv_dense['mean_r2']:.4f}"
    )

    print(
        f"   Hybrid      -> RMSE: "
        f"{cv_hybrid['mean_rmse']:.4f} +/- "
        f"{cv_hybrid['std_rmse']:.4f} | "
        f"R2: {cv_hybrid['mean_r2']:.4f}"
    )

    # ---------------------------------------------------------
    # 6. Train final hybrid model on project-disjoint split
    # ---------------------------------------------------------
    print("\n6. Training final hybrid model...")

    train_df, test_df = project_disjoint_split(
        hybrid_df,
        test_ratio=0.2,
        random_state=42,
    )

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Project-disjoint split produced an empty training or test set."
        )

    # Confirm that training and test projects are disjoint.
    train_projects = (
        set(train_df["project_A_id"])
        | set(train_df["project_B_id"])
    )

    test_projects = (
        set(test_df["project_A_id"])
        | set(test_df["project_B_id"])
    )

    if not train_projects.isdisjoint(test_projects):
        raise ValueError(
            "Project leakage detected between training and test sets."
        )

    print(f"   Training pairs: {len(train_df)}")
    print(f"   Test pairs: {len(test_df)}")
    print(f"   Training projects: {len(train_projects)}")
    print(f"   Test projects: {len(test_projects)}")

    hybrid_model = HybridSectionFusionModel(
        model_type="ridge"
    )

    hybrid_model.fit(
        train_df[HYBRID_FEATURE_NAMES].values,
        train_df["expert_similarity_label"].values,
    )

    model_path = models_dir / "hybrid_fusion_model.joblib"
    hybrid_model.save(model_path)

    print(f"   Saved hybrid model: {model_path}")

    # ---------------------------------------------------------
    # 7. Evaluate on held-out projects
    # ---------------------------------------------------------
    print("\n7. Evaluating on held-out projects...")

    predictions = hybrid_model.predict(
        test_df[HYBRID_FEATURE_NAMES].values
    )

    metrics = evaluate_regression_metrics(
        test_df["expert_similarity_label"].values,
        predictions,
    )

    print("\nUnseen Project-Disjoint Test Metrics:")

    for metric_name, value in metrics.items():
        print(f"   {metric_name:15s}: {value:.4f}")

    print("\nPipeline completed successfully.")
    print(
        "Reminder: the dataset labels are synthetic; "
        "these metrics do not establish real-world originality."
    )


if __name__ == "__main__":
    main()
