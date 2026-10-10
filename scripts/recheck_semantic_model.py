
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "Academic_Project_Similarity_Synthetic_Dataset.xlsx"
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
N_SPLITS = 5

SECTION_COLUMNS = {
    "title": "project_title",
    "abstract": "abstract",
    "problem": "problem_statement",
    "objectives": "objectives",
    "methodology": "proposed_methodology",
    "dataset": "dataset",
    "algorithm": "algorithm_model",
    "technology": "technology_tools",
}


def calculate_metrics(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred), 0.0, 4.0)
    result = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

    if len(y_true) > 1 and np.std(y_pred) > 0:
        result["Pearson_r"] = pearsonr(y_true, y_pred).statistic
        result["Spearman_rho"] = spearmanr(y_true, y_pred).statistic
    else:
        result["Pearson_r"] = np.nan
        result["Spearman_rho"] = np.nan

    return result


def main():
    print("1. Loading dataset...")
    projects = pd.read_excel(DATA_PATH, sheet_name="Projects")
    pairs = pd.read_excel(DATA_PATH, sheet_name="SimilarityPairs")

    projects["project_id"] = projects["project_id"].astype(str).str.strip()
    pairs["project_A_id"] = pairs["project_A_id"].astype(str).str.strip()
    pairs["project_B_id"] = pairs["project_B_id"].astype(str).str.strip()
    pairs["expert_similarity_label"] = pd.to_numeric(
        pairs["expert_similarity_label"], errors="coerce"
    )
    pairs = pairs.dropna(
        subset=[
            "project_A_id",
            "project_B_id",
            "expert_similarity_label",
        ]
    ).copy()

    if not pairs["expert_similarity_label"].between(0, 4).all():
        raise ValueError("Similarity labels must be between 0 and 4.")

    project_ids = projects["project_id"].to_numpy()
    project_index = {
        project_id: idx
        for idx, project_id in enumerate(project_ids)
    }

    print(f"   Projects: {len(projects)}")
    print(f"   Pairs: {len(pairs)}")

    print(f"\n2. Loading pretrained encoder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME)
    print("   Embedding dimensions:", encoder.get_sentence_embedding_dimension())

    print("\n3. Encoding each project section...")
    # The pretrained encoder is not fitted on our dataset or its labels.
    # Its fixed embeddings can be cached and reused in future experiments.
    section_embeddings = {}

    for section, column in SECTION_COLUMNS.items():
        texts = projects[column].fillna("").astype(str).tolist()

        embeddings = encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        section_embeddings[section] = embeddings

    del encoder

    print("\n4. Computing eight semantic pair-similarity features...")
    feature_names = [f"{section}_dense_similarity" for section in SECTION_COLUMNS]
    X = np.zeros((len(pairs), len(feature_names)), dtype=np.float32)

    for row_num, pair in enumerate(
        pairs[["project_A_id", "project_B_id"]].itertuples(
            index=False, name=None
        )
    ):
        id_a, id_b = pair

        if id_a not in project_index or id_b not in project_index:
            raise ValueError(f"Pair references an unknown project: {pair}")

        idx_a = project_index[id_a]
        idx_b = project_index[id_b]

        for col_num, section in enumerate(SECTION_COLUMNS):
            # Embeddings are L2-normalized, so dot product = cosine similarity.
            X[row_num, col_num] = np.dot(
                section_embeddings[section][idx_a],
                section_embeddings[section][idx_b],
            )

    y = pairs["expert_similarity_label"].to_numpy(dtype=float)

    feature_df = pd.DataFrame(X, columns=feature_names)
    feature_df["expert_similarity_label"] = y
    feature_df.to_csv(
        ROOT_DIR / "results" / "dense_semantic_features_recheck.csv",
        index=False,
    )

    print("\n5. Checking individual semantic features against labels...")
    correlations = []

    for col_num, feature_name in enumerate(feature_names):
        rho, p_value = spearmanr(X[:, col_num], y)
        correlations.append({
            "feature": feature_name,
            "spearman_rho": rho,
            "p_value": p_value,
        })

    corr_df = pd.DataFrame(correlations).sort_values(
        "spearman_rho", ascending=False
    )
    print(corr_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n6. Project-disjoint five-fold evaluation...")
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(project_ids), start=1
    ):
        train_ids = set(project_ids[train_idx])
        test_ids = set(project_ids[test_idx])

        train_mask = (
            pairs["project_A_id"].isin(train_ids)
            & pairs["project_B_id"].isin(train_ids)
        )
        test_mask = (
            pairs["project_A_id"].isin(test_ids)
            & pairs["project_B_id"].isin(test_ids)
        )

        train_rows = np.flatnonzero(train_mask.to_numpy())
        test_rows = np.flatnonzero(test_mask.to_numpy())

        if len(train_rows) == 0 or len(test_rows) == 0:
            raise ValueError(
                f"Fold {fold} has no training or test pairs. "
                "Inspect dataset pair coverage."
            )

        # Verify no project appears in both pair partitions.
        train_pair_ids = (
            set(pairs.iloc[train_rows]["project_A_id"])
            | set(pairs.iloc[train_rows]["project_B_id"])
        )
        test_pair_ids = (
            set(pairs.iloc[test_rows]["project_A_id"])
            | set(pairs.iloc[test_rows]["project_B_id"])
        )
        assert train_pair_ids.isdisjoint(test_pair_ids)

        X_train, X_test = X[train_rows], X[test_rows]
        y_train, y_test = y[train_rows], y[test_rows]

        models = {
            "Dense Ridge": Ridge(alpha=1.0),
            "Mean baseline": DummyRegressor(strategy="mean"),
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = calculate_metrics(y_test, predictions)

            fold_results.append({
                "fold": fold,
                "model": model_name,
                "train_pairs": len(train_rows),
                "test_pairs": len(test_rows),
                **metrics,
            })

        print(
            f"   Fold {fold}: train pairs={len(train_rows)}, "
            f"test pairs={len(test_rows)}"
        )

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(
        ROOT_DIR / "results" / "dense_semantic_cv_recheck.csv",
        index=False,
    )

    print("\n7. Fold-by-fold results...")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n8. Mean and standard deviation across folds...")
    summary = results_df.groupby("model")[
        ["MAE", "RMSE", "R2", "Pearson_r", "Spearman_rho"]
    ].agg(["mean", "std"])

    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nSaved files:")
    print("  results/dense_semantic_features_recheck.csv")
    print("  results/dense_semantic_cv_recheck.csv")
    print("\nEvaluation complete.")
    print(
        "Reminder: the target labels are synthetic. "
        "Results do not establish real-world originality."
    )


if __name__ == "__main__":
    main()
