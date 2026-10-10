
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = (
    ROOT / "data" / "raw"
    / "Academic_Project_Similarity_Synthetic_Dataset.xlsx"
)
RESULTS_DIR = ROOT / "results" / "three_model_comparison"

SEED = 42
N_SPLITS = 5
RIDGE_ALPHA = 1.0
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SECTIONS = {
    "title": "project_title",
    "abstract": "abstract",
    "problem": "problem_statement",
    "objectives": "objectives",
    "methodology": "proposed_methodology",
    "dataset": "dataset",
    "algorithm": "algorithm_model",
    "technology": "technology_tools",
}


def metrics(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred), 0.0, 4.0)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Spearman": spearmanr(y_true, y_pred).statistic,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
            "project_A_id", "project_B_id", "expert_similarity_label"
        ]
    ).copy()

    if not pairs["expert_similarity_label"].between(0, 4).all():
        raise ValueError("Labels must be in the range 0 to 4.")

    if projects["project_id"].duplicated().any():
        raise ValueError("Duplicate project IDs found in Projects sheet.")

    ids = projects["project_id"].to_numpy()
    project_index = {pid: i for i, pid in enumerate(ids)}

    unknown_ids = (
        (set(pairs["project_A_id"]) | set(pairs["project_B_id"]))
        - set(ids)
    )
    if unknown_ids:
        raise ValueError(f"Pairs contain unknown project IDs: {list(unknown_ids)[:5]}")

    print(f"   Projects: {len(projects)} | Pairs: {len(pairs)}")

    print("\n2. Loading pretrained semantic encoder...")
    encoder = SentenceTransformer(MODEL_NAME)
    dense_vectors = {}

    # Encode once: the pretrained encoder is fixed and is not trained
    # using this dataset's labels. The resulting vectors are reused per fold.
    for section, column in SECTIONS.items():
        texts = projects[column].fillna("").astype(str).tolist()
        dense_vectors[section] = encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    del encoder

    def pair_features(pair_frame, feature_type, train_ids=None):
        """Compute pair similarities; TF-IDF is fitted only on train projects."""
        result = np.zeros((len(pair_frame), len(SECTIONS)), dtype=np.float32)

        if feature_type in ("tfidf", "hybrid"):
            if train_ids is None:
                raise ValueError("Training project IDs are required for TF-IDF.")

            train_rows = projects[
                projects["project_id"].isin(train_ids)
            ]
            if train_rows.empty:
                raise ValueError("No training projects available for TF-IDF.")

            vectorizers = {}
            tfidf_matrices = {}

            for section, column in SECTIONS.items():
                train_texts = (
                    train_rows[column].fillna("").astype(str).tolist()
                )
                all_texts = (
                    projects[column].fillna("").astype(str).tolist()
                )

                vectorizer = TfidfVectorizer(
                    stop_words="english", ngram_range=(1, 2)
                )

                try:
                    vectorizer.fit(train_texts)
                    tfidf_matrices[section] = vectorizer.transform(all_texts)
                    vectorizers[section] = vectorizer
                except ValueError as exc:
                    if "empty vocabulary" not in str(exc).lower():
                        raise
                    tfidf_matrices[section] = None
                    vectorizers[section] = None

        for row_idx, (id_a, id_b) in enumerate(
            pair_frame[
                ["project_A_id", "project_B_id"]
            ].itertuples(index=False, name=None)
        ):
            ia, ib = project_index[id_a], project_index[id_b]

            for col_idx, section in enumerate(SECTIONS):
                values = []

                if feature_type in ("tfidf", "hybrid"):
                    matrix = tfidf_matrices[section]
                    if matrix is None:
                        lexical_sim = 0.0
                    else:
                        # TF-IDF rows are L2-normalized; dot product is cosine.
                        lexical_sim = float(
                            matrix[ia].multiply(matrix[ib]).sum()
                        )
                    values.append(lexical_sim)

                if feature_type in ("dense", "hybrid"):
                    dense_sim = float(
                        np.dot(
                            dense_vectors[section][ia],
                            dense_vectors[section][ib],
                        )
                    )
                    values.append(dense_sim)

                if feature_type == "tfidf":
                    result[row_idx, col_idx] = values[0]
                elif feature_type == "dense":
                    result[row_idx, col_idx] = values[0]
                else:
                    # Hybrid column order: first 8 lexical, then 8 dense.
                    result[row_idx, col_idx] = values[0]
                    # Dense features are placed in a second matrix below.

        if feature_type != "hybrid":
            return result

        # Recompute dense features separately to keep the 16-feature order clear.
        dense_result = np.zeros_like(result)
        for row_idx, (id_a, id_b) in enumerate(
            pair_frame[
                ["project_A_id", "project_B_id"]
            ].itertuples(index=False, name=None)
        ):
            ia, ib = project_index[id_a], project_index[id_b]
            for col_idx, section in enumerate(SECTIONS):
                dense_result[row_idx, col_idx] = np.dot(
                    dense_vectors[section][ia],
                    dense_vectors[section][ib],
                )

        return np.concatenate([result, dense_result], axis=1)

    print("\n3. Running identical project-disjoint folds...")
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    all_results = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(ids), start=1):
        train_ids = set(ids[train_idx])
        test_ids = set(ids[test_idx])

        train_mask = (
            pairs["project_A_id"].isin(train_ids)
            & pairs["project_B_id"].isin(train_ids)
        )
        test_mask = (
            pairs["project_A_id"].isin(test_ids)
            & pairs["project_B_id"].isin(test_ids)
        )

        train_pairs = pairs.loc[train_mask].copy()
        test_pairs = pairs.loc[test_mask].copy()

        if train_pairs.empty or test_pairs.empty:
            raise ValueError(f"Fold {fold} has empty train/test pairs.")

        train_pair_projects = (
            set(train_pairs["project_A_id"])
            | set(train_pairs["project_B_id"])
        )
        test_pair_projects = (
            set(test_pairs["project_A_id"])
            | set(test_pairs["project_B_id"])
        )
        assert train_pair_projects.isdisjoint(test_pair_projects)

        y_train = train_pairs["expert_similarity_label"].to_numpy(float)
        y_test = test_pairs["expert_similarity_label"].to_numpy(float)

        feature_sets = {
            "TF-IDF only": (
                pair_features(train_pairs, "tfidf", train_ids),
                pair_features(test_pairs, "tfidf", train_ids),
            ),
            "Dense only": (
                pair_features(train_pairs, "dense", train_ids),
                pair_features(test_pairs, "dense", train_ids),
            ),
            "Hybrid": (
                pair_features(train_pairs, "hybrid", train_ids),
                pair_features(test_pairs, "hybrid", train_ids),
            ),
        }

        for model_name, (X_train, X_test) in feature_sets.items():
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            all_results.append({
                "fold": fold,
                "model": model_name,
                "train_pairs": len(train_pairs),
                "test_pairs": len(test_pairs),
                **metrics(y_test, predictions),
            })

        print(
            f"   Fold {fold}: {len(train_pairs)} training pairs, "
            f"{len(test_pairs)} test pairs"
        )

    results = pd.DataFrame(all_results)
    results.to_csv(RESULTS_DIR / "fold_results.csv", index=False)

    summary = results.groupby("model")[
        ["MAE", "RMSE", "R2", "Spearman"]
    ].agg(["mean", "std"])
    summary.to_csv(RESULTS_DIR / "summary.csv")

    print("\n4. Fold-by-fold results")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n5. Mean ± standard deviation across folds")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    print(f"\nSaved results to: {RESULTS_DIR}")
    print(
        "All scores predict synthetic labels. "
        "They do not establish real-world project originality."
    )


if __name__ == "__main__":
    main()
