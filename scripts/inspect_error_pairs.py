import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))



from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ERROR_PATH = ROOT / "results/error_analysis/test_pair_errors.csv"
TFIDF_PATH = (
    ROOT / "results/feature_label_analysis/section_wise_tfidf_features.csv"
)
DATA_PATH = ROOT / "data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx"
DENSE_PATH = ROOT / "models/section_dense_extractor.joblib"

from src.data.loader import load_similarity_pairs
import joblib

errors = pd.read_csv(ERROR_PATH)
tfidf = pd.read_csv(TFIDF_PATH)
pairs = load_similarity_pairs(DATA_PATH)
dense_extractor = joblib.load(DENSE_PATH)
dense = dense_extractor.extract_pair_features(pairs)

# Merge by pair ID, never by row position.
tfidf["pair_id"] = tfidf["pair_id"].astype(str).str.strip()
dense["pair_id"] = dense["pair_id"].astype(str).str.strip()

dense_columns = [
    "title_dense_similarity",
    "abstract_dense_similarity",
    "problem_dense_similarity",
    "objectives_dense_similarity",
    "methodology_dense_similarity",
    "dataset_dense_similarity",
    "algorithm_dense_similarity",
    "technology_dense_similarity",
]

section_columns = [
    "title_similarity",
    "abstract_similarity",
    "problem_similarity",
    "objectives_similarity",
    "methodology_similarity",
    "dataset_similarity",
    "algorithm_similarity",
    "technology_similarity",
]

features = tfidf.merge(
    dense[["pair_id", *dense_columns]],
    on="pair_id",
    how="inner",
    validate="one_to_one",
)

worst = errors.head(10)[["pair_id", "actual_similarity",
                         "predicted_similarity", "absolute_error"]]

result = worst.merge(
    features[["pair_id", *section_columns, *dense_columns]],
    on="pair_id",
    how="left",
    validate="one_to_one",
)

print(result.to_string(index=False))
result.to_csv(
    ROOT / "results/error_analysis/worst_pair_section_features.csv",
    index=False,
)
print("\nSaved worst-pair section features.")
