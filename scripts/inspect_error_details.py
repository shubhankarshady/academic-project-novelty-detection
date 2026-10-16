
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.loader import load_projects, load_similarity_pairs

# Load original project records and pair labels
# Load original project records and pair labels
dataset_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Academic_Project_Similarity_Synthetic_Dataset.xlsx"
)

projects_df = load_projects(dataset_path)
pairs_df = load_similarity_pairs(dataset_path)
print("Projects loaded:", len(projects_df))
print("Similarity pairs loaded:", len(pairs_df))

errors_path = (
    PROJECT_ROOT / "results" / "error_analysis" / "test_pair_errors.csv"
)
errors_df = pd.read_csv(errors_path)

worst_ids = errors_df.nlargest(10, "absolute_error")["pair_id"]

project_columns = [
    "project_title",
    "abstract",
    "problem_statement",
    "objectives",
    "proposed_methodology",
    "dataset",
    "algorithm_model",
    "technology_tools",
]

for pair_id in worst_ids:
    pair = pairs_df.loc[pairs_df["pair_id"] == pair_id]

    if pair.empty:
        print(f"\nPair not found: {pair_id}")
        continue

    row = pair.iloc[0]
    id_a = row["project_A_id"]
    id_b = row["project_B_id"]

    print("\n" + "=" * 90)
    print(f"PAIR: {pair_id}")
    print(f"Project A: {id_a}")
    print(f"Project B: {id_b}")
    print(f"Expert label: {row['expert_similarity_label']}")
    print(f"Category: {row.get('similarity_category', 'N/A')}")
    print(f"Annotation note: {row.get('annotation_note', 'N/A')}")

    for project_id in [id_a, id_b]:
        project = projects_df.loc[
            projects_df["project_id"] == project_id
        ]

        if project.empty:
            print(f"\nProject record not found: {project_id}")
            continue

        project = project.iloc[0]
        print(f"\n--- {project_id} ---")

        for column in project_columns:
            if column in projects_df.columns:
                print(f"\n{column}:")
                print(project[column])
