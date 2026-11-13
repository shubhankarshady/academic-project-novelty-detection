"""
Test Suite for 120-Pair Faculty Manual Verification Dataset (Phase 5.2).

Validates:
1. Exact file generation and existence for both .xlsx and .csv outputs
2. Non-destruction / preservation of original 60-pair verification sample
3. Structure of two 60-pair evaluation sheets: Evaluation_Batch_1 and Evaluation_Batch_2
4. Total sample size of exactly 120 unique pairs with zero duplicates
5. Strict double-blind compliance: absence of synthetic labels, categories, and model predictions
6. Presence and completeness of all eight canonical candidate and reference project sections
7. Unannotated reviewer rating fields and 'PENDING_HUMAN_REVIEW' status initialization
8. Perfect stratification across the 5 ordinal similarity levels (0-4): 24 per level total, 12 per batch
9. Preservation of Development / Calibration (15 per batch) vs Held-Out Evaluation (45 per batch) subsets
10. Inclusion and balanced allocation of all 8 controlled benchmark cases
11. Professional Excel formatting: headers, freeze panes, auto-filters, column widths, conditional formatting
12. Deterministic reproducibility with random_state=42
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import openpyxl

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_faculty_sample import (
    build_faculty_evaluation_sample_120,
    EVALUATION_COLUMNS,
    MASTER_COLUMNS,
)


@pytest.fixture(scope="module")
def annotation_paths():
    base_dir = ROOT_DIR / "data" / "annotation"
    return {
        "xlsx_120": base_dir / "faculty_evaluation_sample_120.xlsx",
        "csv_120": base_dir / "faculty_evaluation_sample_120.csv",
        "master_120": base_dir / "faculty_evaluation_sample_120_master.csv",
        "xlsx_60": base_dir / "faculty_evaluation_sample_60.xlsx",
        "csv_60": base_dir / "faculty_evaluation_sample_60.csv",
    }


class TestFacultySample120Integrity:
    """Core integrity and structure tests for the 120-pair faculty verification dataset."""

    def test_sample_120_files_exist(self, annotation_paths):
        """Verify that all generated 120-pair files exist and are non-empty."""
        for key in ["xlsx_120", "csv_120", "master_120"]:
            p = annotation_paths[key]
            assert p.exists(), f"Missing file: {p}"
            assert p.stat().st_size > 0, f"File is empty: {p}"

    def test_original_60_pair_files_preserved(self, annotation_paths):
        """Ensure the original 60-pair sample files were NOT modified or overwritten."""
        p_csv = annotation_paths["csv_60"]
        p_xlsx = annotation_paths["xlsx_60"]

        assert p_csv.exists(), "Original faculty_evaluation_sample_60.csv must exist"
        assert p_xlsx.exists(), "Original faculty_evaluation_sample_60.xlsx must exist"

        df_60 = pd.read_csv(p_csv)
        assert len(df_60) == 60, f"Expected 60 pairs in original sample, found {len(df_60)}"

        wb_60 = openpyxl.load_workbook(p_xlsx, read_only=True)
        assert "Evaluation_Pairs" in wb_60.sheetnames
        assert "Rating_Rubric" in wb_60.sheetnames

    def test_excel_sheet_names_and_count(self, annotation_paths):
        """Verify that the 120-pair workbook contains the four required sheets."""
        wb = openpyxl.load_workbook(annotation_paths["xlsx_120"], read_only=True)
        expected_sheets = ["Evaluation_Batch_1", "Evaluation_Batch_2", "Rating_Rubric", "Instructions"]
        assert wb.sheetnames == expected_sheets

    def test_batch_sizes_in_excel(self, annotation_paths):
        """Verify that each evaluation sheet contains exactly 60 data rows (61 rows with header)."""
        wb = openpyxl.load_workbook(annotation_paths["xlsx_120"])
        for batch_name in ["Evaluation_Batch_1", "Evaluation_Batch_2"]:
            ws = wb[batch_name]
            assert ws.max_row == 61, f"{batch_name} has {ws.max_row} rows (expected 61: 1 header + 60 data)"
            assert ws.max_column == len(EVALUATION_COLUMNS), f"{batch_name} column count mismatch"

    def test_csv_batch_counts_and_structure(self, annotation_paths):
        """Verify the combined CSV has exactly 120 rows with 60 pairs in each batch."""
        df_csv = pd.read_csv(annotation_paths["csv_120"])
        assert len(df_csv) == 120, f"Expected 120 rows in CSV, found {len(df_csv)}"
        assert "batch" in df_csv.columns, "Missing 'batch' column in combined CSV"
        assert df_csv.columns[0] == "batch", "'batch' must be the first column in the CSV"

        batch_counts = df_csv["batch"].value_counts()
        assert batch_counts["Evaluation_Batch_1"] == 60
        assert batch_counts["Evaluation_Batch_2"] == 60

    def test_pair_ids_unique_and_monotonic(self, annotation_paths):
        """Verify all 120 pair IDs are globally unique, formatted EVAL-001 through EVAL-120."""
        df_csv = pd.read_csv(annotation_paths["csv_120"])
        pair_ids = list(df_csv["pair_id"])

        assert len(pair_ids) == 120
        assert len(set(pair_ids)) == 120, "Pair IDs must be globally unique across all 120 rows"
        expected_ids = [f"EVAL-{i+1:03d}" for i in range(120)]
        assert pair_ids == expected_ids, "Pair IDs must be sequential EVAL-001 to EVAL-120"

        # Check Batch 1 has EVAL-001 to EVAL-060
        b1_ids = list(df_csv[df_csv["batch"] == "Evaluation_Batch_1"]["pair_id"])
        assert b1_ids == [f"EVAL-{i+1:03d}" for i in range(60)]

        # Check Batch 2 has EVAL-061 to EVAL-120
        b2_ids = list(df_csv[df_csv["batch"] == "Evaluation_Batch_2"]["pair_id"])
        assert b2_ids == [f"EVAL-{i+61:03d}" for i in range(60)]

    def test_no_duplicate_pairs_across_or_within_batches(self, annotation_paths):
        """Verify that zero duplicate candidate-reference pairs exist across or within batches."""
        df_csv = pd.read_csv(annotation_paths["csv_120"])

        # Check directed pairs
        directed_pairs = list(zip(df_csv["project_candidate_id"], df_csv["project_reference_id"]))
        assert len(directed_pairs) == 120
        assert len(set(directed_pairs)) == 120, "Duplicate directed project pair detected!"

        # Check undirected pairs (A, B) vs (B, A)
        undirected_pairs = [
            (min(str(a), str(b)), max(str(a), str(b)))
            for a, b in directed_pairs
        ]
        assert len(set(undirected_pairs)) == 120, "Duplicate undirected project pair detected across sample!"

        # Explicitly check intersection between Batch 1 and Batch 2
        b1_df = df_csv[df_csv["batch"] == "Evaluation_Batch_1"]
        b2_df = df_csv[df_csv["batch"] == "Evaluation_Batch_2"]

        b1_keys = set((min(str(a), str(b)), max(str(a), str(b))) for a, b in zip(b1_df["project_candidate_id"], b1_df["project_reference_id"]))
        b2_keys = set((min(str(a), str(b)), max(str(a), str(b))) for a, b in zip(b2_df["project_candidate_id"], b2_df["project_reference_id"]))
        overlap = b1_keys.intersection(b2_keys)
        assert len(overlap) == 0, f"Found overlapping pairs between Batch 1 and Batch 2: {overlap}"


class TestDoubleBlindProtocolSafeguards:
    """Verifies that the faculty-facing workbook contains zero biasing information."""

    BIASING_TERMS = [
        "provisional_synthetic_label",
        "provisional_category",
        "selection_rationale",
        "model_similarity",
        "model_novelty",
        "predicted_score",
        "expected_rating",
        "ridge_score",
        "hybrid_score",
    ]

    def test_faculty_sheets_contain_no_biasing_columns(self, annotation_paths):
        """Verify faculty evaluation sheets do NOT contain synthetic labels, categories, or rationales."""
        wb = openpyxl.load_workbook(annotation_paths["xlsx_120"], read_only=True)
        for batch_name in ["Evaluation_Batch_1", "Evaluation_Batch_2"]:
            ws = wb[batch_name]
            first_row = [cell for cell in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
            for term in self.BIASING_TERMS:
                assert term not in first_row, f"Biasing column '{term}' found in faculty sheet {batch_name}!"

    def test_combined_csv_contains_no_biasing_columns(self, annotation_paths):
        """Verify the combined programmatic CSV is also blind to synthetic labels and model predictions."""
        df_csv = pd.read_csv(annotation_paths["csv_120"])
        for term in self.BIASING_TERMS:
            assert term not in df_csv.columns, f"Biasing column '{term}' found in combined CSV!"

    def test_reviewer_ratings_are_unannotated_and_empty(self, annotation_paths):
        """Verify that reviewer ratings and adjudication ratings are completely empty (null)."""
        df_csv = pd.read_csv(annotation_paths["csv_120"])
        for col in ["reviewer_1_rating", "reviewer_2_rating", "adjudicated_rating"]:
            assert col in df_csv.columns
            assert df_csv[col].isna().all(), f"Column '{col}' must be empty in all rows"

    def test_annotation_status_is_pending_across_all_rows(self, annotation_paths):
        """Verify that annotation_status is initialized to 'PENDING_HUMAN_REVIEW' for all 120 pairs."""
        df_csv = pd.read_csv(annotation_paths["csv_120"])
        assert (df_csv["annotation_status"] == "PENDING_HUMAN_REVIEW").all()


class TestContentCompletenessAndStratification:
    """Verifies completeness of project sections and balanced stratification across similarity levels."""

    def test_all_eight_sections_present_and_non_empty(self, annotation_paths):
        """Verify candidate and reference projects contain all 8 non-empty canonical sections."""
        df_csv = pd.read_csv(annotation_paths["csv_120"])
        sections = [
            "title",
            "abstract",
            "problem",
            "objectives",
            "methodology",
            "dataset",
            "algorithm",
            "technology",
        ]

        for sec in sections:
            cand_col = f"candidate_{sec}"
            ref_col = f"reference_{sec}"
            assert cand_col in df_csv.columns, f"Missing candidate column: {cand_col}"
            assert ref_col in df_csv.columns, f"Missing reference column: {ref_col}"

            assert not df_csv[cand_col].isna().any(), f"Null values found in {cand_col}"
            assert not df_csv[ref_col].isna().any(), f"Null values found in {ref_col}"

            # Check that string values have meaningful length (> 2 characters)
            assert (df_csv[cand_col].str.len() > 2).all(), f"Empty text detected in {cand_col}"
            assert (df_csv[ref_col].str.len() > 2).all(), f"Empty text detected in {ref_col}"

    def test_similarity_levels_stratified_in_master(self, annotation_paths):
        """Verify 24 pairs per similarity level across the 120 sample, and 12 per level in each batch."""
        master_df = pd.read_csv(annotation_paths["master_120"])
        assert len(master_df) == 120

        # Overall 24 per level
        level_counts = master_df["provisional_synthetic_label"].value_counts().to_dict()
        for level in range(5):
            assert level_counts.get(level, 0) == 24, f"Level {level} count was {level_counts.get(level, 0)}, expected 24"

        # 12 per level in each batch
        for batch_name in ["Evaluation_Batch_1", "Evaluation_Batch_2"]:
            b_df = master_df[master_df["batch"] == batch_name]
            b_counts = b_df["provisional_synthetic_label"].value_counts().to_dict()
            for level in range(5):
                assert b_counts.get(level, 0) == 12, f"{batch_name} Level {level} count was {b_counts.get(level, 0)}, expected 12"

    def test_development_and_held_out_subsets_preserved(self, annotation_paths):
        """Verify each batch preserves 15 Dev / Calibration and 45 Held-Out Evaluation pairs."""
        master_df = pd.read_csv(annotation_paths["master_120"])
        for batch_name in ["Evaluation_Batch_1", "Evaluation_Batch_2"]:
            b_df = master_df[master_df["batch"] == batch_name]
            subset_counts = b_df["subset"].value_counts().to_dict()
            assert subset_counts.get("DEV_CALIBRATION", 0) == 15, f"{batch_name} DEV count was {subset_counts.get('DEV_CALIBRATION')}, expected 15"
            assert subset_counts.get("HELD_OUT_EVALUATION", 0) == 45, f"{batch_name} EVAL count was {subset_counts.get('HELD_OUT_EVALUATION')}, expected 45"

    def test_all_eight_controlled_cases_included(self, annotation_paths):
        """Verify all 8 controlled test cases are represented across the 120-pair dataset."""
        master_df = pd.read_csv(annotation_paths["master_120"])
        cand_ids = set(master_df["project_candidate_id"])

        expected_cases = [
            "TEST-CASE-A",
            "TEST-CASE-B",
            "TEST-CASE-C",
            "TEST-CASE-D",
            "TEST-CASE-E",
            "TEST-CASE-F",
            "TEST-CASE-G",
            "TEST-CASE-H",
        ]

        for case_id in expected_cases:
            assert case_id in cand_ids, f"Controlled benchmark {case_id} missing from 120-pair dataset!"

        # Verify 4 controlled cases in Batch 1 and 4 in Batch 2
        b1_cands = set(master_df[master_df["batch"] == "Evaluation_Batch_1"]["project_candidate_id"])
        b2_cands = set(master_df[master_df["batch"] == "Evaluation_Batch_2"]["project_candidate_id"])

        b1_ctrl = [c for c in expected_cases if c in b1_cands]
        b2_ctrl = [c for c in expected_cases if c in b2_cands]

        assert len(b1_ctrl) == 4, f"Batch 1 should have 4 controlled cases, found {len(b1_ctrl)}: {b1_ctrl}"
        assert len(b2_ctrl) == 4, f"Batch 2 should have 4 controlled cases, found {len(b2_ctrl)}: {b2_ctrl}"


class TestExcelFormattingAndReproducibility:
    """Verifies professional Excel formatting rules and deterministic reproducibility."""

    def test_excel_formatting_features(self, annotation_paths):
        """Verify freeze panes, auto-filter, and conditional formatting rules."""
        wb = openpyxl.load_workbook(annotation_paths["xlsx_120"])

        for batch_name in ["Evaluation_Batch_1", "Evaluation_Batch_2"]:
            ws = wb[batch_name]
            # Freeze panes
            assert ws.freeze_panes == "A2", f"{batch_name} must freeze row 1"
            # Auto filter enabled
            assert ws.auto_filter.ref is not None, f"{batch_name} auto_filter must be set"
            assert ws.auto_filter.ref.startswith("A1:"), f"{batch_name} auto_filter range invalid"

            # Check that conditional formatting rules are registered
            cf_rules = list(ws.conditional_formatting)
            assert len(cf_rules) >= 4, f"{batch_name} must have conditional formatting rules"

    def test_reproducibility_with_seed(self):
        """Verify calling build_faculty_evaluation_sample_120 with seed 42 produces identical data."""
        b1_a, b2_a, master_a = build_faculty_evaluation_sample_120(random_state=42)
        b1_b, b2_b, master_b = build_faculty_evaluation_sample_120(random_state=42)

        pd.testing.assert_frame_equal(b1_a, b1_b)
        pd.testing.assert_frame_equal(b2_a, b2_b)
        pd.testing.assert_frame_equal(master_a, master_b)
