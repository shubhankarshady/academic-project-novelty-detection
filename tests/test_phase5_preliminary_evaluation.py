"""
Tests for Phase 5.3: Preliminary Human Evaluation and Model Comparison.
Validates rating integrity, evaluation outputs, metric calculations, and scientific constraints.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_phase5_preliminary import (
    calculate_evaluation_metrics,
    validate_human_ratings,
    SECTIONS
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / "validation"
FIGURES_DIR = RESULTS_DIR / "figures"
EXCEL_PATH = ROOT_DIR / "data" / "annotation" / "faculty_evaluation_sample_120.xlsx"
MASTER_PATH = ROOT_DIR / "data" / "annotation" / "faculty_evaluation_sample_120_master.csv"


class TestPhase5PreliminaryEvaluation:
    """Test suite for preliminary single-reviewer human evaluation pipeline."""

    def test_excel_reviewer_1_ratings_complete_and_valid(self):
        """Verify all 120 reviewer 1 ratings are integers in [0, 4]."""
        b1 = pd.read_excel(EXCEL_PATH, sheet_name="Evaluation_Batch_1")
        b2 = pd.read_excel(EXCEL_PATH, sheet_name="Evaluation_Batch_2")
        combined = pd.concat([b1, b2], ignore_index=True)

        assert len(combined) == 120
        assert combined["reviewer_1_rating"].isna().sum() == 0

        ratings = combined["reviewer_1_rating"]
        assert (ratings % 1 == 0).all()
        assert ((ratings >= 0) & (ratings <= 4)).all()

    def test_reviewer_2_and_adjudication_unfilled(self):
        """Verify reviewer 2 and adjudication columns remain pending."""
        b1 = pd.read_excel(EXCEL_PATH, sheet_name="Evaluation_Batch_1")
        b2 = pd.read_excel(EXCEL_PATH, sheet_name="Evaluation_Batch_2")
        combined = pd.concat([b1, b2], ignore_index=True)

        assert combined["reviewer_2_rating"].notna().sum() == 0
        assert combined["reviewer_2_comments"].notna().sum() == 0
        assert combined["adjudicated_rating"].notna().sum() == 0
        assert combined["adjudication_comments"].notna().sum() == 0

    def test_subset_preservation(self):
        """Verify 30 dev calibration and 90 held-out pairs."""
        master = pd.read_csv(MASTER_PATH)
        assert len(master) == 120
        counts = master["subset"].value_counts().to_dict()
        assert counts.get("DEV_CALIBRATION") == 30
        assert counts.get("HELD_OUT_EVALUATION") == 90

    def test_output_artifacts_exist_and_non_empty(self):
        """Verify all required CSV, JSON, and Markdown reports exist."""
        csv_eval = RESULTS_DIR / "phase5_preliminary_human_evaluation.csv"
        csv_comp = RESULTS_DIR / "phase5_preliminary_model_comparison.csv"
        json_sum = RESULTS_DIR / "phase5_preliminary_evaluation_summary.json"
        rep_md = RESULTS_DIR / "phase5_preliminary_evaluation_report.md"

        assert csv_eval.exists() and csv_eval.stat().st_size > 5000
        assert csv_comp.exists() and csv_comp.stat().st_size > 200
        assert json_sum.exists() and json_sum.stat().st_size > 1000
        assert rep_md.exists() and rep_md.stat().st_size > 2000

    def test_output_figures_exist_and_non_empty(self):
        """Verify all 6 publication-ready figures exist."""
        figure_names = [
            "figure1_human_rating_distribution.png",
            "figure2_model_vs_human_comparison.png",
            "figure3_confusion_matrices.png",
            "figure4_model_error_distribution.png",
            "figure5_actual_vs_predicted.png",
            "figure6_comparative_model_performance.png",
        ]
        for fname in figure_names:
            fpath = FIGURES_DIR / fname
            assert fpath.exists(), f"Missing figure: {fname}"
            assert fpath.stat().st_size > 10000, f"Figure {fname} is suspiciously small: {fpath.stat().st_size} bytes"

    def test_evaluation_csv_content_and_bounds(self):
        """Verify columns and continuous prediction bounds in evaluation CSV."""
        df = pd.read_csv(RESULTS_DIR / "phase5_preliminary_human_evaluation.csv")
        assert len(df) == 120

        required_cols = [
            "pair_id",
            "project_candidate_id",
            "project_reference_id",
            "reviewer_1_rating",
            "tfidf_predicted_similarity",
            "semantic_predicted_similarity",
            "hybrid_predicted_similarity",
            "subset",
            "warnings"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Boundedness checks
        for pred_col in ["tfidf_predicted_similarity", "semantic_predicted_similarity", "hybrid_predicted_similarity"]:
            vals = df[pred_col]
            assert vals.isna().sum() == 0
            assert (vals >= 0.0).all()
            assert (vals <= 4.0).all()

    def test_metric_calculation_accuracy(self):
        """Verify metric calculation formulas on synthetic ground truth."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_cont = np.array([0.2, 0.9, 2.1, 3.4, 3.8])
        y_disc = np.array([0, 1, 2, 3, 4])

        m = calculate_evaluation_metrics(y_true, y_cont, y_disc, "TestModel")

        assert m["Exact_Agreement_Pct"] == 100.0
        assert m["Within_1_Agreement_Pct"] == 100.0
        assert m["MAE"] == pytest.approx(0.20, abs=1e-4)
        assert m["Mean_Bias_Error"] == pytest.approx(0.08, abs=1e-4)
        assert m["Model_Reviewer_QWK"] == pytest.approx(1.0, abs=1e-3)

    def test_json_summary_structure_and_limitations(self):
        """Verify JSON summary status and explicit scientific limitations."""
        with open(RESULTS_DIR / "phase5_preliminary_evaluation_summary.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == "PRELIMINARY_SINGLE_REVIEWER_EVALUATION"
        assert data["sample_size"]["total_pairs"] == 120
        assert data["sample_size"]["held_out_evaluation_pairs"] == 90
        assert data["sample_size"]["dev_calibration_pairs"] == 30

        # Scientific limitations must be explicitly articulated
        limitations = data["scientific_limitations"]
        assert len(limitations) >= 5
        text_joined = " ".join(limitations).lower()
        assert "one human reviewer" in text_joined or "single reviewer" in text_joined
        assert "inter-reviewer" in text_joined
        assert "adjudicated" in text_joined

    def test_phase5_4_tuned_config_and_artifacts(self):
        """Verify Phase 5.4 tuned model config and outputs."""
        config_path = ROOT_DIR / "models" / "phase5_tuned_hybrid_config.json"
        assert config_path.exists()
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        assert cfg["tuned_hyperparameters"]["alpha"] == 0.60
        assert cfg["section_weights"]["weights_modified"] is False
        assert cfg["calibration_cohort"]["sample_size"] == 30

        tuning_res_path = RESULTS_DIR / "phase5_tuning_results.csv"
        comp_path = RESULTS_DIR / "phase5_tuned_model_comparison.csv"
        preds_path = RESULTS_DIR / "phase5_tuned_model_predictions.csv"
        rep_path = RESULTS_DIR / "phase5_tuning_report.md"

        assert tuning_res_path.exists()
        assert comp_path.exists()
        assert preds_path.exists()
        assert rep_path.exists()

        # Check tuning results have all 7 candidate alphas
        tdf = pd.read_csv(tuning_res_path)
        assert len(tdf) == 7
        assert set(np.round(tdf["alpha"], 1)) == {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

        # Check comparison has all 4 models on held-out and dev
        cdf = pd.read_csv(comp_path)
        assert len(cdf) == 8
        held_models = cdf[cdf["evaluation_split"] == "HELD_OUT_EVALUATION"]["model"].tolist()
        assert "Tuned Hybrid (alpha=0.60)" in held_models
        assert "Existing Hybrid (alpha=0.50)" in held_models
        assert "TF-IDF Ridge Model" in held_models
        assert "Dense Semantic Model" in held_models

        # Verify tuned hybrid accuracy improvement on held-out
        tuned_row = cdf[(cdf["evaluation_split"] == "HELD_OUT_EVALUATION") & (cdf["model"] == "Tuned Hybrid (alpha=0.60)")].iloc[0]
        assert tuned_row["Exact_Agreement_Pct"] > 80.0
        assert tuned_row["Within_1_Agreement_Pct"] > 95.0
        assert tuned_row["Macro_F1_Pct"] > 60.0
