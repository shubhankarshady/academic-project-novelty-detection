"""
Unit and Integration Tests for Candidate Constrained Section Fusion Model (Phase 2).

Tests:
CASE A - Exact Copy: X = [1, ..., 1] -> similarity = 4.0, novelty = 0.0%.
CASE B - Completely Disjoint: X = [0, ..., 0] -> similarity = 0.0, novelty = 100.0%.
CASE C - Monotonicity across all 8 sections: d(sim)/d(s_i) >= 0, d(novelty)/d(s_i) <= 0.
CASE D - Partial Similarity: Consistent, bounded scores across mixed vectors.
CASE E - Dataset-Only Similarity: High dataset similarity, other sections 0.0.
CASE F - Exact Match Ranking: Exact copy of reference project ranks #1 in retrieval.
CASE G - Duplicate Detection: Exact copy enters "Very Low Novelty / Potential Duplicate" tier (< 25%).
CASE H - Additive XAI Reconstruction: Intercept (0.0) + sum(contributions) == predicted similarity.
CASE I - Production Model Untouched: Verifies models/ridge_fusion_model.joblib remains unmodified.
"""

import unittest
from pathlib import Path
import numpy as np
import joblib

from src.models.fusion_model import ConstrainedSectionFusion
from src.features.section_tfidf import FEATURE_NAMES, SectionTFIDFExtractor
from src.explainability.explainer import NoveltyExplainer
from src.utils.metrics import calculate_novelty_score, get_novelty_level

ROOT_DIR = Path(__file__).resolve().parent.parent
CANDIDATE_MODEL_PATH = ROOT_DIR / "models" / "candidate_constrained_fusion_model.joblib"
PRODUCTION_RIDGE_PATH = ROOT_DIR / "models" / "ridge_fusion_model.joblib"
TFIDF_EXTRACTOR_PATH = ROOT_DIR / "models" / "section_tfidf_extractor.joblib"


class TestCandidateConstrainedFusionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load candidate model and TF-IDF extractor."""
        if not CANDIDATE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Candidate model artifact not found at {CANDIDATE_MODEL_PATH}. "
                "Run scripts/train_candidate_model.py first."
            )
        cls.candidate_model = ConstrainedSectionFusion.load(CANDIDATE_MODEL_PATH)
        cls.tfidf_extractor = joblib.load(TFIDF_EXTRACTOR_PATH)
        cls.explainer = NoveltyExplainer(
            tfidf_extractor=cls.tfidf_extractor,
            model=cls.candidate_model
        )

    def _predict_scalar(self, x):
        return float(np.asarray(self.candidate_model.predict(x)).ravel()[0])

    def test_case_a_exact_copy(self):
        """CASE A: All eight section similarities are 1.0 -> sim = 4.0, novelty = 0.0%."""
        x_ones = np.ones((1, 8), dtype=float)
        sim = self._predict_scalar(x_ones)
        novelty = float(calculate_novelty_score(sim))

        self.assertAlmostEqual(sim, 4.0, places=4, msg="Exact copy must predict similarity of 4.00")
        self.assertAlmostEqual(novelty, 0.0, places=4, msg="Exact copy must have 0.0% novelty")
        self.assertEqual(get_novelty_level(novelty), "Very Low Novelty / Potential Duplicate")

    def test_case_b_completely_disjoint(self):
        """CASE B: All eight section similarities are 0.0 -> sim = 0.0, novelty = 100.0%."""
        x_zeros = np.zeros((1, 8), dtype=float)
        sim = self._predict_scalar(x_zeros)
        novelty = float(calculate_novelty_score(sim))

        self.assertAlmostEqual(sim, 0.0, places=4, msg="Completely disjoint must predict similarity of 0.00")
        self.assertAlmostEqual(novelty, 100.0, places=4, msg="Completely disjoint must have 100.0% novelty")
        self.assertIn("High", get_novelty_level(novelty))

    def test_case_c_monotonicity_all_eight_sections(self):
        """
        CASE C: Increasing any section's similarity while holding others fixed
        must never decrease predicted similarity or increase novelty.
        Tested for all 8 sections individually.
        """
        rng = np.random.RandomState(42)
        base_vector = rng.uniform(0.1, 0.5, size=8)

        for sec_idx, sec_name in enumerate(FEATURE_NAMES):
            sim_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
            prev_sim = -1.0
            prev_novelty = 101.0

            for level in sim_levels:
                vec = base_vector.copy()
                vec[sec_idx] = level
                current_sim = self._predict_scalar(vec)
                current_novelty = float(calculate_novelty_score(current_sim))

                if prev_sim >= 0.0:
                    self.assertGreaterEqual(
                        current_sim,
                        prev_sim - 1e-7,
                        f"Non-monotonicity violation on section {sec_name}: similarity decreased from {prev_sim} to {current_sim}"
                    )
                    self.assertLessEqual(
                        current_novelty,
                        prev_novelty + 1e-7,
                        f"Non-monotonicity violation on section {sec_name}: novelty increased from {prev_novelty} to {current_novelty}"
                    )

                prev_sim = current_sim
                prev_novelty = current_novelty

    def test_case_d_partial_similarity_boundedness(self):
        """CASE D: Mixed similarity vectors produce consistent, strictly bounded scores in [0, 4] and [0, 100%]."""
        test_vectors = [
            np.array([0.5] * 8),
            np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]),
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            np.array([0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0.9, 0.1])
        ]

        for vec in test_vectors:
            sim = self._predict_scalar(vec)
            novelty = float(calculate_novelty_score(sim))

            self.assertGreaterEqual(sim, 0.0)
            self.assertLessEqual(sim, 4.0)
            self.assertGreaterEqual(novelty, 0.0)
            self.assertLessEqual(novelty, 100.0)

    def test_case_e_dataset_only_similarity(self):
        """
        CASE E: High dataset similarity (1.0) with all other 7 sections at 0.0.
        Verifies exact contribution and bounded novelty score.
        """
        vec = np.zeros(8, dtype=float)
        dataset_idx = FEATURE_NAMES.index("dataset_similarity")
        vec[dataset_idx] = 1.0

        sim = self._predict_scalar(vec)
        novelty = float(calculate_novelty_score(sim))

        expected_dataset_weight = float(self.candidate_model.coef_[dataset_idx])
        self.assertAlmostEqual(sim, expected_dataset_weight, places=4)
        # Weight is ~1.9529, so similarity ~1.9529 and novelty ~51.18%
        self.assertGreater(sim, 1.5)
        self.assertLess(sim, 2.5)
        self.assertEqual(get_novelty_level(novelty), "Moderate Novelty (Partial Overlap)")

    def test_case_f_exact_match_ranking(self):
        """
        CASE F: Exact copy of a reference proposal through scoring/retrieval
        must rank #1 in top matches (assuming self-exclusion is disabled for the test).
        """
        reference_proj = self.tfidf_extractor.projects_df.iloc[0].to_dict()
        report = self.explainer.explain_proposal(reference_proj, top_k=5, exclude_self=False)

        top_match = report["top_matches"][0]
        self.assertEqual(
            top_match["historical_project_id"],
            reference_proj["project_id"],
            f"Exact copy must rank #1. Instead ranked project: {top_match['historical_project_id']}"
        )
        self.assertAlmostEqual(top_match["predicted_similarity"], 4.0, places=3)
        self.assertAlmostEqual(top_match["novelty_score_pct"], 0.0, places=2)

    def test_case_g_duplicate_detection_tier(self):
        """
        CASE G: Exact copy reaches the configured duplicate threshold (< 25% novelty).
        Distinguishes numerical score (0.0%) from duplicate decision tier.
        """
        x_ones = np.ones((1, 8), dtype=float)
        sim = self._predict_scalar(x_ones)
        novelty_pct = float(calculate_novelty_score(sim))
        tier = get_novelty_level(novelty_pct)

        self.assertLess(novelty_pct, 25.0, "Exact copy must score strictly below 25.0% novelty")
        self.assertEqual(tier, "Very Low Novelty / Potential Duplicate")

    def test_case_h_additive_xai_reconstruction(self):
        """
        CASE H: Additive XAI Reconstruction.
        Verifies intercept (0.0) + sum(contributions) == predicted similarity.
        """
        test_proj = self.tfidf_extractor.projects_df.iloc[5].to_dict()
        report = self.explainer.explain_proposal(test_proj, top_k=3, exclude_self=True)

        for match in report["top_matches"]:
            intercept = match["intercept"]
            sum_contribs = match["sum_of_contributions"]
            raw_pred = match["raw_similarity_prediction"]
            pred_sim = match["predicted_similarity"]

            self.assertEqual(intercept, 0.0, "Candidate model must have intercept = 0.0")
            self.assertAlmostEqual(
                intercept + sum_contribs,
                raw_pred,
                places=4,
                msg="Additive reconstruction Intercept + sum(contribs) must match raw prediction"
            )
            self.assertAlmostEqual(
                np.clip(raw_pred, 0.0, 4.0),
                pred_sim,
                places=4,
                msg="Clipped prediction must match predicted similarity"
            )

    def test_case_i_production_ridge_model_unmodified(self):
        """CASE I: Verify production ridge_fusion_model.joblib exists and was NOT modified."""
        self.assertTrue(PRODUCTION_RIDGE_PATH.exists())
        prod_model = joblib.load(PRODUCTION_RIDGE_PATH)
        # Production model is a dict with Ridge instance
        self.assertIn("model", prod_model)
        self.assertAlmostEqual(float(prod_model["model"].intercept_), -0.0706317, places=4)
        # Verify production model still has negative weights (unmodified)
        coefs = prod_model["model"].coef_
        self.assertTrue(np.any(coefs < 0), "Production model must remain untouched with original negative weights")


if __name__ == "__main__":
    unittest.main()
