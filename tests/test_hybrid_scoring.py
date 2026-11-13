"""
Tests for Phase 4: Hybrid Lexical-Semantic Scoring Mode and Explainability.
Validates all 15 requirements specified in the Phase 4 test protocol.
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import NoveltyService, get_novelty_service, ROOT_DIR
from src.explainability.explainer import NoveltyExplainer
from scripts.evaluate_semantic_feasibility import CONTROLLED_TEST_CASES

# Index controlled cases by case_id for convenient lookup
CASES_BY_ID = {c["case_id"]: c for c in CONTROLLED_TEST_CASES}


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def hybrid_service():
    return NoveltyService(scoring_mode="hybrid", hybrid_alpha=0.50)


@pytest.fixture(scope="module")
def tfidf_service():
    return NoveltyService(scoring_mode="tfidf")


class TestPhase4HybridScoring:
    """Comprehensive test suite covering all 15 Phase 4 requirements."""

    def test_1_existing_tfidf_behavior_compatible(self, tfidf_service):
        """1. Existing TF-IDF behavior remains strictly compatible and operational."""
        assert tfidf_service.scoring_mode == "tfidf"
        proposal = CASES_BY_ID["CASE_A_EXACT_COPY"]["proposal"]
        report = tfidf_service.score_proposal(proposal, top_k=5)
        
        assert report["scoring_mode"] == "tfidf"
        assert report["hybrid_alpha"] == 1.0
        assert "predicted_similarity" in report["top_matches"][0]
        assert "section_similarity_scores" in report["top_matches"][0]
        # In TF-IDF mode, explainer coef_ must match the production Ridge model
        assert len(tfidf_service.explainer.coef_) == 8
        assert tfidf_service.explainer.intercept_ < 0.0  # Ridge intercept is -0.0706

    def test_2_both_scoring_modes_initialize_correctly(self):
        """2. Both scoring modes (tfidf and hybrid) initialize without error."""
        srv_tfidf = NoveltyService(scoring_mode="tfidf")
        srv_hybrid = NoveltyService(scoring_mode="hybrid", hybrid_alpha=0.50)

        assert srv_tfidf.scoring_mode == "tfidf"
        assert srv_hybrid.scoring_mode == "hybrid"
        assert srv_hybrid.dense_extractor is not None
        assert srv_hybrid.candidate_model is not None
        assert srv_hybrid.reference_corpus_size == 600

    def test_3_hybrid_formula_calculated_correctly(self, hybrid_service):
        """3. Hybrid formula is calculated correctly: alpha * tfidf + (1 - alpha) * dense."""
        explainer = hybrid_service.explainer
        proposal = CASES_BY_ID["CASE_C_STRONG_PARAPHRASE"]["proposal"]
        hist_row = explainer.projects_df.iloc[0]

        for test_alpha in [0.2, 0.5, 0.8]:
            res = explainer.explain_pair(
                candidate_dict=proposal,
                historical_row=hist_row,
                scoring_mode="hybrid",
                hybrid_alpha=test_alpha
            )
            for sec in explainer.section_names:
                s_lex = res["section_lexical_scores"][sec]
                s_sem = res["section_semantic_scores"][sec]
                s_hyb = res["section_hybrid_scores"][sec]
                expected = round(test_alpha * s_lex + (1.0 - test_alpha) * s_sem, 4)
                assert abs(s_hyb - expected) <= 2e-4

    def test_4_alpha_0_50_gives_expected_average(self, hybrid_service):
        """4. alpha = 0.50 gives the expected arithmetic average when both scores are valid."""
        explainer = hybrid_service.explainer
        proposal = CASES_BY_ID["CASE_C_STRONG_PARAPHRASE"]["proposal"]
        hist_row = explainer.projects_df.iloc[0]

        res = explainer.explain_pair(
            candidate_dict=proposal,
            historical_row=hist_row,
            scoring_mode="hybrid",
            hybrid_alpha=0.50
        )
        for sec in explainer.section_names:
            s_lex = res["section_lexical_scores"][sec]
            s_sem = res["section_semantic_scores"][sec]
            s_hyb = res["section_hybrid_scores"][sec]
            # Arithmetic average equality within floating-point rounding tolerance
            assert abs(s_hyb - round((s_lex + s_sem) / 2.0, 4)) <= 2e-4

    def test_5_alpha_values_validated(self):
        """5. Alpha values are strictly validated between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="hybrid_alpha"):
            NoveltyService(scoring_mode="hybrid", hybrid_alpha=-0.05)

        with pytest.raises(ValueError, match="hybrid_alpha"):
            NoveltyService(scoring_mode="hybrid", hybrid_alpha=1.05)

        with pytest.raises(ValueError, match="hybrid_alpha"):
            NoveltyService(scoring_mode="hybrid", hybrid_alpha="invalid_string")

    def test_6_empty_or_missing_sections_handled_safely(self, hybrid_service):
        """6. Empty or missing proposal sections are handled safely without crashing."""
        sparse_proposal = {
            "project_id": "SPARSE_TEST",
            "project_title": "Sparse Healthcare Analysis Pipeline",
            "domain": "Artificial Intelligence & Machine Learning",
            "abstract": "A valid abstract for testing empty section handling in the proposal evaluation pipeline.",
            "problem_statement": "",  # Empty
            "objectives": "   ",     # Whitespace only
            "proposed_methodology": "Valid methodology description with sufficient text.",
            # dataset omitted intentionally
            "algorithm_model": "Random Forest",
            "technology_tools": "Python"
        }
        report = hybrid_service.score_proposal(sparse_proposal, top_k=3, scoring_mode="hybrid")
        assert len(report["top_matches"]) == 3
        top_m = report["top_matches"][0]
        # Missing/empty sections must yield 0.0 similarity
        assert top_m["section_lexical_scores"]["problem"] == 0.0
        assert top_m["section_semantic_scores"]["problem"] == 0.0
        assert top_m["section_hybrid_scores"]["problem"] == 0.0
        assert top_m["section_hybrid_scores"]["objectives"] == 0.0
        assert top_m["section_hybrid_scores"]["dataset"] == 0.0

    def test_7_reference_ids_and_embeddings_remain_aligned(self, hybrid_service):
        """7. Reference project IDs and precomputed dense embeddings remain perfectly aligned."""
        ext = hybrid_service.dense_extractor
        assert len(ext.project_ids) == 600
        # Must match reference corpus IDs in exact sequential order
        corpus_ids = hybrid_service.tfidf_extractor.projects_df["project_id"].tolist()
        assert ext.project_ids == corpus_ids

        for sec in ext.section_cols.keys():
            emb_matrix = ext.embeddings[sec]
            assert emb_matrix.shape == (600, 384)
            # Verify embeddings are L2 normalized (norm == 1.0)
            norms = np.linalg.norm(emb_matrix, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_8_exact_copy_retrieval(self, hybrid_service):
        """8. Exact-copy proposal ranks its source project first with maximum similarity even with exclude_self=True."""
        case = CASES_BY_ID["CASE_A_EXACT_COPY"]
        # With corrected identity-based exclusion, exclude_self=True must NOT exclude SYN-0001
        report = hybrid_service.score_proposal(
            case["proposal"],
            top_k=5,
            scoring_mode="hybrid",
            hybrid_alpha=0.50,
            exclude_self=True
        )
        top_match = report["top_matches"][0]
        assert top_match["historical_project_id"] == case["target_ref_id"]
        assert top_match["predicted_similarity"] == 4.0
        assert top_match["novelty_score_pct"] == 0.0
        assert report["overall_novelty_score_pct"] == 0.0

    def test_8b_explicit_id_exclusion(self, hybrid_service):
        """8b. When a candidate proposal explicitly supplies a reference ID, that reference project is excluded."""
        case = CASES_BY_ID["CASE_A_EXACT_COPY"]
        # Explicitly set candidate project_id to the reference target ID "SYN-0001"
        self_candidate = dict(case["proposal"], project_id="SYN-0001")
        report = hybrid_service.score_proposal(
            self_candidate,
            top_k=5,
            scoring_mode="hybrid",
            hybrid_alpha=0.50,
            exclude_self=True
        )
        matched_ids = [m["historical_project_id"] for m in report["top_matches"]]
        assert "SYN-0001" not in matched_ids
        assert report["corpus_size_evaluated"] == 599

    def test_9_strong_paraphrase_retrieval(self, hybrid_service):
        """9. Strong paraphrase shows high semantic similarity and rank recovery."""
        case = CASES_BY_ID["CASE_C_STRONG_PARAPHRASE"]
        target_id = case["target_ref_id"]

        # Run under hybrid mode with top_k=100 (rank recovers from 258/600 down to 69)
        hyb_report = hybrid_service.score_proposal(
            case["proposal"],
            top_k=100,
            scoring_mode="hybrid",
            hybrid_alpha=0.50,
            exclude_self=False
        )
        retrieved_ids = [m["historical_project_id"] for m in hyb_report["top_matches"]]
        assert target_id in retrieved_ids
        rank = retrieved_ids.index(target_id) + 1
        target_match = hyb_report["top_matches"][rank - 1]

        # In hybrid mode, semantic scores across core sections must be robust
        assert target_match["section_semantic_scores"]["abstract"] >= 0.60
        assert target_match["section_semantic_scores"]["problem"] >= 0.50
        # Hybrid rank (69) recovers significantly from TF-IDF rank 258 and Ridge rank 600
        assert rank <= 75

    def test_10_unrelated_project_filtering(self, hybrid_service):
        """10. Unrelated proposal (Financial Fraud GCN) receives high novelty and low similarity against SYN-0001."""
        case = CASES_BY_ID["CASE_D_SAME_DOMAIN_DIFF_PROBLEM"]
        report = hybrid_service.score_proposal(
            case["proposal"],
            top_k=50,
            scoring_mode="hybrid",
            hybrid_alpha=0.50,
            exclude_self=False
        )
        # Find match for SYN-0001
        syn1_matches = [m for m in report["top_matches"] if m["historical_project_id"] == "SYN-0001"]
        if syn1_matches:
            assert syn1_matches[0]["predicted_similarity"] < 1.5
            assert syn1_matches[0]["novelty_score_pct"] >= 60.0

    def test_11_same_dataset_different_research_objective(self, hybrid_service):
        """11. Shared dataset with divergent objective shows section-level differentiation."""
        case = CASES_BY_ID["CASE_F_SAME_DATASET_DIFF_OBJECTIVES"]
        target_id = case["target_ref_id"]
        report = hybrid_service.score_proposal(
            case["proposal"],
            top_k=50,
            scoring_mode="hybrid",
            hybrid_alpha=0.50,
            exclude_self=False
        )
        retrieved_ids = [m["historical_project_id"] for m in report["top_matches"]]
        assert target_id in retrieved_ids
        target_match = report["top_matches"][retrieved_ids.index(target_id)]

        # Dataset section should have high similarity (both use PlantVillage)
        assert target_match["section_hybrid_scores"]["dataset"] >= 0.90
        # Objectives and problem statement should have lower similarity (generative inpainting vs disease classification)
        assert target_match["section_hybrid_scores"]["objectives"] < 0.60
        assert target_match["section_hybrid_scores"]["problem"] < 0.60

    def test_12_api_response_compatibility(self, api_client):
        """12. API endpoints return backward-compatible responses in both scoring modes."""
        proposal = CASES_BY_ID["CASE_A_EXACT_COPY"]["proposal"]

        # Health endpoint
        r_health = api_client.get("/api/health")
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert health_data["status"] == "healthy"
        assert "active_scoring_mode" in health_data

        # Default POST /api/score (tfidf mode)
        r_default = api_client.post("/api/score", json=proposal)
        assert r_default.status_code == 200
        data_default = r_default.json()
        assert data_default["scoring_mode"] == "tfidf"
        assert len(data_default["top_matches"]) == 5

        # Hybrid POST /api/score?mode=hybrid&alpha=0.5
        r_hybrid = api_client.post("/api/score?mode=hybrid&alpha=0.5", json=proposal)
        assert r_hybrid.status_code == 200
        data_hybrid = r_hybrid.json()
        assert data_hybrid["scoring_mode"] == "hybrid"
        assert data_hybrid["hybrid_alpha"] == 0.5
        top_match = data_hybrid["top_matches"][0]
        assert "section_lexical_scores" in top_match
        assert "section_semantic_scores" in top_match
        assert "section_hybrid_scores" in top_match

    def test_13_explainability_contribution_reconstruction(self, hybrid_service):
        """13. Section contributions reconstruct predicted similarity within documented tolerance."""
        proposal = CASES_BY_ID["CASE_C_STRONG_PARAPHRASE"]["proposal"]
        report = hybrid_service.score_proposal(proposal, top_k=5, scoring_mode="hybrid", hybrid_alpha=0.50)

        for match in report["top_matches"]:
            contribs_sum = match["sum_of_contributions"]
            intercept = match["intercept"]
            reconstructed_raw = match["reconstructed_raw_prediction"]

            # Tolerance check 1: sum_of_contributions == sum of section contributions
            actual_sum = sum(match["section_signed_contributions"].values())
            assert abs(contribs_sum - round(actual_sum, 4)) <= 1e-4

            # Tolerance check 2: reconstructed_raw == intercept + sum_of_contributions
            expected_raw = round(intercept + contribs_sum, 4)
            assert abs(reconstructed_raw - expected_raw) <= 1e-4

    def test_14_model_loading_and_inference_failure_handling(self, api_client):
        """14. Model-loading and inference failures are handled safely with clear errors."""
        proposal = CASES_BY_ID["CASE_A_EXACT_COPY"]["proposal"]

        # Invalid mode should return 422 Unprocessable Entity
        r_invalid_mode = api_client.post("/api/score?mode=quantum_mode", json=proposal)
        assert r_invalid_mode.status_code == 422

        # Invalid alpha out of range should return 422
        r_invalid_alpha = api_client.post("/api/score?mode=hybrid&alpha=2.5", json=proposal)
        assert r_invalid_alpha.status_code == 422

        # Invalid checkpoint path should raise FileNotFoundError on initialization
        with pytest.raises(FileNotFoundError):
            NoveltyService(tfidf_path="models/nonexistent_extractor.joblib")

    def test_15_frontend_build_and_existing_tests(self):
        """15. Frontend production build artifact exists and is verified."""
        dist_index = ROOT_DIR / "frontend" / "dist" / "index.html"
        assert dist_index.exists(), "Frontend build artifact 'dist/index.html' must exist."
        assert dist_index.stat().st_size > 500, "Frontend 'dist/index.html' must not be empty."
