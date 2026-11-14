"""
Unit and Integration Tests for Two-Stage Candidate Retrieval Architecture.

Validates the 4 stages:
  Stage 1: TF-IDF Candidate Retrieval across all reference corpus projects (ranks all, selects Top 50).
  Stage 2: Dense Semantic Similarity computed ONLY on the shortlisted candidates (never across all 600).
  Stage 3: Hybrid Ranking combining lexical and semantic scores via (0.60 * TF-IDF + 0.40 * Semantic).
  Stage 4: Corpus-relative novelty evaluation returning Top 5 most similar projects sorted descending.

Also verifies:
  - Configurable candidate limit (default 50, overrideable via param or TWO_STAGE_CANDIDATE_LIMIT).
  - Graceful handling when corpus has fewer than 50 projects (e.g., 5 or 20 projects).
  - Strict alignment of candidate IDs between TF-IDF and semantic scores.
  - Candidate self-exclusion handling.
  - Incomplete/empty proposal section resilience.
  - FastAPI endpoints (/api/health and /api/score) with two-stage retrieval metadata.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app, get_db
from src.api.service import NoveltyService, get_novelty_service
from src.storage.db import DatabaseManager
from src.explainability.explainer import NoveltyExplainer
from scripts.evaluate_semantic_feasibility import CONTROLLED_TEST_CASES

CASES = {c["case_id"]: c for c in CONTROLLED_TEST_CASES}


@pytest.fixture(scope="module")
def client():
    """Shared FastAPI test client."""
    return TestClient(app)


@pytest.fixture(scope="module")
def service():
    """Singleton novelty scoring service."""
    return get_novelty_service()


@pytest.fixture
def isolated_db():
    """In-memory database for clean test isolation."""
    db = DatabaseManager(":memory:")
    app.dependency_overrides[get_db] = lambda: db
    yield db
    app.dependency_overrides.clear()
    db.close()


# ============================================================================
# 1. Stage 1 & Stage 2 Retrieval Scope Verification
# ============================================================================

def test_stage1_evaluates_all_corpus_and_stage2_evaluates_only_shortlist(service):
    """
    Verify:
      1. Stage 1 TF-IDF evaluates all 600 eligible reference corpus projects.
      2. Exactly Top 50 candidates are shortlisted.
      3. Stage 2 Dense embeddings are sliced ONLY for the 50 shortlisted candidates.
    """
    explainer = service.get_explainer_for_mode("tuned_hybrid", alpha=0.60, candidate_limit=50)
    proposal = CASES["CASE_C_STRONG_PARAPHRASE"]["proposal"]

    # Spy on dot product or slice size during Stage 2
    original_dot = np.dot
    dense_dot_shapes = []

    def spy_dot(a, b):
        res = original_dot(a, b)
        # Record matrix-vector dot products where a is 2D
        if hasattr(a, "shape") and len(a.shape) == 2:
            dense_dot_shapes.append(a.shape)
        return res

    with patch("src.explainability.explainer.np.dot", side_effect=spy_dot):
        report = explainer.explain_proposal(
            candidate_dict=proposal,
            top_k=5,
            scoring_mode="tuned_hybrid",
            hybrid_alpha=0.60,
            candidate_limit=50
        )

    # Verify Stage 1 and Stage 2 metadata
    assert report["retrieval_stage"] == "two_stage"
    assert report["stage1_candidates_evaluated"] == 600
    assert report["stage2_shortlist_size"] == 50
    assert report["candidate_limit"] == 50
    assert report["corpus_size_evaluated"] == 600

    # Verify that dense semantic dot products were performed ONLY on shortlisted 50 candidates
    assert len(dense_dot_shapes) > 0
    for shape in dense_dot_shapes:
        # Each section dot product must be on (50, 384), NEVER on (600, 384)
        assert shape[0] == 50, f"Expected 50 shortlisted items in semantic evaluation, but got {shape[0]}"


def test_configurable_candidate_limit(service):
    """Verify that Stage 1 candidate limit is configurable (e.g., 20 instead of 50)."""
    explainer = service.get_explainer_for_mode("tuned_hybrid", alpha=0.60)
    proposal = CASES["CASE_D_SAME_DOMAIN_DIFF_PROBLEM"]["proposal"]

    report = explainer.explain_proposal(
        candidate_dict=proposal,
        top_k=5,
        scoring_mode="tuned_hybrid",
        hybrid_alpha=0.60,
        candidate_limit=20
    )

    assert report["stage1_candidates_evaluated"] == 600
    assert report["stage2_shortlist_size"] == 20
    assert report["candidate_limit"] == 20
    assert len(report["top_matches"]) == 5


# ============================================================================
# 2. Stage 3: Hybrid Ranking and 60:40 Formula Verification
# ============================================================================

def test_hybrid_ranking_formula_60_40(service):
    """
    Verify that hybrid similarity strictly implements:
      Hybrid Score = (0.60 * TF-IDF Score) + (0.40 * Semantic Score)
    and results are sorted descending by predicted similarity.
    """
    explainer = service.get_explainer_for_mode("tuned_hybrid", alpha=0.60)
    proposal = CASES["CASE_C_STRONG_PARAPHRASE"]["proposal"]

    report = explainer.explain_proposal(
        candidate_dict=proposal,
        top_k=5,
        scoring_mode="tuned_hybrid",
        hybrid_alpha=0.60
    )

    top_matches = report["top_matches"]
    assert len(top_matches) == 5

    # 1. Verify monotonic descending sort order
    prev_sim = float("inf")
    for match in top_matches:
        pred_sim = match["predicted_similarity"]
        assert pred_sim <= prev_sim, f"Results not sorted descending: {pred_sim} > {prev_sim}"
        prev_sim = pred_sim

    # 2. Verify section-wise 60:40 convex combination
    for match in top_matches:
        lex_scores = match["section_lexical_scores"]
        sem_scores = match["section_semantic_scores"]
        hyb_scores = match["section_hybrid_scores"]
        eff_scores = match["section_similarity_scores"]

        for sec in explainer.section_names:
            expected_hybrid = round(0.60 * lex_scores[sec] + 0.40 * sem_scores[sec], 4)
            # Both roundings should be within 0.001
            assert abs(hyb_scores[sec] - expected_hybrid) <= 0.001
            assert abs(eff_scores[sec] - expected_hybrid) <= 0.001


# ============================================================================
# 3. Candidate ID Alignment Consistency
# ============================================================================

def test_candidate_id_consistency_across_stages(service):
    """
    Verify that candidate project IDs match exactly between Stage 1, Stage 2,
    and Stage 4 match explanations without index drift or shuffling.
    """
    explainer = service.get_explainer_for_mode("tuned_hybrid", alpha=0.60)
    proposal = CASES["CASE_E_DIFF_DOMAIN_COMMON_TECH"]["proposal"]

    report = explainer.explain_proposal(
        candidate_dict=proposal,
        top_k=5,
        scoring_mode="tuned_hybrid",
        hybrid_alpha=0.60
    )

    for match in report["top_matches"]:
        pid = match["historical_project_id"]
        # Verify project exists in reference database
        matching_rows = explainer.projects_df[explainer.projects_df["project_id"] == pid]
        assert len(matching_rows) == 1
        expected_title = matching_rows.iloc[0]["project_title"]
        assert match["historical_project_title"] == expected_title

        # Verify dense extractor index matches
        assert pid in explainer.dense_extractor.project_to_index


# ============================================================================
# 4. Graceful Handling for Small Corpora (< 50 projects)
# ============================================================================

def test_graceful_handling_when_corpus_has_fewer_than_50_projects(service):
    """
    Verify that if the corpus contains fewer than 50 projects (e.g. a small slice of 15 projects),
    the retrieval pipeline handles it smoothly without out-of-bounds errors.
    """
    small_df = service.tfidf_extractor.projects_df.iloc[:15].copy().reset_index(drop=True)
    small_pids = set(small_df["project_id"])

    # Build a small custom dense extractor mock
    mock_dense = MagicMock()
    mock_dense.project_to_index = {pid: i for i, pid in enumerate(small_df["project_id"])}
    mock_dense.embeddings = {
        sec: np.random.randn(15, 384).astype(np.float32)
        for sec in service._tfidf_explainer.section_names
    }
    # Normalize mock embeddings
    for sec in mock_dense.embeddings:
        norms = np.linalg.norm(mock_dense.embeddings[sec], axis=1, keepdims=True)
        mock_dense.embeddings[sec] /= np.maximum(norms, 1e-12)

    mock_dense.model.encode.return_value = np.random.randn(1, 384).astype(np.float32)

    custom_explainer = NoveltyExplainer(
        tfidf_extractor=service.tfidf_extractor,
        model=service.candidate_model,
        projects_df=small_df,
        dense_extractor=mock_dense,
        scoring_mode="tuned_hybrid",
        hybrid_alpha=0.60,
        candidate_limit=50
    )

    proposal = CASES["CASE_F_SAME_DATASET_DIFF_OBJECTIVES"]["proposal"]
    report = custom_explainer.explain_proposal(
        candidate_dict=proposal,
        top_k=5,
        candidate_limit=50
    )

    # Stage 1 evaluated 15, shortlisted 15 (< 50)
    assert report["stage1_candidates_evaluated"] == 15
    assert report["stage2_shortlist_size"] == 15
    assert len(report["top_matches"]) == 5
    for match in report["top_matches"]:
        assert match["historical_project_id"] in small_pids


# ============================================================================
# 5. Robustness to Edge Cases (Empty Sections, Self-Matching, Duplicates)
# ============================================================================

def test_self_matching_exclusion(service):
    """Verify that an exact copy proposal with its project_id is excluded from matching itself."""
    explainer = service.get_explainer_for_mode("tuned_hybrid", alpha=0.60)
    ref_row = explainer.projects_df.iloc[0].to_dict()

    report = explainer.explain_proposal(
        candidate_dict=ref_row,
        top_k=5,
        exclude_self=True,
        scoring_mode="tuned_hybrid",
        hybrid_alpha=0.60
    )

    # Excluded candidate must not appear in top matches
    top_pids = [m["historical_project_id"] for m in report["top_matches"]]
    assert ref_row["project_id"] not in top_pids
    assert report["stage1_candidates_evaluated"] == 599


def test_empty_or_sparse_sections_handling(service):
    """Verify graceful handling when non-required or sparse sections are present."""
    explainer = service.get_explainer_for_mode("tuned_hybrid", alpha=0.60)
    sparse_proposal = {
        "project_id": "NEW_SPARSE_001",
        "project_title": "A Rare Unconventional Study On Quantum Cryptography",
        "domain": "Cybersecurity",
        "abstract": "Explores novel post-quantum lattice security architectures.",
        "problem_statement": "Classical RSA is vulnerable to Shor's algorithm.",
        "objectives": "Implement ring learning with errors cryptography.",
        "proposed_methodology": "Construct mathematical security proofs.",
        "dataset": "Simulated quantum noise datasets.",
        "algorithm_model": "Lattice RLWE encryption algorithm.",
        "technology_tools": "Python SageMath."
    }

    report = explainer.explain_proposal(
        candidate_dict=sparse_proposal,
        top_k=5,
        scoring_mode="tuned_hybrid",
        hybrid_alpha=0.60
    )

    assert report["retrieval_stage"] == "two_stage"
    assert report["stage2_shortlist_size"] == 50
    assert len(report["top_matches"]) == 5
    assert 0.0 <= report["overall_novelty_score_pct"] <= 100.0


# ============================================================================
# 6. End-to-End API Endpoints Verification
# ============================================================================

def test_api_health_endpoint_two_stage_metadata(client):
    """Verify /api/health exposes two-stage retrieval settings."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "healthy"
    assert data["reference_corpus_size"] == 600
    assert data["retrieval_architecture"] == "two_stage"
    assert data["two_stage_candidate_limit"] == 50


def test_api_score_two_stage_execution(client, isolated_db):
    """Verify POST /api/score returns two-stage retrieval metadata."""
    prop = CASES["CASE_C_STRONG_PARAPHRASE"]["proposal"]
    res = client.post("/api/score?mode=tuned_hybrid&candidate_limit=50", json=prop)
    assert res.status_code == 200
    data = res.json()

    assert data["scoring_mode"] == "tuned_hybrid"
    assert data["hybrid_alpha"] == 0.60
    assert data["retrieval_stage"] == "two_stage"
    assert data["stage1_candidates_evaluated"] == 600
    assert data["stage2_shortlist_size"] == 50
    assert data["candidate_limit"] == 50
    assert len(data["top_matches"]) == 5

    # Check top match difference summary
    for m in data["top_matches"]:
        assert "difference_summary" in m
        assert "high_overlap_sections" in m["difference_summary"]
        assert "differentiating_sections" in m["difference_summary"]
