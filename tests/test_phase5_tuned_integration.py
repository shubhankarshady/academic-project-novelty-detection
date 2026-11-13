"""
Integration and End-to-End Tests for Phase 5.5: Tuned Hybrid Model Integration.
Validates:
1. Health endpoint with available scoring modes and tuned configuration metadata.
2. Three supported scoring modes: tfidf, hybrid (alpha 0.50), tuned_hybrid (alpha 0.60).
3. Seven required end-to-end evaluation scenarios:
   - Exact copy of reference project (with and without explicit project_id self-exclusion).
   - Strong paraphrase.
   - Same domain, different problem.
   - Different domains sharing common technology.
   - Same dataset with different objectives.
   - Incomplete / empty proposal (validation errors).
   - Unrelated proposal.
4. Model artifact integrity and dynamic configuration loading from JSON.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_db
from src.api.service import NoveltyService, get_novelty_service, ROOT_DIR, DEFAULT_TUNED_CONFIG_PATH
from src.storage.db import DatabaseManager

# Load reference cases from semantic feasibility script
from scripts.evaluate_semantic_feasibility import CONTROLLED_TEST_CASES

CASES = {c["case_id"]: c for c in CONTROLLED_TEST_CASES}


@pytest.fixture(scope="module")
def client():
    """Shared FastAPI test client with loaded ML models."""
    return TestClient(app)


@pytest.fixture(scope="module")
def service():
    """Singleton novelty service."""
    return get_novelty_service()


@pytest.fixture
def isolated_db():
    """In-memory database for clean test isolation."""
    db = DatabaseManager(":memory:")
    app.dependency_overrides[get_db] = lambda: db
    yield db
    app.dependency_overrides.clear()
    db.close()


# ---------------------------------------------------------------------------
# 1. Health and Configuration Endpoints
# ---------------------------------------------------------------------------

def test_health_reports_all_scoring_modes_and_tuned_config(client):
    """Verify /api/health reports active mode, available modes, and tuned config."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "healthy"
    assert data["reference_corpus_size"] == 600
    assert data["active_scoring_mode"] in {"tfidf", "hybrid", "tuned_hybrid"}
    assert "available_scoring_modes" in data
    assert data["available_scoring_modes"] == ["tfidf", "hybrid", "tuned_hybrid"]

    # Tuned config metadata must be present
    assert "tuned_hybrid_config" in data
    cfg = data["tuned_hybrid_config"]
    assert cfg is not None
    assert cfg["model_name"] == "Phase 5 Tuned Hybrid Fusion Model"
    assert cfg["tuned_hyperparameters"]["alpha"] == 0.60
    assert cfg["section_weights"]["weights_status"] == "FROZEN_CANDIDATE_CONSTRAINED_WEIGHTS"


def test_tuned_config_dynamic_loading():
    """Verify NoveltyService loads tuned configuration dynamically from JSON."""
    assert DEFAULT_TUNED_CONFIG_PATH.exists()
    with open(DEFAULT_TUNED_CONFIG_PATH, "r", encoding="utf-8") as f:
        file_cfg = json.load(f)

    srv = NoveltyService()
    assert srv.tuned_alpha == file_cfg["tuned_hyperparameters"]["alpha"]
    assert srv.tuned_version == file_cfg["model_version"]
    assert srv.tuned_alpha == 0.60


# ---------------------------------------------------------------------------
# 2. Scoring Modes Verification (TF-IDF vs Hybrid vs Tuned Hybrid)
# ---------------------------------------------------------------------------

def test_scoring_mode_tfidf(client, isolated_db):
    """Verify TF-IDF scoring mode returns correct metadata and formula."""
    prop = CASES["CASE_D_SAME_DOMAIN_DIFF_PROBLEM"]["proposal"]
    res = client.post("/api/score?mode=tfidf", json=prop)
    assert res.status_code == 200
    data = res.json()

    assert data["scoring_mode"] == "tfidf"
    assert data["hybrid_alpha"] == 1.0
    assert "predicted_similarity = sum(w_i * tfidf_similarity_i) + intercept" in data["scoring_formula"]
    assert data["model_version"] == "1.0-tfidf"
    for m in data["top_matches"]:
        assert m["scoring_mode"] == "tfidf"
        assert m["hybrid_alpha"] == 1.0


def test_scoring_mode_hybrid_baseline(client, isolated_db):
    """Verify Hybrid baseline mode uses alpha=0.50 and returns hybrid metadata."""
    prop = CASES["CASE_D_SAME_DOMAIN_DIFF_PROBLEM"]["proposal"]
    res = client.post("/api/score?mode=hybrid", json=prop)
    assert res.status_code == 200
    data = res.json()

    assert data["scoring_mode"] == "hybrid"
    assert data["hybrid_alpha"] == 0.50
    assert "0.50 * tfidf_similarity + 0.50 * semantic_similarity" in data["scoring_formula"]
    assert data["model_version"] == "1.0-hybrid"
    for m in data["top_matches"]:
        assert m["scoring_mode"] == "hybrid"
        assert m["hybrid_alpha"] == 0.50


def test_scoring_mode_tuned_hybrid(client, isolated_db):
    """Verify Tuned Hybrid mode uses alpha=0.60 and returns tuned metadata."""
    prop = CASES["CASE_D_SAME_DOMAIN_DIFF_PROBLEM"]["proposal"]
    res = client.post("/api/score?mode=tuned_hybrid", json=prop)
    assert res.status_code == 200
    data = res.json()

    assert data["scoring_mode"] == "tuned_hybrid"
    assert data["hybrid_alpha"] == 0.60
    assert "0.60 * tfidf_similarity + 0.40 * semantic_similarity" in data["scoring_formula"]
    assert data["model_version"] == "phase5_tuned_hybrid_v1.0"
    for m in data["top_matches"]:
        assert m["scoring_mode"] == "tuned_hybrid"
        assert m["hybrid_alpha"] == 0.60


def test_scoring_mode_invalid_raises_422(client, isolated_db):
    """Verify invalid scoring mode raises HTTP 422 Unprocessable Entity."""
    prop = CASES["CASE_D_SAME_DOMAIN_DIFF_PROBLEM"]["proposal"]
    res = client.post("/api/score?mode=quantum_random", json=prop)
    assert res.status_code == 422
    assert "Invalid scoring mode" in res.json()["detail"]


# ---------------------------------------------------------------------------
# 3. Seven End-to-End Scenarios
# ---------------------------------------------------------------------------

def test_scenario_1_exact_copy_not_excluded_without_id(client, isolated_db):
    """
    Scenario 1a: Exact copy of SYN-0001 submitted WITHOUT explicit reference project_id.
    Must NOT be excluded; in tuned_hybrid and hybrid modes, SYN-0001 must be #1 with 4.0 similarity.
    In tfidf mode, candidate copy matches the identical cluster with high similarity.
    """
    prop = dict(CASES["CASE_A_EXACT_COPY"]["proposal"])
    prop["project_id"] = None  # User proposal has no matching ID

    # 1. Tuned Hybrid: candidate constrained model guarantees exact copy reaches 4.0
    res_tuned = client.post("/api/score?mode=tuned_hybrid", json=prop)
    assert res_tuned.status_code == 200
    data_tuned = res_tuned.json()
    assert data_tuned["top_matches"][0]["historical_project_id"] == "SYN-0001"
    assert data_tuned["top_matches"][0]["predicted_similarity"] == 4.0
    assert data_tuned["overall_novelty_score_pct"] == 0.0

    # 2. Hybrid Baseline: also achieves 4.0 on exact copy
    res_hybrid = client.post("/api/score?mode=hybrid", json=prop)
    assert res_hybrid.status_code == 200
    data_hybrid = res_hybrid.json()
    assert data_hybrid["top_matches"][0]["historical_project_id"] == "SYN-0001"
    assert data_hybrid["top_matches"][0]["predicted_similarity"] == 4.0

    # 3. TF-IDF Baseline: matches the exact/near-duplicate cluster without being excluded
    res_tfidf = client.post("/api/score?mode=tfidf", json=prop)
    assert res_tfidf.status_code == 200
    data_tfidf = res_tfidf.json()
    assert data_tfidf["corpus_size_evaluated"] == 600
    assert data_tfidf["max_similarity_score"] >= 2.50


def test_scenario_1_exact_copy_excluded_with_explicit_id(client, isolated_db):
    """
    Scenario 1b: Exact copy of SYN-0001 submitted WITH explicit project_id='SYN-0001'.
    Must be strictly self-excluded; SYN-0001 must NOT appear anywhere in the results.
    """
    prop = dict(CASES["CASE_A_EXACT_COPY"]["proposal"])
    prop["project_id"] = "SYN-0001"  # Explicitly matches existing reference ID

    for mode in ["tfidf", "hybrid", "tuned_hybrid"]:
        res = client.post(f"/api/score?mode={mode}", json=prop)
        assert res.status_code == 200
        data = res.json()

        match_ids = [m["historical_project_id"] for m in data["top_matches"]]
        assert "SYN-0001" not in match_ids
        assert data["corpus_size_evaluated"] == 599  # Exactly 1 project excluded


def test_scenario_2_strong_paraphrase(client, isolated_db):
    """
    Scenario 2: Strong paraphrase of SYN-0001 (reordered syntax, alternative vocabulary).
    Evaluated across scoring modes; Tuned Hybrid captures semantic and topical alignment.
    """
    prop = dict(CASES["CASE_C_STRONG_PARAPHRASE"]["proposal"])
    prop["project_id"] = None

    res_tuned = client.post("/api/score?mode=tuned_hybrid", json=prop)
    assert res_tuned.status_code == 200
    data_tuned = res_tuned.json()

    assert data_tuned["scoring_mode"] == "tuned_hybrid"
    assert data_tuned["hybrid_alpha"] == 0.60
    assert len(data_tuned["top_matches"]) == 5
    # Topical and semantic retrieval surfaces related disease/vision projects
    assert data_tuned["max_similarity_score"] > 1.20


def test_scenario_3_same_domain_diff_problem(client, isolated_db):
    """
    Scenario 3: Same domain (AI/ML), completely different problem (Fraud vs Crop disease).
    Should show low similarity and high novelty across all modes.
    """
    prop = dict(CASES["CASE_D_SAME_DOMAIN_DIFF_PROBLEM"]["proposal"])
    prop["project_id"] = None

    for mode in ["tfidf", "hybrid", "tuned_hybrid"]:
        res = client.post(f"/api/score?mode={mode}", json=prop)
        assert res.status_code == 200
        data = res.json()

        assert data["overall_novelty_score_pct"] >= 55.0
        assert data["max_similarity_score"] <= 1.8


def test_scenario_4_diff_domain_common_technology(client, isolated_db):
    """
    Scenario 4: Different domain (Networking vs Agriculture) sharing technology (CNN, Python).
    Superficial tool overlap should yield low similarity and high novelty.
    """
    prop = dict(CASES["CASE_E_DIFF_DOMAIN_COMMON_TECH"]["proposal"])
    prop["project_id"] = None

    for mode in ["tfidf", "hybrid", "tuned_hybrid"]:
        res = client.post(f"/api/score?mode={mode}", json=prop)
        assert res.status_code == 200
        data = res.json()

        assert data["overall_novelty_score_pct"] >= 60.0
        assert data["max_similarity_score"] <= 1.6


def test_scenario_5_same_dataset_diff_objectives(client, isolated_db):
    """
    Scenario 5: Same dataset (PlantVillage), completely different objective (Inpainting vs Disease detection).
    Dataset similarity is high, but methodology/problem differences produce moderate similarity.
    """
    prop = dict(CASES["CASE_F_SAME_DATASET_DIFF_OBJECTIVES"]["proposal"])
    prop["project_id"] = None

    for mode in ["tfidf", "hybrid", "tuned_hybrid"]:
        res = client.post(f"/api/score?mode={mode}", json=prop)
        assert res.status_code == 200
        data = res.json()

        top_match = data["top_matches"][0]
        # In all modes, dataset similarity is captured
        assert "dataset" in top_match["section_similarity_scores"]
        assert top_match["section_similarity_scores"]["dataset"] >= 0.80
        # But overall similarity is moderate, not a duplicate
        assert 1.0 <= data["max_similarity_score"] <= 2.8


def test_scenario_6_incomplete_and_invalid_proposals(client, isolated_db):
    """
    Scenario 6: Incomplete proposals (empty sections, short sections, missing fields).
    Must return HTTP 422 Unprocessable Entity with descriptive error messages.
    """
    base_prop = CASES["CASE_A_EXACT_COPY"]["proposal"]

    # 1. Missing required section (abstract omitted)
    prop_missing = dict(base_prop)
    del prop_missing["abstract"]
    res = client.post("/api/score", json=prop_missing)
    assert res.status_code == 422
    assert any("abstract" in err["loc"] for err in res.json()["detail"])

    # 2. Empty whitespace in problem_statement
    prop_empty = dict(base_prop)
    prop_empty["problem_statement"] = "     "
    res = client.post("/api/score", json=prop_empty)
    assert res.status_code == 422
    assert any("problem_statement" in err["loc"] for err in res.json()["detail"])

    # 3. Too short methodology (< 10 chars)
    prop_short = dict(base_prop)
    prop_short["proposed_methodology"] = "Too short"
    res = client.post("/api/score", json=prop_short)
    assert res.status_code == 422
    assert any("proposed_methodology" in err["loc"] for err in res.json()["detail"])

    # 4. Null value in dataset
    prop_null = dict(base_prop)
    prop_null["dataset"] = None
    res = client.post("/api/score", json=prop_null)
    assert res.status_code == 422


def test_scenario_7_unrelated_proposal(client, isolated_db):
    """
    Scenario 7: Completely unrelated academic proposal (Quantum Topography / Astrobiology).
    Should produce near-zero similarity, exceptionally high novelty, across all 3 modes.
    """
    unrelated_prop = {
        "project_title": "Quantum Decoherence Characterization in Superconducting Transmon Qubits",
        "domain": "Quantum Physics & Computing",
        "abstract": "We investigate flux-tunable quantum state tomography to characterize high-frequency dielectric loss and two-level system defect clusters in planar transmon architectures.",
        "problem_statement": "Superconducting qubit coherence times are fundamentally bounded by non-equilibrium quasiparticle poisoning and amorphous dielectric surface interfaces.",
        "objectives": "1) Synthesize niobium resonator junctions; 2) measure T1 relaxation and Ramsey T2 dephasing; 3) fit master equation dissipation dynamics.",
        "proposed_methodology": "Implement quantum process tomography in a dilution refrigerator at 10mK using vector network analyzers and microwave pulse shaping.",
        "dataset": "Empirical cryogenic S-parameter transmission measurements",
        "algorithm_model": "Lindblad Quantum Master Equation Dissipative Dynamics",
        "technology_tools": "Qiskit Pulse, LabBER, Keysight AWG, QuTiP, Python"
    }

    for mode in ["tfidf", "hybrid", "tuned_hybrid"]:
        res = client.post(f"/api/score?mode={mode}", json=unrelated_prop)
        assert res.status_code == 200
        data = res.json()

        assert data["overall_novelty_score_pct"] >= 80.0
        assert data["max_similarity_score"] <= 0.8
        assert "High" in data["novelty_assessment_tier"]
