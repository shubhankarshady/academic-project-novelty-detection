"""
Unit and Integration Tests for FastAPI Backend (src/api/).

Tests:
1. GET /api/health: Status, reference corpus size (600), and submitted proposal count.
2. POST /api/score:
   - Valid 8-section proposal returns 200 OK with novelty score, top 5 matches, and disclaimers.
   - Scoring does NOT write to SQLite.
   - Top matches come from historical reference corpus.
   - Missing/empty section returns 422 Unprocessable Entity.
   - Self-matching proposal is excluded from its own candidate matches.
3. POST /api/save:
   - Valid matching proposal and report returns 201 Created.
   - Mismatched report (title difference) returns 400 Bad Request.
   - Incomplete report returns 400 Bad Request.
   - Duplicate submission returns 409 Conflict.
4. GET /api/proposals:
   - Paginated listing with limit/offset.
5. GET /api/proposals/{project_id}:
   - Existing proposal returns full details and evaluation report.
   - Non-existent ID returns 404 Not Found.
"""

import unittest
from fastapi.testclient import TestClient

from src.api.main import app, get_db
from src.api.service import get_novelty_service
from src.storage.db import DatabaseManager

class TestFastAPIBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize the real NoveltyService and FastAPI TestClient."""
        # Eagerly load the trained model service
        cls.service = get_novelty_service()
        cls.client = TestClient(app)

    def setUp(self):
        """Set up an isolated in-memory database for each test."""
        self.test_db = DatabaseManager(":memory:")
        app.dependency_overrides[get_db] = lambda: self.test_db

        self.valid_proposal = {
            "project_title": "Federated Learning for Privacy-Preserving Medical Diagnostics",
            "domain": "Healthcare AI",
            "abstract": "We propose a federated learning framework using convolutional neural networks to train diagnostic models across distributed hospital datasets without centralizing sensitive patient records.",
            "problem_statement": "Medical imaging datasets are siloed due to strict HIPAA regulations and patient confidentiality concerns.",
            "objectives": "Train high-accuracy classification models across multi-site hospitals while maintaining patient privacy.",
            "proposed_methodology": "Implement FedAvg aggregation with differential privacy noise simulated across multiple client nodes.",
            "dataset": "BraTS multi-institutional magnetic resonance imaging benchmarks.",
            "algorithm_model": "3D U-Net and ResNet-50 Convolutional Neural Networks.",
            "technology_tools": "PyTorch, Flower Federated Framework, Python, CUDA."
        }

    def tearDown(self):
        """Clear dependency overrides and clean up in-memory database."""
        app.dependency_overrides.clear()
        self.test_db.close()

    def test_health_endpoint(self):
        """Verify GET /api/health returns status, 600 reference projects, and submitted count."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["reference_corpus_size"], 600)
        self.assertEqual(data["submitted_proposals_count"], 0)
        self.assertTrue(data["model_loaded"])
        self.assertIn("600 baseline historical projects", data["message"])
        self.assertIn("0 user-submitted proposals", data["message"])

    def test_score_valid_proposal(self):
        """Verify POST /api/score evaluates candidate, returns top 5 matches, and DOES NOT write to SQLite."""
        initial_db_count = self.test_db.count_proposals()
        self.assertEqual(initial_db_count, 0)

        response = self.client.post("/api/score", json=self.valid_proposal)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check required fields
        self.assertEqual(data["candidate_proposal_title"], self.valid_proposal["project_title"])
        self.assertEqual(data["corpus_size_evaluated"], 600)
        self.assertIn("overall_novelty_score_pct", data)
        self.assertIn("max_similarity_score", data)
        self.assertIn("novelty_assessment_tier", data)
        self.assertIn("top_matches", data)
        self.assertEqual(len(data["top_matches"]), 5)

        # Verify top matches come from historical reference corpus (SYN-XXXX)
        for match in data["top_matches"]:
            self.assertTrue(match["historical_project_id"].startswith("SYN-"))
            self.assertIn("difference_summary", match)
            self.assertIn("ranked_sections", match)
            self.assertIn("section_evidence", match)

        # Crucial check: Scoring must NOT save the proposal to the database
        post_score_db_count = self.test_db.count_proposals()
        self.assertEqual(post_score_db_count, 0, "Scoring must NOT write to SQLite!")

    def test_score_invalid_empty_section(self):
        """Verify POST /api/score returns 422 when a required section is empty or whitespace."""
        invalid_proposal = dict(self.valid_proposal, abstract="   ")
        response = self.client.post("/api/score", json=invalid_proposal)
        self.assertEqual(response.status_code, 422)

    def test_score_self_exclusion(self):
        """Verify candidate proposal cannot match itself if submitted with an existing corpus ID or text."""
        # Proposal with exact title of baseline project SYN-0001
        corpus_project = self.service.explainer.projects_df.iloc[0]
        self_matching_candidate = {
            "project_id": str(corpus_project["project_id"]),
            "project_title": str(corpus_project["project_title"]),
            "abstract": str(corpus_project["abstract"]),
            "problem_statement": str(corpus_project["problem_statement"]),
            "objectives": str(corpus_project["objectives"]),
            "proposed_methodology": str(corpus_project["proposed_methodology"]),
            "dataset": str(corpus_project["dataset"]),
            "algorithm_model": str(corpus_project["algorithm_model"]),
            "technology_tools": str(corpus_project["technology_tools"])
        }

        response = self.client.post("/api/score", json=self_matching_candidate)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Evaluated corpus size should be decremented due to self-matching exclusion (< 600)
        self.assertLess(data["corpus_size_evaluated"], 600)
        matched_ids = [m["historical_project_id"] for m in data["top_matches"]]
        self.assertNotIn(str(corpus_project["project_id"]), matched_ids)

    def test_save_valid_proposal_and_retrieval(self):
        """Verify POST /api/save saves proposal with matching report and permits subsequent retrieval."""
        # 1. Score proposal first
        score_resp = self.client.post("/api/score", json=self.valid_proposal)
        self.assertEqual(score_resp.status_code, 200)
        eval_report = score_resp.json()

        # 2. Save proposal with matching evaluation report
        save_payload = {
            "proposal": self.valid_proposal,
            "evaluation_report": eval_report
        }
        save_resp = self.client.post("/api/save", json=save_payload)
        self.assertEqual(save_resp.status_code, 201)
        save_data = save_resp.json()

        self.assertEqual(save_data["status"], "saved")
        self.assertEqual(save_data["project_id"], "SUB-0001")
        self.assertEqual(save_data["project_title"], self.valid_proposal["project_title"])

        # Database count should now be 1
        self.assertEqual(self.test_db.count_proposals(), 1)

        # 3. Retrieve proposal by project ID
        get_resp = self.client.get(f"/api/proposals/{save_data['project_id']}")
        self.assertEqual(get_resp.status_code, 200)
        record = get_resp.json()
        self.assertEqual(record["project_title"], self.valid_proposal["project_title"])
        self.assertIn("evaluation_report", record)
        self.assertEqual(record["evaluation_report"]["candidate_proposal_title"], self.valid_proposal["project_title"])

    def test_save_mismatched_report_rejected(self):
        """Verify POST /api/save rejects save request if report title does not match proposal title."""
        mismatched_report = {
            "candidate_proposal_title": "Completely Different Medical Project",
            "max_similarity_score": 0.5,
            "novelty_assessment_tier": "Moderate Novelty (Partial Overlap)",
            "top_matches": []
        }
        payload = {
            "proposal": self.valid_proposal,
            "evaluation_report": mismatched_report
        }
        response = self.client.post("/api/save", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("does not match", response.json()["detail"].lower())

    def test_save_incomplete_report_rejected(self):
        """Verify POST /api/save rejects save request if evaluation report lacks required fields."""
        incomplete_report = {
            "candidate_proposal_title": self.valid_proposal["project_title"],
            # Missing max_similarity_score and novelty_assessment_tier
        }
        payload = {
            "proposal": self.valid_proposal,
            "evaluation_report": incomplete_report
        }
        response = self.client.post("/api/save", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing required field", response.json()["detail"].lower())

    def test_save_duplicate_rejected_with_409(self):
        """Verify saving duplicate proposals returns HTTP 409 Conflict without silent overwrite."""
        score_resp = self.client.post("/api/score", json=self.valid_proposal)
        eval_report = score_resp.json()

        payload = {
            "proposal": self.valid_proposal,
            "evaluation_report": eval_report
        }
        # First save succeeds
        resp_1 = self.client.post("/api/save", json=payload)
        self.assertEqual(resp_1.status_code, 201)

        # Second save with identical content returns 409
        resp_2 = self.client.post("/api/save", json=payload)
        self.assertEqual(resp_2.status_code, 409)
        detail = resp_2.json()["detail"]
        self.assertEqual(detail["error"], "DuplicateProposalError")
        self.assertEqual(detail["existing_project_id"], "SUB-0001")

        # Database count remains 1
        self.assertEqual(self.test_db.count_proposals(), 1)

    def test_list_proposals_and_pagination(self):
        """Verify GET /api/proposals lists saved proposals with limit and offset."""
        for i in range(1, 4):
            prop = dict(
                self.valid_proposal,
                project_title=f"Federated AI Project {i}",
                abstract=f"Abstract variation {i} for testing."
            )
            score_resp = self.client.post("/api/score", json=prop)
            self.client.post("/api/save", json={"proposal": prop, "evaluation_report": score_resp.json()})

        # List all
        resp = self.client.get("/api/proposals?limit=2&offset=0")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 3)
        self.assertEqual(len(data["proposals"]), 2)
        # Newest first
        self.assertEqual(data["proposals"][0]["project_title"], "Federated AI Project 3")
        self.assertEqual(data["proposals"][1]["project_title"], "Federated AI Project 2")

        # Offset 2
        resp_page2 = self.client.get("/api/proposals?limit=2&offset=2")
        self.assertEqual(resp_page2.status_code, 200)
        data_page2 = resp_page2.json()
        self.assertEqual(len(data_page2["proposals"]), 1)
        self.assertEqual(data_page2["proposals"][0]["project_title"], "Federated AI Project 1")

    def test_get_nonexistent_proposal_returns_404(self):
        """Verify GET /api/proposals/{project_id} returns 404 for missing proposal."""
        resp = self.client.get("/api/proposals/SUB-NONEXISTENT")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json()["detail"].lower())

if __name__ == "__main__":
    unittest.main()
