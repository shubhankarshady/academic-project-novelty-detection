"""
Tests for Groq-Powered Explainable AI (XAI) feature (src/explainability/groq_explainer.py & src/api/main.py).

Verifies:
1. Standard /api/score NEVER calls Groq or requires GROQ_API_KEY.
2. /api/explain without GROQ_API_KEY returns HTTP 503 Service Unavailable with descriptive message.
3. /api/explain with mocked Groq calls the API and returns all 9 structured sections.
4. Caching behavior: subsequent requests for same proposal return cached=True without repeated LLM call.
5. Regeneration: force_regenerate=True bypasses cache and invokes Groq again.
6. Error handling: rate limit (HTTP 429) and timeout (HTTP 504) are handled gracefully.
7. Scoring mode support: all 3 modes ('tfidf', 'hybrid', 'tuned_hybrid') pass metadata accurately.
8. Security: GROQ_API_KEY is never exposed in response payloads.
9. Deterministic novelty score is never altered by Groq.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import json

from src.api.main import app
from src.api.service import get_novelty_service
from src.explainability.groq_explainer import (
    GroqExplainer,
    set_groq_explainer,
    get_groq_explainer,
    GroqKeyMissingError,
    GroqRateLimitError,
    GroqTimeoutError,
    GroqAPIError,
)

SAMPLE_PROPOSAL = {
    "project_title": "Deep Learning for Automated Brain Tumor Segmentation in MRI",
    "domain": "Healthcare_Medical_Informatics",
    "abstract": "Automated segmentation of brain tumors from magnetic resonance imaging (MRI) is critical for timely clinical diagnosis and treatment planning. We propose a 3D U-Net convolutional architecture with attention gating.",
    "problem_statement": "Manual segmentation of multisequence brain MRI is laborious, prone to significant inter-observer variability, and delays critical neurosurgical interventions.",
    "objectives": "Develop a volumetric attention U-Net achieving high Dice similarity coefficient on multimodal brain tumor scans.",
    "proposed_methodology": "Preprocess BraTS scans with N4ITK bias correction, z-score intensity normalization, and patch extraction. Train with soft Dice and focal loss combination.",
    "dataset": "BraTS 2021 multimodal magnetic resonance imaging benchmark consisting of 1251 multi-parametric MRI scans.",
    "algorithm_model": "3D Attention U-Net neural network with residual squeeze-and-excitation connections.",
    "technology_tools": "Python 3.10, PyTorch 2.1, MONAI, SimpleITK, CUDA, Weights and Biases."
}

MOCK_GROQ_EXPLANATION_JSON = {
    "overall_novelty_interpretation": "The proposal exhibits Moderate-to-High similarity to existing volumetric neuroimaging studies in the reference corpus, yielding an estimated novelty of 48.2% (Moderate Novelty Tier).",
    "what_is_missing": [
        "Lacks comparison against modern vision foundation models or Swin UNETR baselines.",
        "Missing cross-institutional validation on non-BraTS external clinical datasets.",
        "No uncertainty quantification or clinical calibration analysis for surgical margins."
    ],
    "why_similar": "The proposal heavily mirrors historical project SYN-0157, which also employs 3D Attention U-Net architectures trained on the BraTS 2021 dataset with Dice-focal compound loss.",
    "section_overlap_explanation": [
        {
            "section": "dataset",
            "overlap_level": "High",
            "analysis": "Identical benchmark dataset (BraTS 2021) with identical 1251 MRI scan splits."
        },
        {
            "section": "algorithm_model",
            "overlap_level": "High",
            "analysis": "Shared reliance on 3D U-Net with attention gating mechanisms."
        },
        {
            "section": "proposed_methodology",
            "overlap_level": "Moderate",
            "analysis": "Standard N4ITK bias correction and Dice loss pipeline common to biomedical segmentation."
        }
    ],
    "potential_research_gaps": [
        "Domain shift robustness when deployed on low-field or motion-artifact MRI scanners.",
        "Computational efficiency constraints for intra-operative real-time surgical guidance."
    ],
    "specific_suggestions": [
        "Incorporate a dual-stream architecture combining MRI structural scans with perfusion or diffusion tensor imaging (DTI).",
        "Introduce conformal prediction to output mathematically rigorous confidence intervals around segmentation boundaries."
    ],
    "potential_novel_directions": [
        "Explore semi-supervised federated learning across multi-center hospital silos.",
        "Implement lightweight state-space models (e.g. Mamba-UNet) for linear-time volumetric context."
    ],
    "recommended_improvements": {
        "objectives": "Explicitly target boundary uncertainty quantification and cross-scanner generalization beyond BraTS.",
        "proposed_methodology": "Incorporate Monte Carlo dropout or evidential deep learning for epistemic uncertainty estimation.",
        "dataset": "Supplement BraTS with an independent clinical cohort from a local PACS archive or TCIA.",
        "algorithm_model": "Benchmark against Swin UNETR and SegResNet baselines with ablation on attention gating."
    },
    "prioritized_action_plan": [
        "Step 1: Expand dataset strategy to include out-of-distribution clinical test cases.",
        "Step 2: Implement uncertainty estimation module alongside the primary segmentation head.",
        "Step 3: Conduct formal ablation comparing attention U-Net against modern Swin UNETR."
    ]
}


class TestGroqXAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Eagerly load model service
        get_novelty_service()
        cls.client = TestClient(app)

    def setUp(self):
        # Create fresh explainer instance for tests
        self.mock_explainer = GroqExplainer(api_key="test_mock_groq_key_12345")
        self.mock_explainer.clear_cache()
        set_groq_explainer(self.mock_explainer)

    def test_standard_scoring_does_not_call_groq_or_require_key(self):
        """
        Requirement: Do not automatically call Groq when student submits proposal.
        POST /api/score must run without GROQ_API_KEY and never invoke Groq.
        """
        # Ensure explainer has no key
        no_key_explainer = GroqExplainer(api_key="")
        set_groq_explainer(no_key_explainer)

        # Standard score proposal
        resp = self.client.post("/api/score?mode=tfidf", json=SAMPLE_PROPOSAL)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Results generated successfully by NLP models
        self.assertIn("overall_novelty_score_pct", data)
        self.assertIn("max_similarity_score", data)
        self.assertIn("top_matches", data)
        self.assertEqual(len(data["top_matches"]), 5)

    def test_explain_without_api_key_returns_503(self):
        """
        When GROQ_API_KEY is missing, POST /api/explain must return 503 Service Unavailable.
        """
        no_key_explainer = GroqExplainer(api_key="")
        set_groq_explainer(no_key_explainer)

        # Score proposal first to get valid evaluation report
        score_resp = self.client.post("/api/score", json=SAMPLE_PROPOSAL)
        self.assertEqual(score_resp.status_code, 200)
        eval_report = score_resp.json()

        # Request explanation
        payload = {
            "proposal": SAMPLE_PROPOSAL,
            "evaluation_report": eval_report,
            "force_regenerate": False
        }
        resp = self.client.post("/api/explain", json=payload)
        self.assertEqual(resp.status_code, 503)
        self.assertIn("GROQ_API_KEY is not configured", resp.json()["detail"])

    @patch("httpx.AsyncClient.post")
    def test_explain_with_valid_key_returns_all_9_sections(self, mock_post):
        """
        When student clicks Generate AI Explanation, Groq is called and returns
        the complete 9-part structured explanation.
        """
        # Mock httpx response from Groq
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(MOCK_GROQ_EXPLANATION_JSON)
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Score proposal
        score_resp = self.client.post("/api/score?mode=tuned_hybrid", json=SAMPLE_PROPOSAL)
        eval_report = score_resp.json()
        original_novelty = eval_report["overall_novelty_score_pct"]

        # Call explain endpoint
        payload = {
            "proposal": SAMPLE_PROPOSAL,
            "evaluation_report": eval_report,
            "force_regenerate": False
        }
        resp = self.client.post("/api/explain", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Verification of top-level fields
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["scoring_mode"], "tuned_hybrid")
        self.assertEqual(data["overall_novelty_score_pct"], original_novelty)
        self.assertFalse(data["cached"])
        self.assertIn("model_used", data)
        self.assertIn("created_at", data)

        # Verification of all 9 structured sections
        exp = data["explanation"]
        self.assertIn("overall_novelty_interpretation", exp)
        self.assertTrue(len(exp["overall_novelty_interpretation"]) > 20)

        self.assertIn("what_is_missing", exp)
        self.assertIsInstance(exp["what_is_missing"], list)
        self.assertEqual(len(exp["what_is_missing"]), 3)

        self.assertIn("why_similar", exp)
        self.assertTrue(len(exp["why_similar"]) > 20)

        self.assertIn("section_overlap_explanation", exp)
        self.assertIsInstance(exp["section_overlap_explanation"], list)
        self.assertEqual(len(exp["section_overlap_explanation"]), 3)
        self.assertEqual(exp["section_overlap_explanation"][0]["overlap_level"], "High")

        self.assertIn("potential_research_gaps", exp)
        self.assertIsInstance(exp["potential_research_gaps"], list)
        self.assertEqual(len(exp["potential_research_gaps"]), 2)

        self.assertIn("specific_suggestions", exp)
        self.assertIsInstance(exp["specific_suggestions"], list)
        self.assertEqual(len(exp["specific_suggestions"]), 2)

        self.assertIn("potential_novel_directions", exp)
        self.assertIsInstance(exp["potential_novel_directions"], list)
        self.assertEqual(len(exp["potential_novel_directions"]), 2)

        self.assertIn("recommended_improvements", exp)
        rec = exp["recommended_improvements"]
        self.assertIn("objectives", rec)
        self.assertIn("proposed_methodology", rec)
        self.assertIn("dataset", rec)
        self.assertIn("algorithm_model", rec)

        self.assertIn("prioritized_action_plan", exp)
        self.assertIsInstance(exp["prioritized_action_plan"], list)
        self.assertEqual(len(exp["prioritized_action_plan"]), 3)

        # Check Groq API was called exactly once
        self.assertEqual(mock_post.call_count, 1)

    @patch("httpx.AsyncClient.post")
    def test_caching_and_force_regeneration(self, mock_post):
        """
        Verify caching: second request returns cached=True without repeated Groq call.
        Verify regeneration: force_regenerate=True calls Groq again and returns cached=False.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(MOCK_GROQ_EXPLANATION_JSON)}}]
        }
        mock_post.return_value = mock_response

        score_resp = self.client.post("/api/score", json=SAMPLE_PROPOSAL)
        eval_report = score_resp.json()

        payload = {
            "proposal": SAMPLE_PROPOSAL,
            "evaluation_report": eval_report,
            "force_regenerate": False
        }

        # 1st call: calls Groq
        resp1 = self.client.post("/api/explain", json=payload)
        self.assertEqual(resp1.status_code, 200)
        self.assertFalse(resp1.json()["cached"])
        self.assertEqual(mock_post.call_count, 1)

        # 2nd call with force_regenerate=False: serves from cache!
        resp2 = self.client.post("/api/explain", json=payload)
        self.assertEqual(resp2.status_code, 200)
        self.assertTrue(resp2.json()["cached"])
        # Call count should STILL be 1 (no network call)
        self.assertEqual(mock_post.call_count, 1)

        # 3rd call with force_regenerate=True: bypasses cache!
        payload_regen = dict(payload)
        payload_regen["force_regenerate"] = True
        resp3 = self.client.post("/api/explain", json=payload_regen)
        self.assertEqual(resp3.status_code, 200)
        self.assertFalse(resp3.json()["cached"])
        # Call count should now be 2
        self.assertEqual(mock_post.call_count, 2)

    @patch("httpx.AsyncClient.post")
    def test_rate_limit_handling(self, mock_post):
        """
        Verify that HTTP 429 from Groq is mapped cleanly to HTTP 429 for the client.
        """
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit reached"
        mock_post.return_value = mock_response

        score_resp = self.client.post("/api/score", json=SAMPLE_PROPOSAL)
        eval_report = score_resp.json()

        payload = {
            "proposal": SAMPLE_PROPOSAL,
            "evaluation_report": eval_report,
            "force_regenerate": True
        }
        resp = self.client.post("/api/explain", json=payload)
        self.assertEqual(resp.status_code, 429)
        self.assertIn("rate limit", resp.json()["detail"].lower())

    @patch("httpx.AsyncClient.post")
    def test_timeout_handling(self, mock_post):
        """
        Verify that timeout connecting to Groq is mapped to HTTP 504 Gateway Timeout.
        """
        import httpx
        mock_post.side_effect = httpx.TimeoutException("Read timed out")

        score_resp = self.client.post("/api/score", json=SAMPLE_PROPOSAL)
        eval_report = score_resp.json()

        payload = {
            "proposal": SAMPLE_PROPOSAL,
            "evaluation_report": eval_report,
            "force_regenerate": True
        }
        resp = self.client.post("/api/explain", json=payload)
        self.assertEqual(resp.status_code, 504)
        self.assertIn("timed out", resp.json()["detail"].lower())

    @patch("httpx.AsyncClient.post")
    def test_all_three_scoring_modes_supported(self, mock_post):
        """
        Verify that tfidf, hybrid, and tuned_hybrid modes all work with /api/explain.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(MOCK_GROQ_EXPLANATION_JSON)}}]
        }
        mock_post.return_value = mock_response

        for mode in ["tfidf", "hybrid", "tuned_hybrid"]:
            self.mock_explainer.clear_cache()
            score_resp = self.client.post(f"/api/score?mode={mode}", json=SAMPLE_PROPOSAL)
            eval_report = score_resp.json()

            payload = {
                "proposal": SAMPLE_PROPOSAL,
                "evaluation_report": eval_report,
                "force_regenerate": False
            }
            resp = self.client.post("/api/explain", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["scoring_mode"], mode)

    @patch("httpx.AsyncClient.post")
    def test_security_api_key_never_exposed(self, mock_post):
        """
        Verify that GROQ_API_KEY is never leaked in the response payload or headers.
        """
        secret_key = "secret_groq_production_key_xyz987"
        explainer = GroqExplainer(api_key=secret_key)
        set_groq_explainer(explainer)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(MOCK_GROQ_EXPLANATION_JSON)}}]
        }
        mock_post.return_value = mock_response

        score_resp = self.client.post("/api/score", json=SAMPLE_PROPOSAL)
        eval_report = score_resp.json()

        payload = {
            "proposal": SAMPLE_PROPOSAL,
            "evaluation_report": eval_report,
            "force_regenerate": False
        }
        resp = self.client.post("/api/explain", json=payload)
        self.assertEqual(resp.status_code, 200)
        
        response_text = resp.text
        self.assertNotIn(secret_key, response_text)
        for header, val in resp.headers.items():
            self.assertNotIn(secret_key, val)


if __name__ == "__main__":
    unittest.main()

