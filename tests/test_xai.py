"""
Unit and Integration Tests for Explainable AI (XAI) Module and Scoring CLI.

Tests:
1. Exact keyword evidence from source text (substring presence verification).
2. Correct section-to-score mapping.
3. Additive contribution sum matching the Ridge prediction.
4. Handling of empty or missing sections.
5. No self-matching (exclusion of exact proposal match).
6. Deterministic ranking across multiple invocations.
7. Correct JSON output structure from the scoring CLI.
"""

import unittest
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.section_tfidf import SectionTFIDFExtractor, SECTION_COLUMNS, FEATURE_NAMES
from src.explainability.explainer import (
    NoveltyExplainer,
    extract_snippet,
    is_informative_term,
    verify_snippet,
    GENERIC_LOW_INFO_TERMS
)
from src.utils.metrics import get_novelty_level, calculate_novelty_score

class TestXAIExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up small, synthetic test fixture with fitted vectorizers and a Ridge model."""
        # 3 synthetic reference projects
        cls.synthetic_corpus = pd.DataFrame([
            {
                "project_id": "SYN-TEST-001",
                "project_title": "Deep Learning for Medical Image Segmentation",
                "domain": "Healthcare",
                "abstract": "We develop a convolutional neural network architecture for brain tumor segmentation on clinical MRI scans.",
                "problem_statement": "Manual segmentation of brain lesions is labor intensive and prone to inter-observer variability.",
                "objectives": "Automate lesion boundary delineation using deep learning models.",
                "proposed_methodology": "Train a 3D U-Net convolutional model with Dice loss optimization.",
                "dataset": "BraTS benchmark brain tumor magnetic resonance dataset.",
                "algorithm_model": "3D U-Net Convolutional Neural Network.",
                "technology_tools": "Python, PyTorch, CUDA, SimpleITK."
            },
            {
                "project_id": "SYN-TEST-002",
                "project_title": "Blockchain Smart Contracts for Supply Chain Traceability",
                "domain": "Blockchain",
                "abstract": "A decentralized ledger framework using Ethereum smart contracts for transparent agricultural supply chains.",
                "problem_statement": "Counterfeiting and opacity in organic food logistics cause substantial economic losses.",
                "objectives": "Enable immutable audit logs and real-time shipment verification.",
                "proposed_methodology": "Deploy Solidity contracts on Ethereum testnet with IPFS storage.",
                "dataset": "Simulated cold chain RFID tracking telemetry logs.",
                "algorithm_model": "Proof of Authority consensus verification.",
                "technology_tools": "Solidity, Ethereum, Web3.js, Truffle."
            },
            {
                "project_id": "SYN-TEST-003",
                "project_title": "Reinforcement Learning for Autonomous Drone Navigation",
                "domain": "Robotics",
                "abstract": "Deep Q-learning agent for obstacle avoidance and trajectory optimization in GPS-denied environments.",
                "problem_statement": "Unmanned aerial vehicles struggle to navigate cluttered indoor spaces safely.",
                "objectives": "Achieve collision-free path planning in real time.",
                "proposed_methodology": "Simulate quadrotor dynamics in Gazebo using PPO reinforcement learning.",
                "dataset": "Gazebo indoor obstacle point cloud recordings.",
                "algorithm_model": "Proximal Policy Optimization PPO.",
                "technology_tools": "ROS, Gazebo, PyTorch, Python."
            }
        ])

        # Fit TFIDF extractor on the synthetic corpus
        cls.tfidf_extractor = SectionTFIDFExtractor(
            section_cols=SECTION_COLUMNS,
            ngram_range=(1, 2),
            stop_words="english"
        )
        cls.tfidf_extractor.fit_transform_projects(cls.synthetic_corpus)

        # Synthetic Ridge model with known coefficients
        cls.ridge_model = Ridge(alpha=10.0, random_state=42)
        # 8 features: title, abstract, problem, objectives, methodology, dataset, algorithm, technology
        # Fit on synthetic feature inputs to establish coefficients
        X_dummy = np.array([
            [1.0, 0.8, 0.5, 0.6, 0.7, 0.9, 0.4, 0.5],
            [0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
            [0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1]
        ], dtype=float)
        y_dummy = np.array([3.8, 0.5, 0.1], dtype=float)
        cls.ridge_model.fit(X_dummy, y_dummy)

        cls.explainer = NoveltyExplainer(
            tfidf_extractor=cls.tfidf_extractor,
            model=cls.ridge_model,
            projects_df=cls.synthetic_corpus
        )

    def test_exact_keyword_evidence_from_source_text(self):
        """Verify that every extracted keyword physically appears in both candidate and historical text."""
        candidate = {
            "project_title": "Automated Brain Tumor Segmentation with Convolutional Networks",
            "abstract": "This study applies convolutional neural network methods to MRI brain scans for tumor segmentation.",
            "problem_statement": "Brain lesions are difficult to segment manually.",
            "objectives": "Automate tumor segmentation using neural architectures.",
            "proposed_methodology": "Use 3D U-Net convolutional layers.",
            "dataset": "BraTS brain tumor dataset.",
            "algorithm_model": "U-Net CNN.",
            "technology_tools": "Python, PyTorch."
        }

        # Compare against SYN-TEST-001 (Medical Imaging)
        hist_proj = self.synthetic_corpus.iloc[0]
        explanation = self.explainer.explain_pair(candidate, hist_proj)

        # Check section evidence
        for sec_name, ev in explanation["section_evidence"].items():
            col = SECTION_COLUMNS[sec_name]
            cand_text = candidate.get(col, "").lower()
            hist_text = str(hist_proj.get(col, "")).lower()

            for term_info in ev["shared_terms"]:
                term = term_info["term"].lower()
                # Assertion: Term MUST physically appear in candidate source text
                self.assertIn(
                    term, cand_text,
                    f"Term '{term}' claimed in section '{sec_name}' but NOT found in candidate text!"
                )
                # Assertion: Term MUST physically appear in historical source text
                self.assertIn(
                    term, hist_text,
                    f"Term '{term}' claimed in section '{sec_name}' but NOT found in historical text!"
                )
                # Assertion: Snippets contain the term
                self.assertIn(term, term_info["candidate_snippet"].lower())
                self.assertIn(term, term_info["historical_snippet"].lower())

    def test_correct_section_to_score_mapping(self):
        """Verify section-to-score mapping contains all 8 sections bounded in [0.0, 1.0]."""
        candidate = {
            "project_title": "Supply Chain Blockchain Smart Contracts",
            "abstract": "Decentralized tracking using Ethereum smart contracts.",
            "problem_statement": "Opacity in food logistics.",
            "objectives": "Audit logs verification.",
            "proposed_methodology": "Solidity on Ethereum.",
            "dataset": "RFID tracking logs.",
            "algorithm_model": "Consensus mechanism.",
            "technology_tools": "Solidity, Web3."
        }
        hist_proj = self.synthetic_corpus.iloc[1]
        explanation = self.explainer.explain_pair(candidate, hist_proj)

        scores = explanation["section_similarity_scores"]
        self.assertEqual(len(scores), 8)
        for sec in SECTION_COLUMNS.keys():
            self.assertIn(sec, scores)
            self.assertGreaterEqual(scores[sec], 0.0)
            self.assertLessEqual(scores[sec], 1.0)

    def test_contribution_sum_matches_ridge_prediction(self):
        """Verify that intercept + sum(contributions) reproduces model raw prediction within 1e-5."""
        candidate = {
            "project_title": "Deep Autonomous Quadrotor Flight in Cluttered Spaces",
            "abstract": "Reinforcement learning agent for obstacle avoidance using deep neural networks.",
            "problem_statement": "Indoor drone collisions in GPS-denied environments.",
            "objectives": "Real-time trajectory planning without human intervention.",
            "proposed_methodology": "PPO reinforcement learning simulated in Gazebo.",
            "dataset": "Point cloud recordings from lidar and depth sensors.",
            "algorithm_model": "PPO agent.",
            "technology_tools": "ROS, Python, PyTorch."
        }
        hist_proj = self.synthetic_corpus.iloc[2]
        explanation = self.explainer.explain_pair(candidate, hist_proj)

        intercept = explanation["intercept"]
        contributions = explanation["section_signed_contributions"]
        reconstructed = intercept + sum(contributions.values())
        raw_pred = explanation["raw_similarity_prediction"]

        self.assertAlmostEqual(
            reconstructed, raw_pred, places=4,
            msg=f"Reconstructed raw prediction {reconstructed} did not match model raw prediction {raw_pred}"
        )

    def test_empty_or_missing_sections(self):
        """Verify graceful degradation when candidate has missing or empty sections."""
        empty_candidate = {
            "project_title": "",
            "abstract": "",
            "problem_statement": "",
            "objectives": "",
            "proposed_methodology": "",
            "dataset": "",
            "algorithm_model": "",
            "technology_tools": ""
        }
        hist_proj = self.synthetic_corpus.iloc[0]
        explanation = self.explainer.explain_pair(empty_candidate, hist_proj)

        # All section scores should be 0.0
        for sec, score in explanation["section_similarity_scores"].items():
            self.assertEqual(score, 0.0)

        # All signed contributions should be 0.0
        for sec, contrib in explanation["section_signed_contributions"].items():
            self.assertEqual(contrib, 0.0)

        # Shared terms should be empty
        for sec, ev in explanation["section_evidence"].items():
            self.assertEqual(len(ev["shared_terms"]), 0)

        # Prediction should equal model intercept (clipped to 0.0)
        self.assertEqual(explanation["raw_similarity_prediction"], explanation["intercept"])
        self.assertEqual(explanation["predicted_similarity"], max(0.0, explanation["intercept"]))

    def test_no_self_matching(self):
        """Verify that when a proposal matches a corpus project, that project is excluded from top matches."""
        # Proposal with exact same ID and text as SYN-TEST-002
        self_candidate = self.synthetic_corpus.iloc[1].to_dict()

        report = self.explainer.explain_proposal(self_candidate, top_k=2, exclude_self=True)

        matched_pids = [m["historical_project_id"] for m in report["top_matches"]]
        self.assertNotIn(
            "SYN-TEST-002", matched_pids,
            "Self-match was NOT excluded when exclude_self=True!"
        )
        self.assertEqual(report["corpus_size_evaluated"], 2)

    def test_deterministic_ranking(self):
        """Verify that running explanation twice produces identical rankings and scores."""
        candidate = {
            "project_title": "Deep Neural Networks for Medical Scan Segmentation",
            "abstract": "Convolutional U-Net model for MRI lesion boundaries.",
            "problem_statement": "Manual lesion annotation is slow.",
            "objectives": "Automated boundary detection.",
            "proposed_methodology": "Dice loss on BraTS data.",
            "dataset": "BraTS 2020 dataset.",
            "algorithm_model": "U-Net.",
            "technology_tools": "PyTorch, CUDA."
        }

        report_1 = self.explainer.explain_proposal(candidate, top_k=3, exclude_self=True)
        report_2 = self.explainer.explain_proposal(candidate, top_k=3, exclude_self=True)

        pids_1 = [m["historical_project_id"] for m in report_1["top_matches"]]
        pids_2 = [m["historical_project_id"] for m in report_2["top_matches"]]
        self.assertEqual(pids_1, pids_2)

        sims_1 = [m["predicted_similarity"] for m in report_1["top_matches"]]
        sims_2 = [m["predicted_similarity"] for m in report_2["top_matches"]]
        self.assertEqual(sims_1, sims_2)

    def test_correct_json_output_from_scoring_cli(self):
        """Verify that score_novelty.py produces valid, well-structured JSON when called with --json_output."""
        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "score_novelty.py"),
            "--title", "Automated Brain Tumor Detection Using CNN",
            "--abstract", "Deep learning CNN for automated MRI classification.",
            "--json_output",
            "--top_k", "2"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI returned code {proc.returncode}: {proc.stderr}")

        # Parse JSON
        output_str = proc.stdout.strip()
        json_start = output_str.find("{")
        self.assertGreaterEqual(json_start, 0, f"No JSON start found in output: {output_str}")
        data = json.loads(output_str[json_start:])

        # Verify required keys
        expected_keys = [
            "candidate_proposal_title",
            "corpus_size_evaluated",
            "max_similarity_score",
            "overall_novelty_score_pct",
            "novelty_assessment_tier",
            "top_matches",
            "framework_disclaimers"
        ]
        for key in expected_keys:
            self.assertIn(key, data)

        self.assertEqual(len(data["top_matches"]), 2)
        top_match = data["top_matches"][0]
        self.assertIn("historical_project_id", top_match)
        self.assertIn("predicted_similarity", top_match)
        self.assertIn("reconstructed_raw_prediction", top_match)
        self.assertIn("ranked_sections", top_match)
        self.assertIn("section_evidence", top_match)

    def test_generic_keyword_filtering(self):
        """Verify that generic words such as 'using', 'based', 'proposed', 'project' are filtered out."""
        # 'using', 'based', 'project' should return False from is_informative_term
        self.assertFalse(is_informative_term("using"))
        self.assertFalse(is_informative_term("based"))
        self.assertFalse(is_informative_term("project"))
        self.assertFalse(is_informative_term("proposed"))
        self.assertFalse(is_informative_term("project proposes"))
        self.assertFalse(is_informative_term("using cnn"))

        # Candidate and historical both sharing generic terms
        cand = {
            "project_title": "Project Using Deep Learning Based Methods",
            "abstract": "We propose using a system based on algorithms to develop a solution.",
            "problem_statement": "Manual methods are used.",
            "objectives": "Study and evaluate the proposed system.",
            "proposed_methodology": "Implement using approach.",
            "dataset": "Using dataset.",
            "algorithm_model": "Based on model.",
            "technology_tools": "Using tools."
        }
        hist = self.synthetic_corpus.iloc[0].to_dict()
        exp = self.explainer.explain_pair(cand, hist)
        for sec, ev in exp["section_evidence"].items():
            for term_info in ev["shared_terms"]:
                term = term_info["term"].lower()
                self.assertNotIn(term, GENERIC_LOW_INFO_TERMS)
                self.assertTrue(is_informative_term(term))

    def test_technical_terms_preserved(self):
        """Verify that legitimate technical terms like U-Net, BraTS, PyTorch, CNN, Solidity are preserved."""
        for tech_term in ["u-net", "brats", "pytorch", "cnn", "solidity", "ethereum", "gazebo", "ros"]:
            self.assertTrue(is_informative_term(tech_term), f"Technical term '{tech_term}' was wrongly rejected!")

        candidate = {
            "project_title": "U-Net CNN BraTS Segmentation",
            "abstract": "PyTorch deep learning CNN for MRI scans.",
            "problem_statement": "BraTS lesion segmentation is challenging.",
            "objectives": "Train U-Net in PyTorch.",
            "proposed_methodology": "3D U-Net convolutional network.",
            "dataset": "BraTS benchmark dataset.",
            "algorithm_model": "3D U-Net CNN.",
            "technology_tools": "PyTorch, CUDA, Python."
        }
        hist = self.synthetic_corpus.iloc[0].to_dict()
        exp = self.explainer.explain_pair(candidate, hist)
        # Check that informative technical terms were extracted
        found_terms = []
        for sec, ev in exp["section_evidence"].items():
            for t in ev["shared_terms"]:
                found_terms.append(t["term"].lower())
        self.assertTrue(any("brats" in t or "u-net" in t or "pytorch" in t or "cnn" in t for t in found_terms),
                        f"Expected technical terms in {found_terms}")

    def test_snippet_word_boundaries(self):
        """Verify that extracted snippets never start or end mid-word."""
        text = "Counterfeiting and opacity in organic food logistics cause substantial economic losses."
        # Search for 'food logistics' with small window to force truncation at both ends
        snip = extract_snippet(text, "food logistics", window=10)
        # Must not start or end mid-word
        body = snip.strip(".").strip()
        # Ensure the body starts at the beginning of a word in text
        match_idx = text.find(body)
        self.assertGreaterEqual(match_idx, 0)
        # Preceding char in text should be start of text (0) or whitespace
        if match_idx > 0:
            self.assertTrue(text[match_idx - 1].isspace(), f"Snippet body '{body}' starts mid-word in '{text}'!")
        # Following char should be end of text or whitespace
        end_idx = match_idx + len(body)
        if end_idx < len(text):
            self.assertTrue(text[end_idx].isspace(), f"Snippet body '{body}' ends mid-word in '{text}'!")

    def test_empty_evidence_with_explanatory_message(self):
        """Verify that when no informative shared terms exist, an empty list is returned with an explanatory message."""
        cand_disjoint = {
            "project_title": "Quantum Computing Cryptographic Key Distribution",
            "abstract": "Entangled photons protocol for quantum key distribution against eavesdroppers.",
            "problem_statement": "Post-quantum security vulnerabilities in RSA.",
            "objectives": "Generate polarization-entangled pairs.",
            "proposed_methodology": "BB84 protocol simulation in Qiskit.",
            "dataset": "Quantum state tomography telemetry.",
            "algorithm_model": "BB84 key exchange.",
            "technology_tools": "Qiskit, Python, IBM Quantum Experience."
        }
        # Compare against SYN-TEST-001 (Healthcare / MRI)
        hist = self.synthetic_corpus.iloc[0].to_dict()
        exp = self.explainer.explain_pair(cand_disjoint, hist)
        # Dataset section should have no shared terms
        ev_dataset = exp["section_evidence"]["dataset"]
        self.assertEqual(len(ev_dataset["shared_terms"]), 0)
        self.assertIn("message", ev_dataset)
        self.assertEqual(ev_dataset["message"], "No informative shared terms identified for this section.")

    def test_source_text_verification(self):
        """Verify that every extracted snippet body is an exact substring of the original source text."""
        candidate = {
            "project_title": "Automated Brain Tumor Segmentation with Convolutional Networks",
            "abstract": "This study applies convolutional neural network methods to MRI brain scans for tumor segmentation.",
            "problem_statement": "Brain lesions are difficult to segment manually.",
            "objectives": "Automate tumor segmentation using neural architectures.",
            "proposed_methodology": "Use 3D U-Net convolutional layers with Dice loss optimization.",
            "dataset": "BraTS benchmark brain tumor magnetic resonance dataset.",
            "algorithm_model": "3D U-Net Convolutional Neural Network.",
            "technology_tools": "Python, PyTorch, CUDA, SimpleITK."
        }
        hist = self.synthetic_corpus.iloc[0].to_dict()
        exp = self.explainer.explain_pair(candidate, hist)

        for sec, ev in exp["section_evidence"].items():
            cand_text = candidate[SECTION_COLUMNS[sec]]
            hist_text = hist[SECTION_COLUMNS[sec]]
            for t_info in ev["shared_terms"]:
                term = t_info["term"]
                c_snip = t_info["candidate_snippet"]
                h_snip = t_info["historical_snippet"]
                # Must satisfy verify_snippet
                self.assertTrue(verify_snippet(c_snip, cand_text, term),
                                f"Candidate snippet '{c_snip}' failed verification against source text '{cand_text}'!")
                self.assertTrue(verify_snippet(h_snip, hist_text, term),
                                f"Historical snippet '{h_snip}' failed verification against source text '{hist_text}'!")

    def test_score_reconstruction_and_clipping(self):
        """Verify that intercept + signed contributions reconstructs raw Ridge prediction, and raw is kept distinct from clipped."""
        candidate = {
            "project_title": "Random Unrelated Project",
            "abstract": "Completely unrelated content with zero overlap.",
            "problem_statement": "None.",
            "objectives": "None.",
            "proposed_methodology": "None.",
            "dataset": "None.",
            "algorithm_model": "None.",
            "technology_tools": "None."
        }
        hist = self.synthetic_corpus.iloc[0].to_dict()
        exp = self.explainer.explain_pair(candidate, hist)

        # Reconstructed raw prediction must equal intercept + sum of contributions
        self.assertAlmostEqual(exp["reconstructed_raw_prediction"], exp["raw_similarity_prediction"], places=4)
        # Predicted similarity must be clipped to [0.0, 4.0]
        self.assertGreaterEqual(exp["predicted_similarity"], 0.0)
        self.assertLessEqual(exp["predicted_similarity"], 4.0)
        self.assertEqual(exp["predicted_similarity"], exp["clipped_prediction"])

    def test_self_exclusion_comprehensive(self):
        """Verify self-exclusion by ID, by exact title, and by identical text."""
        # 1. By ID
        cand_by_id = {"project_id": "SYN-TEST-001", "project_title": "Different Title"}
        rep_1 = self.explainer.explain_proposal(cand_by_id, top_k=2, exclude_self=True)
        matched_ids_1 = [m["historical_project_id"] for m in rep_1["top_matches"]]
        self.assertNotIn("SYN-TEST-001", matched_ids_1)

        # 2. By title
        cand_by_title = {
            "project_title": "Deep Learning for Medical Image Segmentation",
            "abstract": "Different abstract text here."
        }
        rep_2 = self.explainer.explain_proposal(cand_by_title, top_k=2, exclude_self=True)
        matched_ids_2 = [m["historical_project_id"] for m in rep_2["top_matches"]]
        self.assertNotIn("SYN-TEST-001", matched_ids_2)

    def test_novelty_wording_and_disclaimers(self):
        """Verify that score >= 75% uses 'High estimated novelty relative to the current database' and disclaimers are accurate."""
        tier = get_novelty_level(80.0)
        self.assertEqual(tier, "High estimated novelty relative to the current database")

        candidate = {
            "project_title": "Novel Quantum Algorithm",
            "abstract": "Quantum computing proposal."
        }
        rep = self.explainer.explain_proposal(candidate, top_k=1, exclude_self=True)
        disclaimers = rep["framework_disclaimers"]
        # Must clearly state corpus-relative and does not establish originality or prove plagiarism
        self.assertIn("relative to the current database", disclaimers["corpus_relativity"].lower())
        self.assertIn("does not prove plagiarism", disclaimers["corpus_relativity"].lower())
        self.assertIn("originality", disclaimers["corpus_relativity"].lower())
        # Model attribution must state non-causal
        top_match = rep["top_matches"][0]
        self.assertIn("not causal explanations", top_match["disclaimers"]["model_attribution"].lower())

if __name__ == "__main__":
    unittest.main()
