"""
Scientific Reliability and Model Behavior Audit Suite.
Tests the 7 controlled cases and XAI attribution integrity without modifying any production code or models.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.api.service import get_novelty_service
from src.explainability.explainer import is_informative_term, verify_snippet
from src.utils.metrics import calculate_novelty_score, get_novelty_level

ROOT_DIR = Path(__file__).resolve().parent.parent

class TestScientificReliabilityAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = get_novelty_service()
        cls.explainer = cls.service.explainer
        cls.projects_df = cls.explainer.projects_df
        
        # Base historical project SYN-0001
        cls.syn_0001 = cls.projects_df[cls.projects_df["project_id"] == "SYN-0001"].iloc[0].to_dict()

    def test_mathematical_novelty_formula_consistency(self):
        """Verify the novelty formula and rounding across the full 0-4 range."""
        # Test example from user: reported max_sim = 1.137 -> novelty = 71.6%
        sim_val = 1.137
        calc_novelty = calculate_novelty_score(sim_val)
        self.assertAlmostEqual(calc_novelty, 71.575, places=3)
        self.assertEqual(round(calc_novelty, 1), 71.6)
        
        # Test extreme boundaries
        self.assertEqual(calculate_novelty_score(0.0), 100.0)
        self.assertEqual(calculate_novelty_score(4.0), 0.0)
        self.assertEqual(calculate_novelty_score(-1.0), 100.0)  # Clipped to 0
        self.assertEqual(calculate_novelty_score(5.0), 0.0)    # Clipped to 4
        
        # Test tier threshold consistency
        self.assertIn("High", get_novelty_level(75.0))
        self.assertIn("Moderate", get_novelty_level(74.9))
        self.assertIn("Moderate", get_novelty_level(50.0))
        self.assertIn("Low", get_novelty_level(49.9))
        self.assertIn("Very Low", get_novelty_level(24.9))

    def test_theoretical_maximum_prediction_capacity(self):
        """
        Audit the theoretical range of the Ridge model when all features = 1.0 (exact duplicate).
        """
        intercept = float(self.explainer.intercept_)
        coefs = self.explainer.coef_
        max_possible_raw = intercept + np.sum(coefs)
        
        print("\n--- Model Theoretical Max Prediction ---")
        print(f"Intercept: {intercept:.4f}")
        print(f"Sum of coefficients: {np.sum(coefs):.4f}")
        print(f"Max raw prediction for identical project: {max_possible_raw:.4f}")
        
        min_novelty_for_exact_copy = calculate_novelty_score(max_possible_raw)
        print(f"Novelty % for 100% identical copy: {min_novelty_for_exact_copy:.2f}%")
        print(f"Category for 100% identical copy: {get_novelty_level(min_novelty_for_exact_copy)}")
        
        # Verify that max_possible_raw is capped at ~2.58 and can NEVER reach 4.0
        self.assertLess(max_possible_raw, 3.0)

    def test_controlled_case_1_exact_copy(self):
        """Case 1: Exact copy of historical project SYN-0001."""
        candidate = dict(self.syn_0001)
        
        # 1a. Test-only evaluation with exclude_self=False to measure identical match behavior
        # Discovers non-monotonicity: negative Ridge coefficients on technology (-0.1456), problem (-0.1259)
        # allow near-peers with differing tools to rank #1-#5 (sim ~2.65), while exact identical match ranks #6 (sim=2.5768)
        report_no_exclude = self.explainer.explain_proposal(candidate, top_k=10, exclude_self=False)
        self_match = [m for m in report_no_exclude["top_matches"] if m["historical_project_id"] == "SYN-0001"]
        self.assertTrue(len(self_match) > 0, "SYN-0001 must appear in top matches when self-exclusion is disabled")
        self_match_item = self_match[0]
        
        self.assertAlmostEqual(self_match_item["section_similarity_scores"]["dataset"], 1.0, places=3)
        self.assertAlmostEqual(self_match_item["section_similarity_scores"]["abstract"], 1.0, places=3)
        self.assertAlmostEqual(self_match_item["raw_similarity_prediction"], 2.5768, places=3)
        print(f"\n[Case 1a] Exact copy self-match sim: {self_match_item['predicted_similarity']} novelty={self_match_item['novelty_score_pct']}% (Ranked #{[m['historical_project_id'] for m in report_no_exclude['top_matches']].index('SYN-0001') + 1})")
        
        # 1b. Production evaluation with exclude_self=True
        report_prod = self.explainer.explain_proposal(candidate, top_k=5, exclude_self=True)
        top_match_prod = report_prod["top_matches"][0]
        
        # Must NOT match SYN-0001
        self.assertNotEqual(top_match_prod["historical_project_id"], "SYN-0001")
        print(f"[Case 1b] Exact copy production top match: {top_match_prod['historical_project_id']} "
              f"sim={top_match_prod['predicted_similarity']} novelty={report_prod['overall_novelty_score_pct']}%")

    def test_controlled_case_2_minor_wording_edits(self):
        """Case 2: Copy with minor wording edits."""
        candidate = dict(self.syn_0001)
        candidate["project_title"] = "A Prototype Crop Disease Detection Framework Using Deep CNN"
        candidate["abstract"] = candidate["abstract"].replace("lightweight", "efficient").replace("analyzes", "evaluates")
        candidate["proposed_methodology"] = "We implement an efficient detection pipeline and benchmark two architectures."
        
        report = self.explainer.explain_proposal(candidate, top_k=5, exclude_self=True)
        top_match = report["top_matches"][0]
        
        print(f"\n[Case 2] Minor edits top match: {top_match['historical_project_id']} "
              f"title='{top_match['historical_project_title']}' sim={top_match['predicted_similarity']} "
              f"novelty={report['overall_novelty_score_pct']}%")

    def test_controlled_case_3_paraphrased_version(self):
        """Case 3: Paraphrased version of SYN-0001 (same concepts, completely different words)."""
        candidate = {
            "project_title": "Automated Foliage Pathogen Recognition with Visual Networks",
            "domain": "Artificial Intelligence & Machine Learning",
            "abstract": "Agricultural yield suffers immense degradation from untreated botanical maladies. We develop an automated diagnostic mechanism leveraging hierarchical convolutional topologies to identify leaf infections across curated botanical photo repositories.",
            "problem_statement": "Manual agrarian inspections by horticulturists remain tedious and unscalable across expansive farming tracts.",
            "objectives": "Devise a high-precision computer vision pipeline to flag foliar blights and assist agronomic triage.",
            "proposed_methodology": "Train multi-layer visual feature extractors over photographic specimens of infected flora.",
            "dataset": "Standard open botanical leaves repository",
            "algorithm_model": "Deep Residual Convolutional Architecture",
            "technology_tools": "PyTorch, CUDA, NumPy, OpenCV"
        }
        
        report = self.explainer.explain_proposal(candidate, top_k=5, exclude_self=True)
        top_match = report["top_matches"][0]
        
        print(f"\n[Case 3] Paraphrased top match: {top_match['historical_project_id']} "
              f"sim={top_match['predicted_similarity']} novelty={report['overall_novelty_score_pct']}%")
        # In lexical TF-IDF, paraphrasing completely drops lexical overlap!
        self.assertGreater(report["overall_novelty_score_pct"], 90.0)

    def test_controlled_case_4_same_title_different_content(self):
        """Case 4: Same title as SYN-0001 but completely different technical content."""
        candidate = {
            "project_title": "Prototype Crop Disease Detection Using CNN",
            "domain": "Artificial Intelligence & Machine Learning",
            "abstract": "We explore drone-based hyperspectral sensing in commercial vineyards using Vision Transformers and LiDAR point clouds to measure soil moisture and canopy nitrogen deficits.",
            "problem_statement": "Canopy nitrogen and water deficits trigger yield loss before visual symptoms appear on leaves.",
            "objectives": "Mount hyperspectral cameras on autonomous UAVs and run real-time point-cloud semantic segmentation.",
            "proposed_methodology": "Deploy Swin Transformer with 3D point cloud convolutions over calibrated aerial flight tracks.",
            "dataset": "Custom Napa Valley Drone Hyperspectral Benchmark 2025",
            "algorithm_model": "Swin Transformer V2 and PointNet++",
            "technology_tools": "ROS2, PyTorch Geometric, Jetson Orin Nano"
        }
        
        report = self.explainer.explain_proposal(candidate, top_k=5, exclude_self=True)
        top_match = report["top_matches"][0]
        
        self.assertNotEqual(top_match["historical_project_id"], "SYN-0001")
        print(f"\n[Case 4] Same title different content top match: {top_match['historical_project_id']} "
              f"title='{top_match['historical_project_title']}' sim={top_match['predicted_similarity']} "
              f"novelty={report['overall_novelty_score_pct']}%")

    def test_controlled_case_5_unrelated_topic(self):
        """Case 5: Unrelated topic (Quantum Key Distribution)."""
        candidate = {
            "project_title": "Continuous-Variable Quantum Key Distribution for Low-Earth-Orbit Satellite Constellations",
            "domain": "Cybersecurity & Quantum Computing",
            "abstract": "We design an atmospheric beam-tracking and phase-noise compensation protocol for continuous-variable quantum key distribution between LEO satellites and optical ground stations.",
            "problem_statement": "Satellite-to-ground optical downlinks experience severe atmospheric turbulence, beam wander, and Doppler shifts that collapse quantum state coherence.",
            "objectives": "Achieve secret key generation rates exceeding 10 kbps through simulated Beaufort Sea optical turbulent atmospheric channels.",
            "proposed_methodology": "Simulate coherent optical homodyne detection with adaptive Kalman wavefront correction and LDPC reconciliation codes.",
            "dataset": "ESA Optical Ground Station Tenerife Channel Benchmark",
            "algorithm_model": "Multi-dimensional reconciliation LDPC codes with Kalman filter phase tracking",
            "technology_tools": "QKD-Sim, C++, GNU Radio, MATLAB"
        }
        
        report = self.explainer.explain_proposal(candidate, top_k=5, exclude_self=True)
        top_match = report["top_matches"][0]
        
        print(f"\n[Case 5] Unrelated topic top match: {top_match['historical_project_id']} "
              f"sim={top_match['predicted_similarity']} novelty={report['overall_novelty_score_pct']}%")
        self.assertGreater(report["overall_novelty_score_pct"], 80.0)

    def test_controlled_case_6_only_one_overlapping_section(self):
        """Case 6: Candidate with only ONE overlapping section (dataset = PlantVillage) but completely different research."""
        candidate = {
            "project_title": "Autonomous Underwater Acoustic Sensor Networks for Coral Bleaching Monitoring",
            "domain": "Internet of Things",
            "abstract": "We deploy submerged piezoceramic hydrophones to record low-frequency acoustic signatures in Great Barrier Reef marine protected areas.",
            "problem_statement": "Subsurface thermal shock degrades coral reef ecosystems without timely acoustic bio-indicator warnings.",
            "objectives": "Detect reef degradation patterns through passive acoustic telemetry over 50 km underwater mesh networks.",
            "proposed_methodology": "Deploy low-power LoRa-modulated acoustic modems and perform underwater time-of-flight synchronization.",
            "dataset": "PlantVillage",  # Overlapping dataset name
            "algorithm_model": "Acoustic OFDM and Viterbi decoding",
            "technology_tools": "NS-3, C++, FreeRTOS, STM32"
        }
        
        report = self.explainer.explain_proposal(candidate, top_k=5, exclude_self=True)
        top_match = report["top_matches"][0]
        
        print(f"\n[Case 6] Dataset-only overlap top match: {top_match['historical_project_id']} "
              f"title='{top_match['historical_project_title']}' "
              f"dataset_sim={top_match['section_similarity_scores']['dataset']} "
              f"pred_sim={top_match['predicted_similarity']} novelty={report['overall_novelty_score_pct']}%")
        # Because dataset weight is +2.2054, having dataset=PlantVillage raises similarity to ~2.13 and drops novelty to 46.87%
        self.assertAlmostEqual(top_match["section_similarity_scores"]["dataset"], 1.0, places=3)
        self.assertGreater(top_match["predicted_similarity"], 2.0)

    def test_controlled_case_7_generic_boilerplate(self):
        """Case 7: Generic academic boilerplate."""
        candidate = {
            "project_title": "Development of an Application for Data Analysis Using Algorithms",
            "domain": "Computer Science",
            "abstract": "This project aims to develop a system for processing data and evaluating results using machine learning techniques and standard methodologies.",
            "problem_statement": "Traditional methods require manual analysis and can be inefficient when data volumes grow in modern software applications.",
            "objectives": "1) Collect data; 2) analyze system; 3) implement model; 4) evaluate performance metrics; 5) document findings.",
            "proposed_methodology": "We implement a software pipeline using standard tools and compare baseline approaches on the chosen dataset.",
            "dataset": "Standard benchmark data",
            "algorithm_model": "Standard algorithm",
            "technology_tools": "Python, libraries"
        }
        
        report = self.explainer.explain_proposal(candidate, top_k=5, exclude_self=True)
        top_match = report["top_matches"][0]
        
        print(f"\n[Case 7] Generic boilerplate top match: {top_match['historical_project_id']} "
              f"sim={top_match['predicted_similarity']} novelty={report['overall_novelty_score_pct']}%")

    def test_top_match_and_xai_attribution_correctness(self):
        """Audit Section D: snippet accuracy, ID existence, and linear additive reconstruction."""
        candidate = dict(self.syn_0001)
        candidate["project_title"] = "Experimental Crop Disease Detection Using Deep CNN"
        
        report = self.explainer.explain_proposal(candidate, top_k=5, exclude_self=True)
        
        for match in report["top_matches"]:
            hist_id = match["historical_project_id"]
            
            # 1. Verify historical project ID exists in reference corpus
            self.assertIn(hist_id, self.projects_df["project_id"].values)
            
            # 2. Match titles and text correspond to the correct source record
            src_row = self.projects_df[self.projects_df["project_id"] == hist_id].iloc[0]
            self.assertEqual(match["historical_project_title"], src_row["project_title"])
            
            # 3. Additive Ridge reconstruction check: intercept + sum(contribs) == raw_prediction within 1e-4
            intercept = match["intercept"]
            sum_contribs = match["sum_of_contributions"]
            reconstructed = intercept + sum_contribs
            raw_pred = match["raw_similarity_prediction"]
            self.assertAlmostEqual(reconstructed, raw_pred, places=3)
            
            # 4. Verify snippet and term integrity
            for sec_name, ev in match["section_evidence"].items():
                col = self.explainer.section_cols[sec_name]
                cand_text = str(candidate.get(col, ""))
                hist_text = str(src_row.get(col, ""))
                
                for term_item in ev["shared_terms"]:
                    term = term_item["term"]
                    cand_snip = term_item["candidate_snippet"]
                    hist_snip = term_item["historical_snippet"]
                    
                    # Term actually occurs in both source texts
                    self.assertIn(term.lower(), cand_text.lower())
                    self.assertIn(term.lower(), hist_text.lower())
                    
                    # Snippets are exact substrings
                    self.assertTrue(verify_snippet(cand_snip, cand_text, term))
                    self.assertTrue(verify_snippet(hist_snip, hist_text, term))

if __name__ == "__main__":
    unittest.main()
