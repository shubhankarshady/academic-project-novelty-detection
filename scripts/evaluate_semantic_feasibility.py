#!/usr/bin/env python
"""
Phase 3 Semantic Feasibility Study & Controlled Paraphrase Evaluation.

Compares:
1. Lexical TF-IDF cosine similarity.
2. Dense transformer embedding cosine similarity (all-MiniLM-L6-v2).
3. Production Ridge Model on TF-IDF.
4. Candidate Constrained SLSQP Model on TF-IDF.
5. Semantic-Only scoring (Candidate weights on dense similarities).
6. Hybrid Lexical-Semantic Fusion at alpha in [0.25, 0.50, 0.75]:
   hybrid_similarity = alpha * tfidf_similarity + (1 - alpha) * dense_similarity
   predicted_similarity = sum(w_i * hybrid_similarity_i)

Evaluates on 8 documented controlled cases against 600 reference projects.
Saves results to results/model_optimization/semantic_feasibility_study.json.
"""

import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.section_tfidf import SectionTFIDFExtractor, FEATURE_NAMES, SECTION_COLUMNS
from src.features.dense_embeddings import SectionDenseEmbeddingExtractor
from src.models.fusion_model import ConstrainedSectionFusion
from src.utils.metrics import calculate_novelty_score, get_novelty_level


# ---------------------------------------------------------------------------
# 1. Define the 8 Controlled Test Cases
# ---------------------------------------------------------------------------
CONTROLLED_TEST_CASES = [
    {
        "case_id": "CASE_A_EXACT_COPY",
        "case_name": "Exact Copy of Reference Proposal",
        "target_ref_id": "SYN-0001",
        "expected_relationship": "Duplicate",
        "provisional_expected_similarity": 4.0,
        "provisional_expected_novelty": 0.0,
        "description": "Verbatim copy of SYN-0001 across all 8 text sections.",
        "proposal": {
            "project_id": "TEST-CASE-A",
            "domain": "Artificial Intelligence & Machine Learning",
            "project_title": "Prototype Crop Disease Detection Using CNN",
            "abstract": "This project proposes a ug mini project solution for crop disease detection in the artificial intelligence & machine learning domain. The system uses CNN to analyze plantvillage and generate useful predictions or classifications. The proposed solution focuses on improving practical decision support, reducing manual effort, and providing measurable performance using standard evaluation metrics.",
            "problem_statement": "Existing approaches to crop disease detection can require substantial manual analysis and may provide inconsistent results when data volume or operational conditions increase. The project addresses the need for an automated and reliable approach that can process relevant data and support timely decisions.",
            "objectives": "1) Collect and preprocess relevant data; 2) develop a model for crop disease detection; 3) evaluate the model using suitable performance measures; 4) develop a usable prototype or experimental framework; 5) analyze limitations and possible improvements.",
            "proposed_methodology": "Implement a lightweight detection or prediction pipeline and compare two baseline approaches using a controlled experimental dataset.",
            "dataset": "PlantVillage",
            "algorithm_model": "CNN",
            "technology_tools": "Python, XGBoost"
        }
    },
    {
        "case_id": "CASE_B_LIGHT_PARAPHRASE",
        "case_name": "Light Paraphrase (Grammatical Edits & Local Synonyms)",
        "target_ref_id": "SYN-0001",
        "expected_relationship": "Near-Duplicate / Very High Overlap",
        "provisional_expected_similarity": 3.4,
        "provisional_expected_novelty": 15.0,
        "description": "Minor wording changes, replacing 'Prototype' with 'A System for', 'mini project solution' with 'undergraduate framework', and adding minor grammatical modifiers while preserving exact technical terms.",
        "proposal": {
            "project_id": "TEST-CASE-B",
            "domain": "Artificial Intelligence & Machine Learning",
            "project_title": "A System for Crop Disease Detection Using Convolutional Neural Networks",
            "abstract": "This project proposes an undergraduate framework for crop disease identification in the artificial intelligence & machine learning domain. The system applies CNN architectures to analyze PlantVillage images and generate accurate predictions and classifications. The proposed methodology emphasizes practical agricultural decision support, reducing manual effort, and providing measurable performance using standard evaluation metrics.",
            "problem_statement": "Current techniques for crop disease identification demand substantial manual inspection and deliver inconsistent results under varied operational conditions. This work addresses the urgent requirement for an automated and reliable approach that can process agricultural imagery and support timely decisions.",
            "objectives": "1) Acquire and preprocess leaf image data; 2) develop a model for crop disease detection; 3) evaluate model performance using standard evaluation metrics; 4) develop an experimental prototype framework; 5) analyze operational limitations and prospective improvements.",
            "proposed_methodology": "Implement an efficient detection pipeline and evaluate two baseline neural architectures using a controlled benchmark dataset.",
            "dataset": "PlantVillage dataset",
            "algorithm_model": "Convolutional Neural Network (CNN)",
            "technology_tools": "Python, PyTorch, XGBoost"
        }
    },
    {
        "case_id": "CASE_C_STRONG_PARAPHRASE",
        "case_name": "Strong Paraphrase (Synonym Substitution & Reordered Syntax)",
        "target_ref_id": "SYN-0001",
        "expected_relationship": "Strong Conceptual Overlap / Paraphrased Duplicate",
        "provisional_expected_similarity": 3.0,
        "provisional_expected_novelty": 25.0,
        "description": "Completely alternative vocabulary and sentence structures expressing the exact same research task, dataset, and neural vision approach without shared n-grams.",
        "proposal": {
            "project_id": "TEST-CASE-C",
            "domain": "Artificial Intelligence & Machine Learning",
            "project_title": "Deep Vision Architecture for Automated Agricultural Pathology Recognition",
            "abstract": "We develop an automated deep visual recognition pipeline to diagnose botanical afflictions in farming ecosystems. Employing multi-layer convolutional feature representations over public foliar leaf collections, our diagnostic software generates actionable classification outputs. The framework streamlines agronomic surveillance, curtails human field inspection demands, and benchmarks predictive accuracy.",
            "problem_statement": "Agronomic phytopathology diagnosis typically depends on labor-intensive human visual scouting, which suffers from subjective discrepancies and delayed intervention when acreage expands. There exists an acute necessity for computational diagnostic tools capable of rapid leaf assessment.",
            "objectives": "1) Curate foliar image corpora; 2) construct a deep vision classifier for foliar pathogen identification; 3) validate diagnostic capability via standard statistical rubrics; 4) package a functional software artifact; 5) document performance bottlenecks.",
            "proposed_methodology": "Construct an end-to-end vision classification workflow and contrast dual convolutional baselines on standardized leaf imagery.",
            "dataset": "Public open-access tomato and foliar botanical image collection",
            "algorithm_model": "Multi-layer Deep Convolutional Feature Extractor",
            "technology_tools": "Keras, Scikit-learn, Anaconda"
        }
    },
    {
        "case_id": "CASE_D_SAME_DOMAIN_DIFF_PROBLEM",
        "case_name": "Same Domain (AI/ML), Completely Different Research Problem",
        "target_ref_id": "SYN-0001",
        "expected_relationship": "Unrelated / Minimal Overlap",
        "provisional_expected_similarity": 0.3,
        "provisional_expected_novelty": 92.5,
        "description": "Same AI/ML domain, but focuses on Graph Convolutional Networks for Financial Credit Card Fraud Detection.",
        "proposal": {
            "project_id": "TEST-CASE-D",
            "domain": "Artificial Intelligence & Machine Learning",
            "project_title": "Financial Transaction Fraud Detection Using Graph Convolutional Networks",
            "abstract": "This project develops an anomaly detection framework for identifying fraudulent credit card transactions using relational graph neural networks. By mapping credit accounts and transaction entities into bipartite graphs, the model detects subtle syndicated fraud rings in streaming financial ledgers.",
            "problem_statement": "High-velocity financial fraud methods rapidly evolve to bypass static rule-based threshold detectors, causing severe institutional fiscal losses and delayed incident mitigation.",
            "objectives": "1) Construct dynamic transaction entity graphs; 2) train inductive graph convolutional architectures; 3) optimize AUCPR on skewed fraud classes; 4) evaluate real-time inference latency.",
            "proposed_methodology": "Benchmark inductive spatial graph convolution versus relational temporal graph transformers under extreme class imbalance.",
            "dataset": "IEEE-CIS Fraud Detection Benchmark",
            "algorithm_model": "Relational Graph Convolutional Network (R-GCN)",
            "technology_tools": "PyTorch Geometric, DGL, NetworkX, Python"
        }
    },
    {
        "case_id": "CASE_E_DIFF_DOMAIN_COMMON_TECH",
        "case_name": "Different Domain, Shared Technical Vocabulary",
        "target_ref_id": "SYN-0001",
        "expected_relationship": "Superficial Vocabulary Overlap Only",
        "provisional_expected_similarity": 0.4,
        "provisional_expected_novelty": 90.0,
        "description": "Computer Networks & SDN domain using CNNs and Python for DDoS attack traffic packet mitigation.",
        "proposal": {
            "project_id": "TEST-CASE-E",
            "domain": "Computer Networks & SDN",
            "project_title": "Automated Distributed Denial of Service Detection in SDN Controllers Using 1D CNN",
            "abstract": "This project develops a high-throughput intrusion detection mechanism in software-defined network control planes. By converting raw OpenFlow packet telemetry into temporal spatial vectors, a 1D CNN classifies volumetric flooding attacks with sub-millisecond latency.",
            "problem_statement": "Software-defined networking controllers are highly susceptible to control channel exhaustion from coordinated volumetric DDoS floods, compromising campus network availability.",
            "objectives": "1) Capture OpenFlow switch flow-table statistics; 2) train 1D convolutional intrusion classifier; 3) deploy mitigation rules into Ryu controller; 4) evaluate line-rate throughput impact.",
            "proposed_methodology": "Simulate Mininet topologies, generate attack traffic with Scapy, and benchmark 1D CNN against Random Forest packet filters.",
            "dataset": "CIC-DDoS2019 Telemetry Benchmark",
            "algorithm_model": "1D Convolutional Neural Network (1D-CNN)",
            "technology_tools": "Python, Ryu SDN Controller, Mininet, Scapy, PyTorch"
        }
    },
    {
        "case_id": "CASE_F_SAME_DATASET_DIFF_OBJECTIVES",
        "case_name": "Same Dataset (PlantVillage), Different Research Objectives",
        "target_ref_id": "SYN-0001",
        "expected_relationship": "Dataset Overlap with Distinct Methodology & Task",
        "provisional_expected_similarity": 1.6,
        "provisional_expected_novelty": 60.0,
        "description": "Uses PlantVillage dataset, but the research task is self-supervised masked generative inpainting for leaf occlusion restoration, not disease diagnosis.",
        "proposal": {
            "project_id": "TEST-CASE-F",
            "domain": "Computer Vision",
            "project_title": "Self-Supervised Generative Inpainting for Occluded Foliar Leaves",
            "abstract": "We introduce a masked autoencoder framework designed to reconstruct heavily occluded or damaged botanical leaf images. Trained self-supervised on unlabelled leaf imagery, the generative architecture inpaints missing venation structures and surface textures for synthetic data augmentation.",
            "problem_statement": "Robotic field cameras frequently capture partially occluded leaves blocked by soil, dew, or foliage, causing downstream vision models to fail due to incomplete boundary contours.",
            "objectives": "1) Implement masked patch token reconstruction; 2) optimize perceptual L1 reconstruction loss; 3) validate structural fidelity via SSIM and PSNR; 4) demonstrate data augmentation benefits.",
            "proposed_methodology": "Train a Vision Transformer masked autoencoder with random 75% patch masking and evaluate reconstruction fidelity against diffusion inpainters.",
            "dataset": "PlantVillage",
            "algorithm_model": "Masked Autoencoder (MAE / ViT-Base)",
            "technology_tools": "PyTorch, Torchvision, OpenCV, CUDA"
        }
    },
    {
        "case_id": "CASE_G_SAME_ALGO_DIFF_APPLICATION",
        "case_name": "Same Algorithm (CNN), Completely Different Application Domain",
        "target_ref_id": "SYN-0001",
        "expected_relationship": "Algorithmic Tool Overlap with Different Application",
        "provisional_expected_similarity": 0.8,
        "provisional_expected_novelty": 80.0,
        "description": "Uses CNN in Healthcare & Medical Informatics for Brain Tumor Segmentation in MRI Scans.",
        "proposal": {
            "project_id": "TEST-CASE-G",
            "domain": "Healthcare & Medical Informatics",
            "project_title": "Automated Brain Tumor Segmentation in Multi-Parametric MRI Using U-Net CNN",
            "abstract": "This project presents an automated deep learning pipeline for glioblastoma segmentation in multi-parametric magnetic resonance imaging. The system uses a 2D U-Net CNN architecture to delineate active tumor core and edema regions to assist neurosurgical planning.",
            "problem_statement": "Manual slice-by-slice contouring of multi-parametric brain MRI volumes is tedious, observer-dependent, and introduces surgical planning delays for oncology patients.",
            "objectives": "1) Preprocess skull-stripped multi-modal MRI scans; 2) train 2D convolutional encoder-decoder network; 3) compute Dice similarity coefficient against expert radiologist masks; 4) deploy clinician inspection tool.",
            "proposed_methodology": "Implement a 2D U-Net CNN with skip connections, optimize generalized Dice loss, and validate using 5-fold cross-validation.",
            "dataset": "BraTS 2021 Multi-Modal Brain Tumor Dataset",
            "algorithm_model": "CNN (U-Net Encoder-Decoder)",
            "technology_tools": "Python, SimpleITK, MONAI, PyTorch, Nibabel"
        }
    },
    {
        "case_id": "CASE_H_SUPERFICIAL_WORDING_DIFF_OBJECTIVES",
        "case_name": "Superficially Similar Wording, Substantially Different Research",
        "target_ref_id": "SYN-0001",
        "expected_relationship": "Superficial Jargon Overlap / Different Academic Field",
        "provisional_expected_similarity": 0.5,
        "provisional_expected_novelty": 87.5,
        "description": "Contains terms 'crop disease detection', 'automated approach', but the work is a socio-legal policy and governance study with no ML model, code, or technical implementation.",
        "proposal": {
            "project_id": "TEST-CASE-H",
            "domain": "Education Technology",
            "project_title": "Socio-Legal Governance and Policy Frameworks for Agricultural Technology Adoption",
            "abstract": "This research analyzes regulatory policy barriers affecting agricultural technology adoption for automated crop disease detection across rural smallholder farming communities. The study evaluates institutional governance frameworks, data privacy risks, and farmer compliance through socio-economic surveys.",
            "problem_statement": "Despite technological advancements in automated approaches to crop disease detection, adoption rates among smallholder farmers remain depressed due to governance deficits, ambiguous data rights, and lack of localized policy support.",
            "objectives": "1) Survey 300 farming households regarding technology readiness; 2) evaluate legal liability frameworks for automated diagnostic errors; 3) develop a policy roadmap for cooperative extension programs; 4) analyze rural institutional limitations.",
            "proposed_methodology": "Qualitative empirical policy research combining structured stakeholder interviews, econometric regression of survey responses, and comparative statutory legal analysis.",
            "dataset": "Primary Fieldwork Survey of 300 Smallholder Farmers",
            "algorithm_model": "Qualitative Thematic Analysis & Multivariate Probit Econometric Modeling",
            "technology_tools": "R, SPSS, NVivo, Qualtrics"
        }
    }
]


def run_evaluation():
    print("=== Phase 3: Semantic Embedding Feasibility & Paraphrase Evaluation ===")

    # 1. Load artifacts
    tfidf_path = ROOT_DIR / "models" / "section_tfidf_extractor.joblib"
    dense_path = ROOT_DIR / "models" / "section_dense_extractor.joblib"
    candidate_path = ROOT_DIR / "models" / "candidate_constrained_fusion_model.joblib"
    ridge_path = ROOT_DIR / "models" / "ridge_fusion_model.joblib"

    print("Loading models...")
    tfidf_extractor: SectionTFIDFExtractor = joblib.load(tfidf_path)
    dense_extractor: SectionDenseEmbeddingExtractor = joblib.load(dense_path)
    candidate_model: ConstrainedSectionFusion = ConstrainedSectionFusion.load(candidate_path)
    ridge_artifact = joblib.load(ridge_path)
    ridge_model = ridge_artifact["model"]

    # Candidate weights (w_i >= 0, sum to 4.0)
    w_candidate = candidate_model.coef_  # Array of length 8
    print(f"Loaded candidate weights: {w_candidate}")
    print(f"Loaded production Ridge coefs: {ridge_model.coef_}")

    # Lazy-load sentence-transformer model in dense extractor
    print("Initializing SentenceTransformer for candidate encoding...")
    dense_extractor._lazy_load_model()
    print("Encoder ready.")

    projects_df = tfidf_extractor.projects_df
    ref_proj_ids = list(projects_df["project_id"])

    results = []

    print("\n--- Evaluating 8 Controlled Test Cases ---")
    for case in CONTROLLED_TEST_CASES:
        cid = case["case_id"]
        cname = case["case_name"]
        target_id = case["target_ref_id"]
        prop = case["proposal"]

        print(f"\nEvaluating: [{cid}] {cname}")

        # Compute TF-IDF candidate similarity against all 600 projects
        sim_tfidf_df = tfidf_extractor.compute_candidate_similarity(prop)
        X_tfidf = sim_tfidf_df[FEATURE_NAMES].values  # Shape (600, 8)

        # Compute Dense candidate similarity against all 600 projects
        sim_dense_df = dense_extractor.compute_candidate_similarity(prop)
        dense_feat_names = [f"{sec}_dense_similarity" for sec in SECTION_COLUMNS.keys()]
        X_dense = sim_dense_df[dense_feat_names].values  # Shape (600, 8)

        # Extract similarities specifically against target project (SYN-0001)
        target_idx = ref_proj_ids.index(target_id)
        s_target_tfidf = X_tfidf[target_idx]
        s_target_dense = X_dense[target_idx]

        # Compute overall similarity predictions across all 600 projects for each method:
        # Method 1: Production Ridge on TF-IDF
        preds_ridge = np.clip(ridge_model.predict(X_tfidf), 0.0, 4.0)

        # Method 2: Candidate Constrained on TF-IDF
        preds_cand_tfidf = np.clip(X_tfidf @ w_candidate, 0.0, 4.0)

        # Method 3: Semantic-Only (Candidate weights applied to Dense similarities)
        preds_semantic = np.clip(X_dense @ w_candidate, 0.0, 4.0)

        # Method 4a: Hybrid alpha = 0.25 (75% semantic, 25% lexical)
        X_hybrid_25 = 0.25 * X_tfidf + 0.75 * X_dense
        preds_hybrid_25 = np.clip(X_hybrid_25 @ w_candidate, 0.0, 4.0)

        # Method 4b: Hybrid alpha = 0.50 (50% semantic, 50% lexical)
        X_hybrid_50 = 0.50 * X_tfidf + 0.50 * X_dense
        preds_hybrid_50 = np.clip(X_hybrid_50 @ w_candidate, 0.0, 4.0)

        # Method 4c: Hybrid alpha = 0.75 (25% semantic, 75% lexical)
        X_hybrid_75 = 0.75 * X_tfidf + 0.25 * X_dense
        preds_hybrid_75 = np.clip(X_hybrid_75 @ w_candidate, 0.0, 4.0)

        methods = {
            "prod_ridge_tfidf": preds_ridge,
            "cand_constrained_tfidf": preds_cand_tfidf,
            "semantic_only": preds_semantic,
            "hybrid_alpha_025": preds_hybrid_25,
            "hybrid_alpha_050": preds_hybrid_50,
            "hybrid_alpha_075": preds_hybrid_75
        }

        # Determine ranking of target_id (SYN-0001) and top-1 retrieved project for each method
        method_eval = {}
        for mname, mpreds in methods.items():
            ranked_indices = np.argsort(-mpreds)
            target_rank = int(np.where(ranked_indices == target_idx)[0][0]) + 1
            top1_proj_id = ref_proj_ids[ranked_indices[0]]
            top1_sim = float(mpreds[ranked_indices[0]])
            top5_proj_ids = [ref_proj_ids[i] for i in ranked_indices[:5]]

            target_sim = float(mpreds[target_idx])
            target_novelty = float(calculate_novelty_score(target_sim))

            method_eval[mname] = {
                "target_sim": round(target_sim, 4),
                "target_novelty_pct": round(target_novelty, 2),
                "target_tier": get_novelty_level(target_novelty),
                "target_rank": target_rank,
                "in_top_1": (target_rank == 1),
                "in_top_3": (target_rank <= 3),
                "in_top_5": (target_rank <= 5),
                "top_1_project_id": top1_proj_id,
                "top_1_similarity": round(top1_sim, 4),
                "top_5_retrieved": top5_proj_ids
            }

        # Build comparison summary for this case
        case_summary = {
            "case_id": cid,
            "case_name": cname,
            "target_ref_id": target_id,
            "expected_relationship": case["expected_relationship"],
            "section_similarities_against_target": {
                sec: {
                    "tfidf": round(float(s_target_tfidf[i]), 4),
                    "dense": round(float(s_target_dense[i]), 4),
                    "diff_dense_minus_tfidf": round(float(s_target_dense[i] - s_target_tfidf[i]), 4)
                }
                for i, sec in enumerate(SECTION_COLUMNS.keys())
            },
            "method_evaluations": method_eval
        }
        results.append(case_summary)

        # Print quick console report
        print(f"  Target [{target_id}] Similarities:")
        print(f"    TF-IDF Sections: title={s_target_tfidf[0]:.2f}, abstract={s_target_tfidf[1]:.2f}, dataset={s_target_tfidf[5]:.2f}, algo={s_target_tfidf[6]:.2f}")
        print(f"    Dense  Sections: title={s_target_dense[0]:.2f}, abstract={s_target_dense[1]:.2f}, dataset={s_target_dense[5]:.2f}, algo={s_target_dense[6]:.2f}")
        print(f"  Target Rank: TF-IDF Candidate=#{method_eval['cand_constrained_tfidf']['target_rank']} (sim={method_eval['cand_constrained_tfidf']['target_sim']:.2f}, nov={method_eval['cand_constrained_tfidf']['target_novelty_pct']:.1f}%)")
        print(f"  Target Rank: Semantic-Only   =#{method_eval['semantic_only']['target_rank']} (sim={method_eval['semantic_only']['target_sim']:.2f}, nov={method_eval['semantic_only']['target_novelty_pct']:.1f}%)")
        print(f"  Target Rank: Hybrid (a=0.50) =#{method_eval['hybrid_alpha_050']['target_rank']} (sim={method_eval['hybrid_alpha_050']['target_sim']:.2f}, nov={method_eval['hybrid_alpha_050']['target_novelty_pct']:.1f}%)")

    # -----------------------------------------------------------------------
    # Summary of Retrieval Success Rates
    # -----------------------------------------------------------------------
    paraphrase_cases = ["CASE_A_EXACT_COPY", "CASE_B_LIGHT_PARAPHRASE", "CASE_C_STRONG_PARAPHRASE"]
    distractor_cases = ["CASE_D_SAME_DOMAIN_DIFF_PROBLEM", "CASE_E_DIFF_DOMAIN_COMMON_TECH",
                        "CASE_F_SAME_DATASET_DIFF_OBJECTIVES", "CASE_G_SAME_ALGO_DIFF_APPLICATION",
                        "CASE_H_SUPERFICIAL_WORDING_DIFF_OBJECTIVES"]

    summary_stats = {}
    for mname in ["prod_ridge_tfidf", "cand_constrained_tfidf", "semantic_only", "hybrid_alpha_025", "hybrid_alpha_050", "hybrid_alpha_075"]:
        p_top1 = sum(1 for r in results if r["case_id"] in paraphrase_cases and r["method_evaluations"][mname]["in_top_1"])
        p_top3 = sum(1 for r in results if r["case_id"] in paraphrase_cases and r["method_evaluations"][mname]["in_top_3"])
        p_top5 = sum(1 for r in results if r["case_id"] in paraphrase_cases and r["method_evaluations"][mname]["in_top_5"])

        d_filtered = sum(1 for r in results if r["case_id"] in distractor_cases and not r["method_evaluations"][mname]["in_top_1"])

        summary_stats[mname] = {
            "paraphrase_cases_top1": f"{p_top1}/{len(paraphrase_cases)}",
            "paraphrase_cases_top3": f"{p_top3}/{len(paraphrase_cases)}",
            "paraphrase_cases_top5": f"{p_top5}/{len(paraphrase_cases)}",
            "distractor_filtered_top1": f"{d_filtered}/{len(distractor_cases)}"
        }

    output_payload = {
        "study_objective": "Evaluate dense semantic embeddings vs TF-IDF for paraphrase and distractors",
        "reference_corpus_size": len(ref_proj_ids),
        "embedding_model": "all-MiniLM-L6-v2",
        "controlled_case_evaluations": results,
        "retrieval_summary_stats": summary_stats
    }

    output_file = ROOT_DIR / "results" / "model_optimization" / "semantic_feasibility_study.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved detailed evaluation study to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
