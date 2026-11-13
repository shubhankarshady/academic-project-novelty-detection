"""
Re-evaluates the Eight Controlled Test Cases using the Live Novelty Service across 4 scoring configurations:
1. Production TF-IDF (Ridge fusion)
2. Candidate Constrained TF-IDF (alpha=1.0)
3. Semantic-Only (Candidate weights on dense similarities, alpha=0.0)
4. Hybrid Fusion (alpha=0.50)

Outputs comparative summary table and saves detailed results to:
results/model_optimization/phase4_controlled_evaluation.json
"""

import sys
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.api.service import NoveltyService
from scripts.evaluate_semantic_feasibility import CONTROLLED_TEST_CASES


def run_evaluation():
    # Initialize service instances
    prod_tfidf_service = NoveltyService(scoring_mode="tfidf")
    hybrid_service = NoveltyService(scoring_mode="hybrid", hybrid_alpha=0.50)

    results = []

    for case in CONTROLLED_TEST_CASES:
        case_id = case["case_id"]
        case_name = case["case_name"]
        target_ref_id = case["target_ref_id"]
        proposal = case["proposal"]

        # 1. Production TF-IDF
        prod_report = prod_tfidf_service.score_proposal(
            proposal,
            top_k=600,
            scoring_mode="tfidf",
            exclude_self=False
        )
        prod_ids = [m["historical_project_id"] for m in prod_report["top_matches"]]
        prod_rank = prod_ids.index(target_ref_id) + 1 if target_ref_id in prod_ids else 600
        prod_target_match = [m for m in prod_report["top_matches"] if m["historical_project_id"] == target_ref_id][0]

        # 2. Candidate Constrained TF-IDF (Hybrid with alpha=1.0)
        cand_report = hybrid_service.score_proposal(
            proposal,
            top_k=600,
            scoring_mode="hybrid",
            hybrid_alpha=1.0,
            exclude_self=False
        )
        cand_ids = [m["historical_project_id"] for m in cand_report["top_matches"]]
        cand_rank = cand_ids.index(target_ref_id) + 1 if target_ref_id in cand_ids else 600
        cand_target_match = [m for m in cand_report["top_matches"] if m["historical_project_id"] == target_ref_id][0]

        # 3. Semantic-Only (Hybrid with alpha=0.0)
        sem_report = hybrid_service.score_proposal(
            proposal,
            top_k=600,
            scoring_mode="hybrid",
            hybrid_alpha=0.0,
            exclude_self=False
        )
        sem_ids = [m["historical_project_id"] for m in sem_report["top_matches"]]
        sem_rank = sem_ids.index(target_ref_id) + 1 if target_ref_id in sem_ids else 600
        sem_target_match = [m for m in sem_report["top_matches"] if m["historical_project_id"] == target_ref_id][0]

        # 4. Hybrid (alpha=0.50)
        hyb_report = hybrid_service.score_proposal(
            proposal,
            top_k=600,
            scoring_mode="hybrid",
            hybrid_alpha=0.50,
            exclude_self=False
        )
        hyb_ids = [m["historical_project_id"] for m in hyb_report["top_matches"]]
        hyb_rank = hyb_ids.index(target_ref_id) + 1 if target_ref_id in hyb_ids else 600
        hyb_target_match = [m for m in hyb_report["top_matches"] if m["historical_project_id"] == target_ref_id][0]

        case_summary = {
            "case_id": case_id,
            "case_name": case_name,
            "target_ref_id": target_ref_id,
            "expected_relationship": case.get("expected_relationship", "N/A"),
            "prod_ridge_tfidf": {
                "similarity": prod_target_match["predicted_similarity"],
                "novelty_pct": prod_target_match["novelty_score_pct"],
                "target_rank": prod_rank,
                "top_1_id": prod_ids[0],
                "top_1_sim": prod_report["top_matches"][0]["predicted_similarity"]
            },
            "cand_constrained_tfidf": {
                "similarity": cand_target_match["predicted_similarity"],
                "novelty_pct": cand_target_match["novelty_score_pct"],
                "target_rank": cand_rank,
                "top_1_id": cand_ids[0],
                "top_1_sim": cand_report["top_matches"][0]["predicted_similarity"]
            },
            "semantic_only": {
                "similarity": sem_target_match["predicted_similarity"],
                "novelty_pct": sem_target_match["novelty_score_pct"],
                "target_rank": sem_rank,
                "top_1_id": sem_ids[0],
                "top_1_sim": sem_report["top_matches"][0]["predicted_similarity"]
            },
            "hybrid_alpha_050": {
                "similarity": hyb_target_match["predicted_similarity"],
                "novelty_pct": hyb_target_match["novelty_score_pct"],
                "target_rank": hyb_rank,
                "top_1_id": hyb_ids[0],
                "top_1_sim": hyb_report["top_matches"][0]["predicted_similarity"],
                "top_5_retrieved": hyb_ids[:5]
            }
        }
        results.append(case_summary)

    # Save to JSON
    out_dir = ROOT_DIR / "results" / "model_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase4_controlled_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print formatted markdown table
    print("\n### Phase 4 Controlled Cases Comparative Evaluation (Live Service)\n")
    print("| Case ID | Case Description | Metric | Prod TF-IDF (Ridge) | Cand TF-IDF | Semantic Only | Hybrid (α=0.50) |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")

    for r in results:
        cid = r["case_id"].replace("CASE_", "").replace("_", " ")
        cname = r["case_name"][:35]
        
        sim_row = f"| **{cid}** | {cname} | **Similarity** | {r['prod_ridge_tfidf']['similarity']:.4f} | {r['cand_constrained_tfidf']['similarity']:.4f} | {r['semantic_only']['similarity']:.4f} | **{r['hybrid_alpha_050']['similarity']:.4f}** |"
        nov_row = f"| | | **Novelty %** | {r['prod_ridge_tfidf']['novelty_pct']:.1f}% | {r['cand_constrained_tfidf']['novelty_pct']:.1f}% | {r['semantic_only']['novelty_pct']:.1f}% | **{r['hybrid_alpha_050']['novelty_pct']:.1f}%** |"
        rnk_row = f"| | | **Target Rank** | #{r['prod_ridge_tfidf']['target_rank']} | #{r['cand_constrained_tfidf']['target_rank']} | #{r['semantic_only']['target_rank']} | **#{r['hybrid_alpha_050']['target_rank']}** |"
        top_row = f"| | | **Top-1 Match** | {r['prod_ridge_tfidf']['top_1_id']} | {r['cand_constrained_tfidf']['top_1_id']} | {r['semantic_only']['top_1_id']} | **{r['hybrid_alpha_050']['top_1_id']}** |"
        
        print(sim_row)
        print(nov_row)
        print(rnk_row)
        print(top_row)
        print("| --- | --- | --- | --- | --- | --- | --- |")

if __name__ == "__main__":
    run_evaluation()
