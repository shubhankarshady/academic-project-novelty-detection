#!/usr/bin/env python
"""
CLI Tool to score the Novelty of a candidate academic project proposal
with Explainable AI (XAI) transparent attribution and evidence extraction.

Supports:
- Preferred interpretable candidate: Ridge Regression on 8 section TF-IDF similarities.
- Backward-compatible Hybrid Fusion (TF-IDF + Dense Transformer Embeddings).
- Full section-level ranking, keyword evidence snippets, and additive score reconstruction.
- Output in human-readable inspection report or machine-readable JSON format.
"""

import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.section_tfidf import SectionTFIDFExtractor, FEATURE_NAMES as TFIDF_FEATURE_NAMES, SECTION_COLUMNS
from src.features.dense_embeddings import SectionDenseEmbeddingExtractor, DENSE_FEATURE_NAMES
from src.models.hybrid_fusion import HybridSectionFusionModel, HYBRID_FEATURE_NAMES
from src.utils.metrics import calculate_novelty_score, get_novelty_level
from src.explainability.explainer import NoveltyExplainer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Score Novelty of a candidate project proposal with Explainable AI (XAI) evidence."
    )
    parser.add_argument("--json", type=str, help="Path to JSON file containing project proposal details.")
    parser.add_argument("--title", type=str, help="Project title")
    parser.add_argument("--abstract", type=str, help="Project abstract")
    parser.add_argument("--problem", type=str, help="Problem statement")
    parser.add_argument("--objectives", type=str, help="Project objectives")
    parser.add_argument("--methodology", type=str, help="Proposed methodology")
    parser.add_argument("--dataset", type=str, help="Dataset tools")
    parser.add_argument("--algorithm", type=str, help="Algorithm/model used")
    parser.add_argument("--technology", type=str, help="Technology stack/tools")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top similar prior projects to display")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["ridge_tfidf", "hybrid"],
        default="ridge_tfidf",
        help="Model architecture: 'ridge_tfidf' (interpretable candidate) or 'hybrid' (TF-IDF + Dense)."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        default=True,
        help="Include full section ranking, keyword evidence, and additive contribution breakdown."
    )
    parser.add_argument(
        "--json_output",
        action="store_true",
        help="Output results as structured JSON instead of human-readable text."
    )
    parser.add_argument(
        "--exclude_id",
        type=str,
        help="Project ID to exclude from historical comparison (prevent self-matching)."
    )
    return parser.parse_args()

def load_candidate_dict(args) -> dict:
    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            candidate = json.load(f)
    else:
        candidate = {
            "project_title": args.title or "",
            "abstract": args.abstract or "",
            "problem_statement": args.problem or "",
            "objectives": args.objectives or "",
            "proposed_methodology": args.methodology or "",
            "dataset": args.dataset or "",
            "algorithm_model": args.algorithm or "",
            "technology_tools": args.technology or ""
        }
    if args.exclude_id:
        candidate["project_id"] = args.exclude_id
    return candidate

def run_ridge_tfidf_explanation(candidate: dict, top_k: int, root_dir: Path) -> dict:
    tfidf_path = root_dir / "models" / "section_tfidf_extractor.joblib"
    ridge_path = root_dir / "models" / "ridge_fusion_model.joblib"

    if not tfidf_path.exists() or not ridge_path.exists():
        raise FileNotFoundError(
            f"Required model checkpoints not found. Checked:\n  {tfidf_path}\n  {ridge_path}"
        )

    tfidf_extractor = joblib.load(tfidf_path)
    ridge_artifact = joblib.load(ridge_path)

    # Instantiate XAI engine
    explainer = NoveltyExplainer(
        tfidf_extractor=tfidf_extractor,
        model=ridge_artifact
    )

    explanation = explainer.explain_proposal(
        candidate_dict=candidate,
        top_k=top_k,
        exclude_self=True
    )
    return explanation

def print_human_readable_report(result: dict):
    print("\n" + "=" * 78)
    print("         ACADEMIC PROJECT NOVELTY INSPECTION & XAI EXPLANATION REPORT")
    print("=" * 78)
    print(f"Candidate Title       : {result.get('candidate_proposal_title', 'N/A')}")
    print(f"Reference Corpus Size : {result.get('corpus_size_evaluated', 'N/A')} historical projects")
    print(f"Overall Novelty Score : {result.get('overall_novelty_score_pct', 0.0):.1f}%")
    print(f"Novelty Assessment    : {result.get('novelty_assessment_tier', 'N/A')}")
    print(f"Max Similarity Score  : {result.get('max_similarity_score', 0.0):.2f} / 4.00")
    print("Notice                : Estimated relative to the current database.")
    print("                        Does not establish originality or prove plagiarism.")
    print("=" * 78)

    top_matches = result.get("top_matches", [])
    print(f"\nTop {len(top_matches)} Most Similar Prior Projects in Corpus:")
    print("-" * 78)

    for rank, match in enumerate(top_matches, start=1):
        pid = match.get("historical_project_id", "N/A")
        title = match.get("historical_project_title", "N/A")
        domain = match.get("historical_domain", "N/A")
        sim = match.get("predicted_similarity", 0.0)
        raw_sim = match.get("raw_similarity_prediction", 0.0)
        nov = match.get("novelty_score_pct", 0.0)
        cat = match.get("novelty_category", "N/A")

        print(f"\n[Rank {rank}] Project ID: {pid} | Domain: {domain}")
        print(f"  Title               : {title}")
        print(f"  Predicted Similarity: {sim:.2f} / 4.00 (Raw: {raw_sim:.2f}) | Novelty: {nov:.1f}% ({cat})")
        print(f"  Additive Model Attribution : Intercept ({match.get('intercept', 0.0):+.3f}) + Contributions ({match.get('sum_of_contributions', 0.0):+.3f}) = Raw {match.get('reconstructed_raw_prediction', 0.0):.2f} (Clipped: {sim:.2f})")

        # Section-level Ranking
        print("  Section Overlap Ranking:")
        for sec_info in match.get("ranked_sections", []):
            sec = sec_info["section"]
            raw_s = sec_info["raw_similarity"]
            w = sec_info["model_weight"]
            c = sec_info["signed_contribution"]
            print(f"    - {sec:12s}: Lexical Sim = {raw_s:.3f} | Model Weight = {w:+.3f} | Score Contrib = {c:+.3f}")

        # Shared Keywords Evidence
        print("  Key Lexical Evidence (Shared N-Grams):")
        has_evidence = False
        for sec_name, ev in match.get("section_evidence", {}).items():
            terms = ev.get("shared_terms", [])
            if terms:
                has_evidence = True
                term_strs = [f"'{t['term']}' (sim +{t['term_cosine_contribution']:.3f})" for t in terms[:3]]
                print(f"    - {sec_name.capitalize():11s}: {', '.join(term_strs)}")
                # Show first snippet
                snip_cand = terms[0].get("candidate_snippet")
                snip_hist = terms[0].get("historical_snippet")
                if snip_cand and snip_hist:
                    print(f"        Candidate : \"{snip_cand}\"")
                    print(f"        Historical: \"{snip_hist}\"")
        if not has_evidence:
            print("    - No informative shared terms identified.")
        print("-" * 78)

    disclaimers = result.get("framework_disclaimers", {})
    print("\nMethodology Disclaimers (Corpus-Relative Heuristic - Does Not Establish Originality or Prove Plagiarism):")
    for k, v in disclaimers.items():
        print(f"  - {v}")
    print("=" * 78 + "\n")

def main():
    args = parse_args()
    candidate = load_candidate_dict(args)

    if args.model_type == "ridge_tfidf":
        result = run_ridge_tfidf_explanation(candidate, top_k=args.top_k, root_dir=ROOT_DIR)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print_human_readable_report(result)
    else:
        # Legacy Hybrid Fusion Execution
        tfidf_path = ROOT_DIR / "models" / "section_tfidf_extractor.joblib"
        dense_path = ROOT_DIR / "models" / "section_dense_extractor.joblib"
        model_path = ROOT_DIR / "models" / "hybrid_fusion_model.joblib"

        tfidf_extractor = joblib.load(tfidf_path)
        dense_extractor = joblib.load(dense_path)
        hybrid_model = HybridSectionFusionModel.load(str(model_path))

        tfidf_sim_df = tfidf_extractor.compute_candidate_similarity(candidate)
        dense_sim_df = dense_extractor.compute_candidate_similarity(candidate)

        hybrid_sim_df = tfidf_sim_df.copy()
        for col in DENSE_FEATURE_NAMES:
            hybrid_sim_df[col] = dense_sim_df[col]

        X_cand = hybrid_sim_df[HYBRID_FEATURE_NAMES].values
        predicted_sims = hybrid_model.predict(X_cand)
        hybrid_sim_df["predicted_similarity"] = predicted_sims
        hybrid_sim_df["novelty_score_pct"] = calculate_novelty_score(predicted_sims)

        max_idx = hybrid_sim_df["predicted_similarity"].idxmax()
        top_match = hybrid_sim_df.loc[max_idx]
        highest_sim = float(top_match["predicted_similarity"])
        overall_novelty = float(calculate_novelty_score(highest_sim))
        novelty_label = get_novelty_level(overall_novelty)

        top_matches = hybrid_sim_df.sort_values("predicted_similarity", ascending=False).head(args.top_k)

        legacy_result = {
            "candidate_proposal_title": candidate.get("project_title", "N/A"),
            "overall_novelty_score_pct": round(overall_novelty, 2),
            "novelty_assessment_tier": novelty_label,
            "max_similarity_score": round(highest_sim, 2),
            "top_matches": [
                {
                    "historical_project_id": row["project_id"],
                    "predicted_similarity": round(float(row["predicted_similarity"]), 2),
                    "novelty_score_pct": round(float(row["novelty_score_pct"]), 2),
                    "lexical_matches": {sec: round(float(row[f"{sec}_similarity"]), 2) for sec in SECTION_COLUMNS},
                    "semantic_matches": {sec: round(float(row[f"{sec}_dense_similarity"]), 2) for sec in SECTION_COLUMNS}
                }
                for _, row in top_matches.iterrows()
            ]
        }

        if args.json_output:
            print(json.dumps(legacy_result, indent=2))
        else:
            print("\n" + "=" * 75)
            print("           ACADEMIC PROJECT HYBRID NOVELTY INSPECTION REPORT")
            print("=" * 75)
            print(f"Candidate Title       : {candidate.get('project_title', 'N/A')}")
            print(f"Overall Novelty Score : {overall_novelty:.1f}%")
            print(f"Novelty Assessment    : {novelty_label}")
            print(f"Max Similarity Score  : {highest_sim:.2f} / 4.00 (against Project ID: {top_match['project_id']})")
            print("=" * 75)
            for rank, (_, row) in enumerate(top_matches.iterrows(), 1):
                print(f"Rank {rank}: Project ID [{row['project_id']}]")
                print(f"   Predicted Similarity: {row['predicted_similarity']:.2f} / 4.00")
                print("-" * 75)

if __name__ == "__main__":
    main()
