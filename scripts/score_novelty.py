#!/usr/bin/env python
"""
CLI Tool to score the Novelty of a candidate academic project proposal using Hybrid Section Fusion (TF-IDF + Dense Transformer Embeddings).
"""
import sys
import json
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.section_tfidf import SectionTFIDFExtractor, FEATURE_NAMES as TFIDF_FEATURE_NAMES
from src.features.dense_embeddings import SectionDenseEmbeddingExtractor, DENSE_FEATURE_NAMES
from src.models.hybrid_fusion import HybridSectionFusionModel, HYBRID_FEATURE_NAMES
from src.utils.metrics import calculate_novelty_score, get_novelty_level

def main():
    parser = argparse.ArgumentParser(description="Score Novelty of a candidate project proposal using Hybrid Semantic Fusion.")
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

    args = parser.parse_args()

    candidate = {}
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

    root_dir = Path(__file__).resolve().parent.parent
    tfidf_path = root_dir / "models" / "section_tfidf_extractor.joblib"
    dense_path = root_dir / "models" / "section_dense_extractor.joblib"
    model_path = root_dir / "models" / "hybrid_fusion_model.joblib"

    if not tfidf_path.exists() or not dense_path.exists() or not model_path.exists():
        print("Model checkpoints not found. Running training script first...")
        import train
        train.main()

    tfidf_extractor = joblib.load(tfidf_path)
    dense_extractor = joblib.load(dense_path)
    hybrid_model = HybridSectionFusionModel.load(model_path)

    # Extract 8 TF-IDF similarities + 8 Dense similarities
    tfidf_sim_df = tfidf_extractor.compute_candidate_similarity(candidate)
    dense_sim_df = dense_extractor.compute_candidate_similarity(candidate)

    hybrid_sim_df = tfidf_sim_df.copy()
    for col in DENSE_FEATURE_NAMES:
        hybrid_sim_df[col] = dense_sim_df[col]

    # Predict similarity score across all corpus projects
    X_candidate = hybrid_sim_df[HYBRID_FEATURE_NAMES].values
    predicted_sims = hybrid_model.predict(X_candidate)
    hybrid_sim_df["predicted_similarity"] = predicted_sims
    hybrid_sim_df["novelty_score_pct"] = calculate_novelty_score(predicted_sims)

    # Find highest similarity match in corpus
    max_idx = hybrid_sim_df["predicted_similarity"].idxmax()
    top_match = hybrid_sim_df.loc[max_idx]
    
    highest_sim = float(top_match["predicted_similarity"])
    overall_novelty = float(calculate_novelty_score(highest_sim))
    novelty_label = get_novelty_level(overall_novelty)

    # Calculate average lexical vs semantic similarity against top match
    lexical_match_pct = float(np.mean([top_match[f] for f in TFIDF_FEATURE_NAMES])) * 100.0
    semantic_match_pct = float(np.mean([top_match[f] for f in DENSE_FEATURE_NAMES])) * 100.0

    print("\n" + "=" * 75)
    print("           ACADEMIC PROJECT HYBRID NOVELTY INSPECTION REPORT")
    print("=" * 75)
    print(f"Candidate Title       : {candidate.get('project_title', 'N/A')}")
    print(f"Overall Novelty Score : {overall_novelty:.1f}%")
    print(f"Novelty Assessment    : {novelty_label}")
    print(f"Max Similarity Score  : {highest_sim:.2f} / 4.00 (against Project ID: {top_match['project_id']})")
    print(f"  • Lexical Overlap   : {lexical_match_pct:.1f}% keyword matching")
    print(f"  • Conceptual Match  : {semantic_match_pct:.1f}% dense semantic similarity")
    print("=" * 75)

    top_matches = hybrid_sim_df.sort_values("predicted_similarity", ascending=False).head(args.top_k)
    print(f"\nTop {args.top_k} Most Similar Existing Projects in Corpus:")
    print("-" * 75)
    for rank, (_, row) in enumerate(top_matches.iterrows(), 1):
        print(f"Rank {rank}: Project ID [{row['project_id']}]")
        print(f"   Predicted Similarity: {row['predicted_similarity']:.2f} / 4.00")
        print(f"   Lexical Matches     : Title={row['title_similarity']:.2f}, Abstract={row['abstract_similarity']:.2f}, Dataset={row['dataset_similarity']:.2f}")
        print(f"   Semantic Matches    : Title={row['title_dense_similarity']:.2f}, Abstract={row['abstract_dense_similarity']:.2f}, Dataset={row['dataset_dense_similarity']:.2f}")
        print("-" * 75)

if __name__ == "__main__":
    main()
