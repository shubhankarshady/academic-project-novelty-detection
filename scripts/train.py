#!/usr/bin/env python
"""
Train section TF-IDF vectorizers, dense sentence transformers, and hybrid section fusion models.
Saves checkpoints into models/ directory and logs project-disjoint metrics.
"""
import sys
from pathlib import Path
import joblib
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_projects, load_similarity_pairs, load_feature_csv
from src.features.section_tfidf import SectionTFIDFExtractor, FEATURE_NAMES as TFIDF_FEATURE_NAMES
from src.features.dense_embeddings import SectionDenseEmbeddingExtractor, DENSE_FEATURE_NAMES
from src.models.fusion_model import SectionFusionRegressor
from src.models.hybrid_fusion import HybridSectionFusionModel, HYBRID_FEATURE_NAMES
from src.utils.metrics import evaluate_regression_metrics, project_disjoint_split, project_disjoint_cv

def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_path = root_dir / "data" / "raw" / "Academic_Project_Similarity_Synthetic_Dataset.xlsx"
    feat_path = root_dir / "results" / "feature_label_analysis" / "section_wise_tfidf_features.csv"
    models_dir = root_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print("1. Loading raw projects and ground-truth similarity pairs...")
    projects_df = load_projects(data_path)
    pairs_df = load_similarity_pairs(data_path)
    tfidf_df = load_feature_csv(feat_path)

    print("\n2. Fitting SectionTFIDFExtractor on 600 corpus projects...")
    tfidf_extractor = SectionTFIDFExtractor()
    tfidf_extractor.fit_transform_projects(projects_df)
    tfidf_path = models_dir / "section_tfidf_extractor.joblib"
    joblib.dump(tfidf_extractor, tfidf_path)
    print(f"   Saved SectionTFIDFExtractor to {tfidf_path}")

    print("\n3. Fitting SectionDenseEmbeddingExtractor (all-MiniLM-L6-v2) on 600 corpus projects...")
    dense_extractor = SectionDenseEmbeddingExtractor(model_name="all-MiniLM-L6-v2")
    dense_extractor.fit_transform_projects(projects_df)
    dense_pairs_df = dense_extractor.extract_pair_features(pairs_df)
    dense_path = models_dir / "section_dense_extractor.joblib"
    joblib.dump(dense_extractor, dense_path)
    print(f"   Saved SectionDenseEmbeddingExtractor to {dense_path}")

    print("\n4. Building 16-feature Hybrid Feature Matrix (TF-IDF + Dense Embeddings)...")
    hybrid_df = tfidf_df.copy()
    for col in DENSE_FEATURE_NAMES:
        hybrid_df[col] = dense_pairs_df[col]

    print("\n5. Evaluating Project-Disjoint 5-Fold Cross Validation...")
    cv_tfidf = project_disjoint_cv(hybrid_df, TFIDF_FEATURE_NAMES, n_splits=5)
    cv_dense = project_disjoint_cv(hybrid_df, DENSE_FEATURE_NAMES, n_splits=5)
    cv_hybrid = project_disjoint_cv(hybrid_df, HYBRID_FEATURE_NAMES, n_splits=5)

    print(f"   TF-IDF Only (8 features)  -> Mean RMSE: {cv_tfidf['mean_rmse']:.4f} +/- {cv_tfidf['std_rmse']:.4f} | R2: {cv_tfidf['mean_r2']:.4f}")
    print(f"   Dense Only (8 features)   -> Mean RMSE: {cv_dense['mean_rmse']:.4f} +/- {cv_dense['std_rmse']:.4f} | R2: {cv_dense['mean_r2']:.4f}")
    print(f"   Hybrid Model (16 features)-> Mean RMSE: {cv_hybrid['mean_rmse']:.4f} +/- {cv_hybrid['std_rmse']:.4f} | R2: {cv_hybrid['mean_r2']:.4f}")

    print("\n6. Training Final Hybrid Section Fusion Model on Unseen Project-Disjoint Split...")
    train_df, test_df = project_disjoint_split(hybrid_df, test_ratio=0.2, random_state=42)
    
    hybrid_model = HybridSectionFusionModel(model_type="ridge")
    hybrid_model.fit(train_df[HYBRID_FEATURE_NAMES].values, train_df["expert_similarity_label"].values)
    
    hybrid_model_path = models_dir / "hybrid_fusion_model.joblib"
    hybrid_model.save(hybrid_model_path)
    print(f"   Saved Hybrid Model checkpoint to {hybrid_model_path}")

    # Evaluate on unseen test projects
    preds_test = hybrid_model.predict(test_df[HYBRID_FEATURE_NAMES].values)
    test_metrics = evaluate_regression_metrics(test_df["expert_similarity_label"].values, preds_test)

    print("\nUnseen Project-Disjoint Test Metrics (Hybrid Model):")
    for k, v in test_metrics.items():
        print(f"   {k:15s}: {v:.4f}")

    print("\nModel Training & Checkpoint Generation Completed Successfully!")

if __name__ == "__main__":
    main()
