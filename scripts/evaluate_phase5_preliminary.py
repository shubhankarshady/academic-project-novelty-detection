#!/usr/bin/env python
"""
Phase 5.3: Preliminary Human Evaluation and Model Comparison Pipeline.

Evaluates existing models:
1. TF-IDF Ridge Model (Production)
2. Dense Semantic Embedding Model (all-MiniLM-L6-v2)
3. Hybrid Lexical-Semantic Fusion Model (alpha=0.50)

Against preliminary human faculty annotations (Reviewer 1, 120 pairs:
30 Dev / Calibration, 90 Held-Out Evaluation).

IMPORTANT:
This is a preliminary single-reviewer human evaluation.
It does NOT represent completed faculty validation, independent double-review,
or statistically established ground truth.
"""

import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from scipy.stats import pearsonr, spearmanr
import joblib

# Set root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Section definitions matching system architecture
SECTIONS = [
    "title",
    "abstract",
    "problem",
    "objectives",
    "methodology",
    "dataset",
    "algorithm",
    "technology",
]

SECTION_CAND_COLS = {
    "title": "candidate_title",
    "abstract": "candidate_abstract",
    "problem": "candidate_problem",
    "objectives": "candidate_objectives",
    "methodology": "candidate_methodology",
    "dataset": "candidate_dataset",
    "algorithm": "candidate_algorithm",
    "technology": "candidate_technology",
}

SECTION_REF_COLS = {
    "title": "reference_title",
    "abstract": "reference_abstract",
    "problem": "reference_problem",
    "objectives": "reference_objectives",
    "methodology": "reference_methodology",
    "dataset": "reference_dataset",
    "algorithm": "reference_algorithm",
    "technology": "reference_technology",
}


def load_data_and_artifacts():
    print("1. Loading evaluation data and trained model artifacts...")
    excel_path = ROOT_DIR / "data" / "annotation" / "faculty_evaluation_sample_120.xlsx"
    master_path = ROOT_DIR / "data" / "annotation" / "faculty_evaluation_sample_120_master.csv"

    b1 = pd.read_excel(excel_path, sheet_name="Evaluation_Batch_1")
    b2 = pd.read_excel(excel_path, sheet_name="Evaluation_Batch_2")
    master = pd.read_csv(master_path)

    b1["batch"] = "Evaluation_Batch_1"
    b2["batch"] = "Evaluation_Batch_2"

    combined = pd.concat([b1, b2], ignore_index=True)
    merged = pd.merge(
        combined,
        master[["pair_id", "subset", "provisional_synthetic_label", "provisional_category", "selection_rationale"]],
        on="pair_id",
        how="left"
    )

    tfidf_ext = joblib.load(ROOT_DIR / "models" / "section_tfidf_extractor.joblib")
    dense_ext = joblib.load(ROOT_DIR / "models" / "section_dense_extractor.joblib")
    ridge_dict = joblib.load(ROOT_DIR / "models" / "ridge_fusion_model.joblib")
    cand_model = joblib.load(ROOT_DIR / "models" / "candidate_constrained_fusion_model.joblib")
    
    return merged, tfidf_ext, dense_ext, ridge_dict["model"], cand_model


def validate_human_ratings(df):
    print("\n2. Validating Human Ratings (Reviewer 1)...")
    checks = {}

    # Check 1: All 120 reviewer_1_rating values present
    missing_r1 = int(df["reviewer_1_rating"].isna().sum())
    checks["all_120_present"] = (missing_r1 == 0 and len(df) == 120)

    # Check 2: Ratings are integers between 0 and 4
    r1_numeric = pd.to_numeric(df["reviewer_1_rating"], errors="coerce")
    is_integer = bool((r1_numeric.dropna() % 1 == 0).all())
    in_range = bool(((r1_numeric.dropna() >= 0) & (r1_numeric.dropna() <= 4)).all())
    checks["ratings_valid_integers_0_to_4"] = (is_integer and in_range)

    # Check 3: No project pairs duplicated
    pair_tuples = list(zip(df["project_candidate_id"], df["project_reference_id"]))
    dup_pairs = len(pair_tuples) - len(set(pair_tuples))
    dup_pair_ids = int(df["pair_id"].duplicated().sum())
    checks["no_duplicates"] = (dup_pairs == 0 and dup_pair_ids == 0)

    # Check 4: Batch 1 and Batch 2 each contain 60 pairs
    b1_count = int((df["batch"] == "Evaluation_Batch_1").sum())
    b2_count = int((df["batch"] == "Evaluation_Batch_2").sum())
    checks["dual_batch_balance_60_each"] = (b1_count == 60 and b2_count == 60)

    # Check 5: Reviewer 2 and adjudication fields remain unfilled
    r2_vals = int(df["reviewer_2_rating"].notna().sum())
    r2_comments = int(df["reviewer_2_comments"].notna().sum())
    adj_vals = int(df["adjudicated_rating"].notna().sum())
    adj_comments = int(df["adjudication_comments"].notna().sum())
    checks["reviewer_2_and_adjudication_unfilled"] = (
        r2_vals == 0 and r2_comments == 0 and adj_vals == 0 and adj_comments == 0
    )

    # Check 6: Dev/calibration (30) and held-out evaluation (90) preserved
    dev_count = int((df["subset"] == "DEV_CALIBRATION").sum())
    held_count = int((df["subset"] == "HELD_OUT_EVALUATION").sum())
    checks["subsets_preserved_30_90"] = (dev_count == 30 and held_count == 90)

    for k, v in checks.items():
        print(f"   [CHECK] {k}: {'PASS' if v else 'FAIL'}")

    rating_dist_all = df["reviewer_1_rating"].value_counts().sort_index().to_dict()
    rating_dist_held = df[df["subset"] == "HELD_OUT_EVALUATION"]["reviewer_1_rating"].value_counts().sort_index().to_dict()
    rating_dist_dev = df[df["subset"] == "DEV_CALIBRATION"]["reviewer_1_rating"].value_counts().sort_index().to_dict()

    print("\n   Rating Distribution (Reviewer 1):")
    print(f"   Overall (120 pairs): {rating_dist_all}")
    print(f"   Held-out (90 pairs): {rating_dist_held}")
    print(f"   Dev/Calib (30 pairs): {rating_dist_dev}")

    return checks, {
        "overall_distribution": rating_dist_all,
        "held_out_distribution": rating_dist_held,
        "dev_calibration_distribution": rating_dist_dev
    }


def compute_pair_features_and_predictions(df, tfidf_ext, dense_ext, ridge_model, cand_model):
    print("\n3. Computing Section Features and Model Predictions...")

    # Ensure dense encoder is loaded for missing candidates
    dense_ext._lazy_load_model()

    results = []
    w_ridge = np.asarray(ridge_model.coef_, dtype=float)
    b_ridge = float(ridge_model.intercept_)
    w_cand = np.asarray(cand_model.coef_, dtype=float)

    for idx, row in df.iterrows():
        pair_id = str(row["pair_id"])
        cand_id = str(row["project_candidate_id"])
        ref_id = str(row["project_reference_id"])
        r1_rating = int(row["reviewer_1_rating"])
        subset = str(row["subset"])

        s_tfidf = []
        s_dense = []

        # Check if candidate or reference are in precomputed embeddings
        c_idx = dense_ext.project_to_index.get(cand_id)
        r_idx = dense_ext.project_to_index.get(ref_id)

        for sec_name in SECTIONS:
            c_text = str(row[SECTION_CAND_COLS[sec_name]]).strip()
            r_text = str(row[SECTION_REF_COLS[sec_name]]).strip()

            # 1. TF-IDF Cosine Similarity
            vec = tfidf_ext.vectorizers.get(sec_name)
            if c_text and r_text and vec is not None:
                v_cand = vec.transform([c_text])
                v_ref = vec.transform([r_text])
                dot = float(v_cand.multiply(v_ref).sum())
                lexical_sim = float(np.clip(dot, 0.0, 1.0))
            else:
                lexical_sim = 0.0
            s_tfidf.append(lexical_sim)

            # 2. Dense Semantic Cosine Similarity
            if c_idx is not None and cand_id.startswith("SYN-"):
                c_emb = dense_ext.embeddings[sec_name][c_idx]
            else:
                c_emb = dense_ext.model.encode([c_text], convert_to_numpy=True, normalize_embeddings=True)[0]

            if r_idx is not None and ref_id.startswith("SYN-"):
                r_emb = dense_ext.embeddings[sec_name][r_idx]
            else:
                r_emb = dense_ext.model.encode([r_text], convert_to_numpy=True, normalize_embeddings=True)[0]

            dense_sim = float(np.clip(np.dot(c_emb, r_emb), 0.0, 1.0))
            s_dense.append(dense_sim)

        s_tfidf = np.array(s_tfidf, dtype=float)
        s_dense = np.array(s_dense, dtype=float)
        s_hybrid = 0.50 * s_tfidf + 0.50 * s_dense

        # Model 1: Production TF-IDF Ridge
        raw_tfidf = float(b_ridge + np.dot(w_ridge, s_tfidf))
        pred_tfidf = float(np.clip(raw_tfidf, 0.0, 4.0))

        # Model 2: Dense Semantic Embedding (Live Candidate Constrained Weights)
        raw_semantic = float(np.dot(w_cand, s_dense))
        pred_semantic = float(np.clip(raw_semantic, 0.0, 4.0))

        # Model 2b: Unweighted Mean Semantic Embedding Scaled to 4.0
        pred_semantic_mean = float(np.clip(4.0 * np.mean(s_dense), 0.0, 4.0))

        # Model 3: Hybrid Lexical-Semantic Fusion (alpha=0.50)
        raw_hybrid = float(np.dot(w_cand, s_hybrid))
        pred_hybrid = float(np.clip(raw_hybrid, 0.0, 4.0))

        # Identify relevant warnings
        warnings_list = []
        if cand_id.startswith("TEST-CASE"):
            warnings_list.append("Controlled benchmark case")
        if pred_tfidf < 1.0 and pred_semantic >= 3.0:
            warnings_list.append("Extreme lexical-semantic divergence (paraphrase)")
        if s_tfidf[5] == 1.0 and s_tfidf[0] < 0.2:
            warnings_list.append("Shared dataset with disjoint titles")
        if s_tfidf[0] == 1.0 and s_tfidf[5] == 0.0:
            warnings_list.append("Shared title with distinct datasets")
        if pred_tfidf >= 2.5 and raw_tfidf < 2.58 and np.all(s_tfidf > 0.95):
            warnings_list.append("Ridge model ceiling effect on near-duplicate")

        row_dict = {
            "pair_id": pair_id,
            "project_candidate_id": cand_id,
            "project_reference_id": ref_id,
            "domain_candidate": row["domain_candidate"],
            "domain_reference": row["domain_reference"],
            "reviewer_1_rating": r1_rating,
            "provisional_synthetic_label": row["provisional_synthetic_label"],
            "subset": subset,
            "batch": row["batch"],
            "tfidf_predicted_similarity": round(pred_tfidf, 4),
            "semantic_predicted_similarity": round(pred_semantic, 4),
            "semantic_mean_predicted_similarity": round(pred_semantic_mean, 4),
            "hybrid_predicted_similarity": round(pred_hybrid, 4),
            "tfidf_discrete_rating": int(np.clip(np.round(pred_tfidf), 0, 4)),
            "semantic_discrete_rating": int(np.clip(np.round(pred_semantic), 0, 4)),
            "semantic_mean_discrete_rating": int(np.clip(np.round(pred_semantic_mean), 0, 4)),
            "hybrid_discrete_rating": int(np.clip(np.round(pred_hybrid), 0, 4)),
            "warnings": "; ".join(warnings_list) if warnings_list else "None",
        }

        # Add section similarities for auditing
        for i, sec in enumerate(SECTIONS):
            row_dict[f"{sec}_sim_tfidf"] = round(s_tfidf[i], 4)
            row_dict[f"{sec}_sim_dense"] = round(s_dense[i], 4)
            row_dict[f"{sec}_sim_hybrid"] = round(s_hybrid[i], 4)

        results.append(row_dict)

    res_df = pd.DataFrame(results)
    return res_df


def calculate_evaluation_metrics(y_true, y_continuous, y_discrete, model_name):
    """
    Calculate full suite of continuous and discrete evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_continuous = np.asarray(y_continuous, dtype=float)
    y_discrete = np.asarray(y_discrete, dtype=int)

    mae = float(mean_absolute_error(y_true, y_continuous))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_continuous)))
    mbe = float(np.mean(y_continuous - y_true))

    p_corr, p_pval = pearsonr(y_true, y_continuous)
    s_corr, s_pval = spearmanr(y_true, y_continuous)

    exact_acc = float(np.mean(y_discrete == y_true) * 100.0)
    within_1_acc = float(np.mean(np.abs(y_discrete - y_true) <= 1) * 100.0)

    # Quadratic Weighted Kappa between model classification and reviewer rating
    # (Note: this is model-reviewer alignment, NOT inter-reviewer reliability!)
    qwk = float(cohen_kappa_score(y_true, y_discrete, weights="quadratic"))
    lwk = float(cohen_kappa_score(y_true, y_discrete, weights="linear"))

    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(y_true, y_discrete, labels=labels)

    # Per-class metrics
    macro_f1 = float(f1_score(y_true, y_discrete, labels=labels, average="macro", zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_discrete, labels=labels, average="weighted", zero_division=0) * 100.0)

    precision_per_class = precision_score(y_true, y_discrete, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_discrete, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_discrete, labels=labels, average=None, zero_division=0)

    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[f"class_{lbl}"] = {
            "support": int(np.sum(y_true == lbl)),
            "predicted_count": int(np.sum(y_discrete == lbl)),
            "precision": float(round(precision_per_class[i] * 100.0, 2)),
            "recall": float(round(recall_per_class[i] * 100.0, 2)),
            "f1": float(round(f1_per_class[i] * 100.0, 2))
        }

    return {
        "model": model_name,
        "n_samples": len(y_true),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Mean_Bias_Error": round(mbe, 4),
        "Pearson_r": round(float(p_corr), 4),
        "Pearson_p_value": float(p_pval),
        "Spearman_rho": round(float(s_corr), 4),
        "Spearman_p_value": float(s_pval),
        "Exact_Agreement_Pct": round(exact_acc, 2),
        "Within_1_Agreement_Pct": round(within_1_acc, 2),
        "Model_Reviewer_QWK": round(qwk, 4),
        "Model_Reviewer_Linear_Kappa": round(lwk, 4),
        "Macro_F1_Pct": round(macro_f1, 2),
        "Weighted_F1_Pct": round(weighted_f1, 2),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }


def generate_visualizations(eval_df, metrics_summary, figures_dir):
    print("\n4. Generating Phase 5.3 Validation Visualizations...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    palette = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    held_df = eval_df[eval_df["subset"] == "HELD_OUT_EVALUATION"].copy()

    # -------------------------------------------------------------
    # Figure 1: Human Rating Distribution (Held-Out & Calibration)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: Held-Out distribution
    counts_held = held_df["reviewer_1_rating"].value_counts().reindex(range(5), fill_value=0)
    bars1 = axes[0].bar(range(5), counts_held.values, color="#3B82F6", edgecolor="#1E3A8A", width=0.6)
    axes[0].set_title("Held-Out Evaluation Set (N = 90 Pairs)\nReviewer 1 Rating Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Ordinal Similarity Rating (0–4)", fontsize=11)
    axes[0].set_ylabel("Number of Project Pairs", fontsize=11)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(["0\n(Not Similar)", "1\n(Slightly)", "2\n(Moderately)", "3\n(Highly)", "4\n(Duplicate)"])
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{int(yval)} ({yval/90*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 65)

    # Right: Dev / Calibration distribution
    dev_df = eval_df[eval_df["subset"] == "DEV_CALIBRATION"]
    counts_dev = dev_df["reviewer_1_rating"].value_counts().reindex(range(5), fill_value=0)
    bars2 = axes[1].bar(range(5), counts_dev.values, color="#10B981", edgecolor="#065F46", width=0.6)
    axes[1].set_title("Development / Calibration Set (N = 30 Pairs)\nReviewer 1 Rating Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Ordinal Similarity Rating (0–4)", fontsize=11)
    axes[1].set_ylabel("Number of Project Pairs", fontsize=11)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(["0\n(Not Similar)", "1\n(Slightly)", "2\n(Moderately)", "3\n(Highly)", "4\n(Duplicate)"])
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{int(yval)} ({yval/30*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 20)

    plt.tight_layout()
    fig1_path = figures_dir / "figure1_human_rating_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"   Saved {fig1_path.name}")

    # -------------------------------------------------------------
    # Figure 2: Model vs Human Rating Comparison (Box / Strip plots)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    models_info = [
        ("tfidf_predicted_similarity", "1. TF-IDF Ridge Model", "#3B82F6"),
        ("semantic_predicted_similarity", "2. Semantic Model (all-MiniLM-L6-v2)", "#10B981"),
        ("hybrid_predicted_similarity", "3. Hybrid Model (alpha=0.50)", "#8B5CF6"),
    ]

    for ax, (col, title, color) in zip(axes, models_info):
        sns.boxplot(x="reviewer_1_rating", y=col, data=held_df, ax=ax, color=color, boxprops=dict(alpha=0.4), showmeans=True, meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"})
        sns.stripplot(x="reviewer_1_rating", y=col, data=held_df, ax=ax, color="black", alpha=0.5, jitter=0.2, size=5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Human Reviewer 1 Rating (0–4)", fontsize=10)
        ax.set_ylabel("Continuous Predicted Similarity (0–4)", fontsize=10)
        ax.set_ylim(-0.2, 4.2)
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(4.0, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Held-Out Evaluation Set (N = 90 Pairs): Model Predictions vs. Human Ratings", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = figures_dir / "figure2_model_vs_human_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"   Saved {fig2_path.name}")

    # -------------------------------------------------------------
    # Figure 3: Confusion Matrices for All Three Models
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cm_models = [
        ("tfidf_discrete_rating", "TF-IDF Ridge Model", axes[0]),
        ("semantic_discrete_rating", "Semantic Model", axes[1]),
        ("hybrid_discrete_rating", "Hybrid Model (alpha=0.50)", axes[2]),
    ]

    labels = [0, 1, 2, 3, 4]
    for col, title, ax in cm_models:
        cm = confusion_matrix(held_df["reviewer_1_rating"], held_df[col], labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_title(f"{title}\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Rating (Discrete 0–4)", fontsize=10)
        ax.set_ylabel("Actual Reviewer 1 Rating", fontsize=10)

    plt.suptitle("Held-Out Evaluation Set (N = 90 Pairs): Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = figures_dir / "figure3_confusion_matrices.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"   Saved {fig3_path.name}")

    # -------------------------------------------------------------
    # Figure 4: Model Error Distribution (Residuals)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    err_tfidf = held_df["tfidf_predicted_similarity"] - held_df["reviewer_1_rating"]
    err_sem = held_df["semantic_predicted_similarity"] - held_df["reviewer_1_rating"]
    err_hyb = held_df["hybrid_predicted_similarity"] - held_df["reviewer_1_rating"]

    sns.kdeplot(err_tfidf, ax=ax, label=f"TF-IDF Ridge (Mean Bias: {err_tfidf.mean():+.2f})", color="#2563EB", linewidth=2)
    sns.kdeplot(err_sem, ax=ax, label=f"Semantic Model (Mean Bias: {err_sem.mean():+.2f})", color="#10B981", linewidth=2)
    sns.kdeplot(err_hyb, ax=ax, label=f"Hybrid Model (Mean Bias: {err_hyb.mean():+.2f})", color="#8B5CF6", linewidth=2)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Zero Error (Ideal)")

    ax.set_title("Prediction Error Distribution (Residuals: Pred - Actual) on Held-Out Set (N = 90)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prediction Error (Similarity Points on 0–4 Scale)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig4_path = figures_dir / "figure4_model_error_distribution.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"   Saved {fig4_path.name}")

    # -------------------------------------------------------------
    # Figure 5: Actual vs Predicted Scatter Plot with Regression Lines
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(held_df))

    for ax, (col, title, color) in zip(axes, models_info):
        ax.scatter(held_df["reviewer_1_rating"] + jitter, held_df[col], alpha=0.6, color=color, edgecolors="none", s=40)
        # Identity line y = x
        ax.plot([0, 4], [0, 4], color="red", linestyle="--", alpha=0.7, label="y = x (Perfect Match)")
        
        # Trendline
        z = np.polyfit(held_df["reviewer_1_rating"], held_df[col], 1)
        p = np.poly1d(z)
        ax.plot(np.linspace(0, 4, 100), p(np.linspace(0, 4, 100)), color=color, linewidth=2, label=f"Fit (Slope: {z[0]:.2f})")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Reviewer 1 Rating (with Jitter)", fontsize=10)
        ax.set_ylabel("Continuous Prediction", fontsize=10)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.2, 4.2)
        ax.legend(loc="lower right", fontsize=9)

    plt.suptitle("Actual vs. Predicted Similarity on Held-Out Set (N = 90)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig5_path = figures_dir / "figure5_actual_vs_predicted.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"   Saved {fig5_path.name}")

    # -------------------------------------------------------------
    # Figure 6: Comparative Model Performance Summary Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    metrics_to_plot = ["Exact_Agreement_Pct", "Within_1_Agreement_Pct", "Macro_F1_Pct", "Weighted_F1_Pct"]
    metric_labels = ["Exact Agreement (%)", "Within-1 Agreement (%)", "Macro F1 (%)", "Weighted F1 (%)"]

    m_names = [m["model"] for m in metrics_summary]
    x = np.arange(len(metric_labels))
    width = 0.25

    for i, m_dict in enumerate(metrics_summary):
        vals = [m_dict[k] for k in metrics_to_plot]
        offset = (i - 1) * width
        rects = ax.bar(x + offset, vals, width, label=m_dict["model"], color=palette[i % len(palette)], edgecolor="black", alpha=0.85)
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2.0, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("Preliminary Model Performance Comparison (Held-Out N = 90 Pairs)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Performance (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    fig6_path = figures_dir / "figure6_comparative_model_performance.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"   Saved {fig6_path.name}")


def main():
    merged_df, tfidf_ext, dense_ext, ridge_model, cand_model = load_data_and_artifacts()

    # Step 2: Validate human ratings
    checks, rating_distributions = validate_human_ratings(merged_df)

    # Step 3 & 4: Compute features and model predictions
    eval_df = compute_pair_features_and_predictions(merged_df, tfidf_ext, dense_ext, ridge_model, cand_model)

    # Step 5: Calculate metrics on 90 held-out pairs
    held_df = eval_df[eval_df["subset"] == "HELD_OUT_EVALUATION"]
    dev_df = eval_df[eval_df["subset"] == "DEV_CALIBRATION"]

    models_config = [
        ("TF-IDF Ridge Model", "tfidf_predicted_similarity", "tfidf_discrete_rating"),
        ("Semantic Model (all-MiniLM-L6-v2)", "semantic_predicted_similarity", "semantic_discrete_rating"),
        ("Hybrid Model (alpha=0.50)", "hybrid_predicted_similarity", "hybrid_discrete_rating"),
    ]

    held_metrics = []
    print("\n--- Held-Out Evaluation Metrics (N = 90 Pairs) ---")
    for m_title, cont_col, disc_col in models_config:
        m = calculate_evaluation_metrics(
            held_df["reviewer_1_rating"],
            held_df[cont_col],
            held_df[disc_col],
            m_title
        )
        held_metrics.append(m)
        print(f"\nModel: {m_title}")
        print(f"   MAE: {m['MAE']} | RMSE: {m['RMSE']} | Mean Bias: {m['Mean_Bias_Error']:+.4f}")
        print(f"   Pearson r: {m['Pearson_r']} (p={m['Pearson_p_value']:.2e}) | Spearman rho: {m['Spearman_rho']} (p={m['Spearman_p_value']:.2e})")
        print(f"   Exact Agreement: {m['Exact_Agreement_Pct']}% | Within-1: {m['Within_1_Agreement_Pct']}%")
        print(f"   Model-Reviewer QWK: {m['Model_Reviewer_QWK']} | Macro F1: {m['Macro_F1_Pct']}% | Weighted F1: {m['Weighted_F1_Pct']}%")

    dev_metrics = []
    for m_title, cont_col, disc_col in models_config:
        m_dev = calculate_evaluation_metrics(
            dev_df["reviewer_1_rating"],
            dev_df[cont_col],
            dev_df[disc_col],
            f"{m_title} (Dev)"
        )
        dev_metrics.append(m_dev)

    # Step 6: Create visualizations
    figures_dir = ROOT_DIR / "results" / "validation" / "figures"
    generate_visualizations(eval_df, held_metrics, figures_dir)

    # Step 7: Save output files
    results_dir = ROOT_DIR / "results" / "validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Output 1: phase5_preliminary_human_evaluation.csv
    csv_eval_path = results_dir / "phase5_preliminary_human_evaluation.csv"
    eval_df.to_csv(csv_eval_path, index=False)
    print(f"\nSaved evaluation dataset to: {csv_eval_path}")

    # Output 2: phase5_preliminary_model_comparison.csv
    summary_rows = []
    for m in held_metrics:
        row = {
            "evaluation_split": "HELD_OUT_EVALUATION",
            "model": m["model"],
            "sample_size": m["n_samples"],
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "Mean_Bias_Error": m["Mean_Bias_Error"],
            "Pearson_r": m["Pearson_r"],
            "Spearman_rho": m["Spearman_rho"],
            "Exact_Agreement_Pct": m["Exact_Agreement_Pct"],
            "Within_1_Agreement_Pct": m["Within_1_Agreement_Pct"],
            "Model_Reviewer_QWK": m["Model_Reviewer_QWK"],
            "Macro_F1_Pct": m["Macro_F1_Pct"],
            "Weighted_F1_Pct": m["Weighted_F1_Pct"]
        }
        summary_rows.append(row)

    for m in dev_metrics:
        row = {
            "evaluation_split": "DEV_CALIBRATION",
            "model": m["model"],
            "sample_size": m["n_samples"],
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "Mean_Bias_Error": m["Mean_Bias_Error"],
            "Pearson_r": m["Pearson_r"],
            "Spearman_rho": m["Spearman_rho"],
            "Exact_Agreement_Pct": m["Exact_Agreement_Pct"],
            "Within_1_Agreement_Pct": m["Within_1_Agreement_Pct"],
            "Model_Reviewer_QWK": m["Model_Reviewer_QWK"],
            "Macro_F1_Pct": m["Macro_F1_Pct"],
            "Weighted_F1_Pct": m["Weighted_F1_Pct"]
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_comp_path = results_dir / "phase5_preliminary_model_comparison.csv"
    summary_df.to_csv(csv_comp_path, index=False)
    print(f"Saved model comparison table to: {csv_comp_path}")

    # Output 4: phase5_preliminary_evaluation_summary.json
    json_path = results_dir / "phase5_preliminary_evaluation_summary.json"
    full_summary = {
        "status": "PRELIMINARY_SINGLE_REVIEWER_EVALUATION",
        "date": "2026-11-13",
        "sample_size": {
            "total_pairs": len(eval_df),
            "dev_calibration_pairs": len(dev_df),
            "held_out_evaluation_pairs": len(held_df)
        },
        "validation_checks": checks,
        "rating_distributions": rating_distributions,
        "held_out_metrics": held_metrics,
        "dev_calibration_metrics": dev_metrics,
        "scientific_limitations": [
            "Only one human reviewer (Reviewer 1) has completed ratings.",
            "Independent second-reviewer ratings are pending.",
            "Adjudicated consensus ratings are pending.",
            "Inter-reviewer agreement (IRR) cannot yet be measured.",
            "The 120-pair sample is a limited evaluation cohort.",
            "Severe class skew towards Level 3 (57.5% of total sample; 60% of held-out set).",
            "Continuous-to-discrete integer rounding introduces boundary sensitivity.",
            "Results do not establish real-world statistical generalization to all academic projects."
        ]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved JSON summary to: {json_path}")

    print("\nPreliminary Evaluation Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()

