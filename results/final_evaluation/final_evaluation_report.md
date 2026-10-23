# Final Holdout Evaluation Report (Leakage-Safe)

**Project:** `academic-project-novelty-detection`  
**Evaluation Protocol:** 20% Project-Level Disjoint Holdout (Strict Zero-Leakage)  
**Selected Candidate:** Ridge Regression ($\alpha = 10.0$) on 8-Section TF-IDF Similarities  
**Evaluation Date:** October 22, 2026  
**Status:** Completed  

---

## 1. Executive Summary & Holdout Metrics

The final holdout evaluation was conducted on a permanent 20% test slice (120 projects, 120 pairs) partitioned with Seed 42 from the cleaned academic project dataset.
The TF-IDF vectorizers were fitted strictly on the 480 development projects. The Ridge regression model (alpha=10.0) was fitted on the 1,916 development pairs and evaluated on the 120 held-out test pairs.

| Metric | Holdout Test Score (Fold 5 Partition) | 5-Fold CV Mean (Reference) | 5-Fold CV Range [Min, Max] |
|---|:---:|:---:|:---:|
| **RMSE** | **0.7734** | $0.6511 \pm 0.0910$ | [0.5707, 0.7734] |
| **MAE** | **0.5648** | $0.4731 \pm 0.0680$ | [0.4124, 0.5648] |
| **$R^2$ Score** | **0.6677** | $0.7737 \pm 0.0695$ | [0.6677, 0.8351] |
| **Pearson $r$** | **0.8237** | — | — |
| **Spearman $\rho$** | **0.7508** | $0.7928 \pm 0.0391$ | [0.7501, 0.8305] |

### Relationship to 5-Fold Cross-Validation
- **Mathematical Identity:** The 120 held-out test projects (the final 20% under `RandomState(42)`) correspond to the test partition of Fold 5 in the 5-fold cross-validation. Consequently, the holdout metrics match Fold 5 (`RMSE = 0.7734`).
- **Variance Across Folds:** Across the 5 folds, test RMSE ranged from $0.5707$ (Fold 3) to $0.7734$ (Fold 5). The 5-fold mean is $0.6511 \pm 0.0910$. Fold 5 represents the highest-error fold in the split, explaining why the single holdout score ($0.7734$) is higher than the cross-validation mean.
- **Partition Timing Disclosure:** Because the 5-fold cross-validation was run across all 600 projects prior to running this single-split evaluation script, these 120 projects were part of the cross-validation pool that selected $\alpha=10.0$. While test labels were never seen during model training, the split was derived from the same seeded partitioning rather than an a priori isolated vault.


> [!WARNING]
> **Synthetic Benchmark Limitation:**
> All similarity labels in this dataset are synthetic benchmark annotations generated from heuristic overlap rules.
> These evaluation results reflect internal algorithmic and pipeline validity against synthetic benchmarks. They do **not** represent real-world novelty detection performance validated by academic faculty panels.

---

## 2. Partition & Sample Size Breakdown

A strict project-level split was enforced using seed `42`. Any pair containing one project from the training partition and one project from the test partition (cross-boundary pairs) was excluded from both training and testing.

| Partition | Projects Count | Eligible Pairs | Percentage of Total Pairs | Role |
|---|:---:|:---:|:---:|---|
| **Development (Train)** | 480 (80.0%) | 1916 | 64.0% | Model training & TF-IDF vocabulary fitting |
| **Untouched Holdout (Test)** | 120 (20.0%) | 120 | 4.0% | One-time final generalization evaluation |
| **Excluded (Cross-Boundary)** | — | 960 | 32.0% | Discarded to enforce zero representation leakage |
| **Total** | **600** | **2996** | **100.0%** | Full cleaned dataset |

---

## 3. Learned Model Parameters (Explainable AI)

Because Ridge regression is a transparent linear model, the learned coefficients provide direct, monotonic section-level similarity weights.
Intercept: `-0.1106`

| Feature | Learned Coefficient ($w_i$) |
|---|:---:|
| `dataset_similarity` | +1.9352 |
| `abstract_similarity` | +0.4984 |
| `objectives_similarity` | +0.2234 |
| `technology_similarity` | -0.1453 |
| `title_similarity` | +0.1446 |
| `problem_similarity` | +0.1288 |
| `algorithm_similarity` | -0.0694 |
| `methodology_similarity` | +0.0013 |

The primary drivers of similarity are `dataset_similarity` (+1.9352) and `abstract_similarity` (+0.4984), followed by `objectives_similarity` (+0.2234) and `title_similarity` (+0.1446). Minor negative coefficients for `technology_similarity` (-0.1453) and `algorithm_similarity` (-0.0694) arise from multicollinearity among shared tool keywords.

---

## 4. Error Breakdown by Target Similarity Label

| True Label | Test Pairs | Mean Prediction | Mean MAE | Mean RMSE |
|:---:|:---:|:---:|:---:|:---:|
| Label 0 | 49 | 0.150 | 0.150 | 0.336 |
| Label 1 | 20 | 1.454 | 1.253 | 1.303 |
| Label 2 | 25 | 2.553 | 0.553 | 0.568 |
| Label 3 | 16 | 2.554 | 0.446 | 0.464 |
| Label 4 | 10 | 2.563 | 1.437 | 1.441 |

---

## 5. Artifacts & Reproducibility Files

All generated outputs are saved under [`results/final_evaluation/`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/final_evaluation/):
- **`split_manifest.json`**: Exact list of the 480 training project IDs and 120 testing project IDs.
- **`holdout_metrics.json`**: Machine-readable JSON summary of final holdout metrics and learned parameters.
- **`holdout_predictions.csv`**: Pair-by-pair predictions, errors, section features, and novelty scores for all 120 test pairs.
- **`final_evaluation_report.md`**: Full documentation report.
