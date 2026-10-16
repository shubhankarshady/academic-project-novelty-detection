# Walkthrough - Academic Project Novelty Detection

We have completed the implementation across all three planned phases:

1. **Notebook 05 Implementation**: Full section fusion optimization, model training (Ridge, Random Forest, SLSQP Constrained Weights), metric evaluation, and novelty score distribution analysis.
2. **Codebase Modularization (`src/`)**: Created production-grade Python package with data loaders, section TF-IDF vectorizers, section fusion models, and evaluation metrics.
3. **CLI / API Tooling (`scripts/`)**: Created training (`scripts/train.py`), candidate project scoring (`scripts/score_novelty.py`), and pipeline execution (`scripts/run_pipeline.py`).

---

## 1. Accomplished Tasks

### Phase 1: Notebook 05 — Weighted Section Fusion
- **File**: [`notebooks/05_weighted_section_fusion.ipynb`](file:///D:/mega/academic-project-novelty-detection/notebooks/05_weighted_section_fusion.ipynb)
- **Status**: Executed & Fully Rendered (143 KB notebook output).
- **Key Results**:
  - **Unweighted Mean Baseline**: RMSE `0.9187`
  - **Constrained Optimization (SLSQP)**: RMSE `0.7280` ($R^2 = 0.7315$)
    - Section Weights: Dataset (`49.6%`), Title (`22.7%`), Technology (`18.9%`), Methodology (`8.7%`)
  - **Ridge Regression**: RMSE `0.6867` ($R^2 = 0.7610$)
  - **Random Forest Regressor**: RMSE `0.7063` ($R^2 = 0.7472$)
  - **Novelty Score Range**: `22.7%` to `99.4%` (Mean: `65.5%`)

---

### Phase 2: Modular Codebase (`src/`)
- [`src/data/loader.py`](file:///D:/mega/academic-project-novelty-detection/src/data/loader.py): Load raw Excel projects, ground-truth similarity pairs, and feature CSVs.
- [`src/features/section_tfidf.py`](file:///D:/mega/academic-project-novelty-detection/src/features/section_tfidf.py): `SectionTFIDFExtractor` class to vectorize 8 text sections and compute pairwise/candidate similarities.
- [`src/models/fusion_model.py`](file:///D:/mega/academic-project-novelty-detection/src/models/fusion_model.py): `ConstrainedSectionFusion` and `SectionFusionRegressor` (Ridge & Random Forest wrappers with joblib serialization).
- [`src/utils/metrics.py`](file:///D:/mega/academic-project-novelty-detection/src/utils/metrics.py): Regression evaluation (RMSE, MAE, $R^2$, Pearson $r$, Spearman $\rho$) and Novelty Score calculations.

---

### Phase 3: CLI Tools (`scripts/`)
- [`scripts/train.py`](file:///D:/mega/academic-project-novelty-detection/scripts/train.py): Trains vectorizers and models, saving checkpoints to [`models/`](file:///D:/mega/academic-project-novelty-detection/models/).
- [`scripts/score_novelty.py`](file:///D:/mega/academic-project-novelty-detection/scripts/score_novelty.py): CLI tool to evaluate candidate project proposals against corpus projects and report percentage novelty score + top $K$ matches.
- [`scripts/run_pipeline.py`](file:///D:/mega/academic-project-novelty-detection/scripts/run_pipeline.py): End-to-end pipeline runner.

---

## 2. Verification & Validation

### Automated Verification Output

1. **Model Training Script (`scripts/train.py`)**:
```text
Learned Weights:
   title_similarity         :  23.2%
   abstract_similarity      :   0.0%
   problem_similarity       :   0.0%
   objectives_similarity    :   0.0%
   methodology_similarity   :   8.0%
   dataset_similarity       :  49.2%
   algorithm_similarity     :   0.0%
   technology_similarity    :  19.6%

Training Metrics (Ridge):
   rmse           : 0.6726
   mae            : 0.4826
   r2             : 0.7709
   pearson_r      : 0.8780
   spearman_rho   : 0.7898
```

2. **Novelty Scoring CLI (`scripts/score_novelty.py`)**:
```text
======================================================================
           ACADEMIC PROJECT NOVELTY INSPECTION REPORT
======================================================================
Candidate Title       : Quantum Machine Learning for Medical Image Segmentation
Overall Novelty Score : 75.7%
Novelty Assessment    : High Novelty (Original Work)
Max Similarity Score  : 0.97 / 4.00 (against Project ID: SYN-0425)
======================================================================

Top 3 Most Similar Existing Projects:
----------------------------------------------------------------------
Rank 1: Project ID [SYN-0425]
   Predicted Similarity: 0.97 / 4.00
   Section Matches     : Title=0.00, Abstract=0.03, Dataset=0.45, Algo=0.00
```
