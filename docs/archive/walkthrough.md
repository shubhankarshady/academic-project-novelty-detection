# Walkthrough - Academic Project Novelty Detection

This document summarizes the development milestones of the Academic Project Novelty Detection platform, comparing exploratory notebook results with the verified active production implementation.

---

## 1. System Implementation Summary

The project consists of three core functional phases:
1. **Exploratory Modeling (`notebooks/`)**: Section fusion optimization, model training (Ridge, Random Forest, SLSQP Constrained Weights), dense semantic transformer experiments, and error analysis.
2. **Modular Library (`src/`)**: Clean package structure featuring:
   - [`src/data/loader.py`](file:///src/data/loader.py): Dataset and feature loaders.
   - [`src/features/section_tfidf.py`](file:///src/features/section_tfidf.py): 8-section TF-IDF vectorizers and cosine similarity calculations.
   - [`src/features/dense_embeddings.py`](file:///src/features/dense_embeddings.py): Dense transformer (`all-MiniLM-L6-v2`) section embedding extractor.
   - [`src/models/fusion_model.py`](file:///src/models/fusion_model.py) & [`src/models/hybrid_fusion.py`](file:///src/models/hybrid_fusion.py): Ridge, Random Forest, and Hybrid fusion models.
   - [`src/explainability/explainer.py`](file:///src/explainability/explainer.py): XAI engine with additive Ridge reconstruction and whole-word keyword evidence.
   - [`src/storage/db.py`](file:///src/storage/db.py): SQLite persistence layer with deterministic SHA-256 deduplication.
   - [`src/api/`](file:///src/api/): FastAPI application with endpoints for evaluation, saving, and historical querying.
3. **Application & Verification Tooling**:
   - `frontend/`: React 19 + Vite 8 + Tailwind CSS v4 interactive dashboard, evaluation forms, and audit history.
   - `scripts/`: CLI scoring tool ([`scripts/score_novelty.py`](file:///scripts/score_novelty.py)), training pipeline ([`scripts/train.py`](file:///scripts/train.py)), and holdout evaluation ([`scripts/evaluate_holdout.py`](file:///scripts/evaluate_holdout.py)).
   - `tests/`: 49 comprehensive unit and integration tests across API, storage, XAI, and scientific reliability.

---

## 2. Active Model vs. Historical Notebook Experiments

### Exploratory Notebook 05 (Historical SLSQP Exploration)
During initial exploration in [`notebooks/05_weighted_section_fusion.ipynb`](file:///notebooks/05_weighted_section_fusion.ipynb), a constrained least squares (SLSQP) model was tested where section weights were constrained to sum to $1.0$:
- Constrained SLSQP Weights: Dataset ($49.2\%$), Title ($23.2\%$), Technology ($19.6\%$), Methodology ($8.0\%$), all others ($0.0\%$).

### Active Deployed Production Model ([`models/ridge_fusion_model.joblib`](file:///models/ridge_fusion_model.joblib))
The active model used by [`NoveltyService`](file:///src/api/service.py) and [`score_novelty.py`](file:///scripts/score_novelty.py) is an **unconstrained Ridge Regression model ($\alpha=1.0$)** operating on 8 continuous section similarities:
- **Intercept ($b$):** `-0.0706`
- **Active Section Coefficients ($w_i$):**
  - `title_similarity`: `+0.1340`
  - `abstract_similarity`: `+0.3980`
  - `problem_similarity`: `-0.1259` *(Negative)*
  - `objectives_similarity`: `+0.2367`
  - `methodology_similarity`: `+0.0099`
  - `dataset_similarity`: `+2.2054` *(Dominates >55% of total scale)*
  - `algorithm_similarity`: `-0.0651` *(Negative)*
  - `technology_similarity`: `-0.1456` *(Negative)*

### Inactive 16-Feature Hybrid Model ([`models/hybrid_fusion_model.joblib`](file:///models/hybrid_fusion_model.joblib))
A hybrid model combining 8 TF-IDF similarities with 8 dense transformer similarities (`all-MiniLM-L6-v2`) exists as a saved artifact. It is currently **inactive in runtime API serving** due to feature collinearity and negative dense coefficients on synthetic benchmark labels.

---

## 3. Important Implementation & Evaluation Realities

> [!WARNING]
> **Synthetic Ground Truth Labels:**
> All 3,000 pairwise similarity ratings in the benchmark dataset originate from heuristic generation rules (`annotation_note: "SYNTHETIC BENCHMARK LABEL"`). Zero completed human faculty annotations currently exist. High regression metrics evaluate consistency with the synthetic rule function, not real-world academic novelty detection.

> [!CAUTION]
> **Known Prediction Range & Non-Monotonicity Defects:**
> 1. **Theoretical Ceiling:** The maximum possible raw prediction of the active Ridge model for an identical duplicate ($X=\vec{1}$) is $b + \sum w_i = 2.5768 / 4.0$. Consequently, an exact copy receives a novelty score of $35.58\%$, and the $< 25\%$ "Potential Duplicate" tier is mathematically unreachable.
> 2. **Non-Monotonicity:** Because `technology_tools` ($-0.1456$), `problem_statement` ($-0.1259$), and `algorithm_model` ($-0.0651$) have negative weights, sharing identical technologies penalizes predicted similarity, causing an exact duplicate of a project to rank #6 behind near-peers with differing tools.

> [!NOTE]
> **Database Segregation & Reference Corpus Independence:**
> User proposals saved to SQLite via POST `/api/save` are strictly persisted for auditing and user history. They do **not** enter the TF-IDF reference matrix and do **not** trigger model retraining. The reference corpus remains fixed at the 600 baseline projects.

---

## 4. Verified Automated Test Results

The test suite validates the operational integrity of the system:
```text
tests/test_api.py ..........                                             [ 20%]
tests/test_scientific_reliability.py ..........                          [ 40%]
tests/test_storage.py ..............                                     [ 69%]
tests/test_xai.py ...............                                        [100%]

======================= 49 passed, 3 warnings in 9.80s ========================
```
*(Warnings reflect scikit-learn unpickling version differences: model artifacts created with 1.9.1 loaded in 1.8.0 environment).*

### Verified CLI Novelty Scoring Execution
```bash
python scripts/score_novelty.py \
  --title "Quantum Machine Learning for Medical Image Segmentation" \
  --abstract "We propose quantum graph networks for medical image segmentation." \
  --top_k 2
```
```text
======================================================================
           ACADEMIC PROJECT NOVELTY INSPECTION REPORT
======================================================================
Candidate Title       : Quantum Machine Learning for Medical Image Segmentation
Overall Novelty Score : 99.4%
Novelty Assessment    : High estimated novelty relative to the current database
Max Similarity Score  : 0.02 / 4.00 (against Project ID: SYN-0569)
======================================================================
```
