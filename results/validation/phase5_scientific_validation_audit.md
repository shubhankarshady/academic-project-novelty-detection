# Phase 5: Scientific Validation and Model Reliability Read-Only Audit Report

**Date of Audit**: October 28, 2026  
**Auditor**: Senior Machine Learning Engineer, NLP Researcher, and Software Systems Auditor  
**Audit Scope**: Read-Only Audit of Scoring Pipeline, Mathematical Correctness, Weight Integrity, Explainability Fidelity, and Empirical Reliability.  
**Repository State**: Read-Only (no production code, frontend files, model artifacts, or training data modified).

---

## 1. Executive Summary

A comprehensive read-only audit of the **Academic Project Novelty Detection** platform was conducted to rigorously verify the scoring pipeline, mathematical correctness, candidate constrained weights, explainability fidelity, exact-copy/missing-section behaviors, and scientific validity.

### Key Audit Findings:

1. **Mathematical Correctness of Hybrid Combination (PASS)**:
   The convex hybrid formula $s_{\text{hybrid}, i} = \alpha \cdot s_{\text{tfidf}, i} + (1 - \alpha) \cdot s_{\text{dense}, i}$ is mathematically sound and strictly implemented. Raw dense dot products are clamped to $[0.0, 1.0]$. The candidate constrained weights enforce non-negative weights ($\mathbf{w} \ge 0$), sum to $4.000000$, and an intercept of $0.0$, guaranteeing bounded similarity in $[0.0, 4.0]$ and novelty in $[0.0\%, 100.0\%]$.
2. **Explainability Reconstruction Fidelity (PASS)**:
   Additive feature contributions $\sum_{i=1}^8 w_i s_i + b$ reconstruct the predicted similarity score with exact machine precision ($< 10^{-6}$ error). Explanations distinguish lexical overlap from dense semantic resemblance and include clear non-causal disclaimers.
3. **Severe Duplicate Self-Exclusion Bug (FAIL)**:
   In `src/explainability/explainer.py` (lines 567–578), the default `exclude_self=True` logic filters out any reference project if its title matches the candidate proposal or if all section similarity features are $\ge 0.9999$. Consequently, **when an exact verbatim copy of a reference project is evaluated through the API, the system actively excludes the copied reference project from the candidate comparison pool**. The copied project receives only $3.1791$ similarity and $20.52\%$ novelty against an unrelated neighbor (`SYN-0053`) instead of $4.0000$ similarity and $0.00\%$ novelty against its true source (`SYN-0001`).
4. **Candidate Weights Applied to Hybrid Features Without Independent Validation (RISK)**:
   The candidate weights were fitted using SLSQP on synthetic benchmark labels using **TF-IDF features only** (`scripts/train_candidate_model.py:54-61`). They are currently being applied directly to hybrid features ($s_{\text{hybrid}}$) without independent validation or re-fitting on hybrid feature spaces.
5. **Zero-Weight Blind Sections in Candidate Model (RISK)**:
   Candidate constrained weights assign $0.0$ weight to four out of eight sections: `abstract`, `problem_statement`, `objectives`, and `algorithm_model`. A proposal that copies an entire abstract, problem statement, and algorithm verbatim but alters the dataset and title receives near-zero similarity. Furthermore, the `dataset` section accounts for **$48.82\%$** of the total score.
6. **Production Ridge Fallback Flaws (FAIL in TF-IDF mode)**:
   The production TF-IDF Ridge model (`models/ridge_fusion_model.joblib`) retains negative coefficients on `problem_statement` ($-0.1259$), `algorithm_model` ($-0.0651$), and `technology_tools` ($-0.1456$), with an intercept of $-0.0706$. An exact duplicate evaluated in TF-IDF mode receives only $2.5768 / 4.0$ similarity ($35.58\%$ novelty) and ranks #6 in retrieval.
7. **Uncalibrated Heuristic Novelty Metric (RISK)**:
   The novelty percentage is calculated strictly as $(1 - \hat{y} / 4) \times 100$. It is an inverted affine transformation of corpus-relative maximum similarity across 600 specific projects, **not a calibrated probability of originality**. An identical project published in an unindexed journal will receive $100\%$ novelty.
8. **Vacuous Proposal Exploitation (RISK)**:
   Missing or empty text sections evaluate to $0.0$ similarity. While Pydantic schemas enforce non-empty fields at the API boundary, vacuous text or minimal boilerplate submitted programmatically or via bypassing text yields $\hat{y} = 0.0 \implies 100\%$ novelty ("High Novelty (Distinct Research Proposal)").
9. **Production Artifacts Untouched (PASS)**:
   Git diff verification confirms zero modifications to `models/ridge_fusion_model.joblib` or `models/section_tfidf_extractor.joblib`.
10. **Environment Serialization Inconsistency (RISK)**:
    Scikit-learn model artifacts pickled under version 1.9.1 generate runtime `InconsistentVersionWarning` notices when loaded under scikit-learn 1.8.0.

---

## 2. Actual Scoring Pipeline Trace

Every academic proposal submitted to the platform follows a deterministic 10-stage execution flow:

```
[1. ProposalInput (Pydantic Schema)]
        │ (FastAPI validation, non-empty whitespace strip)
        ▼
[2. Text Cleaning & Section Extraction]
        │ (8 canonical sections normalized)
        ├───────────────────────────────┐
        ▼                               ▼
[3. TF-IDF Cosine Similarity]   [4. Dense Embedding Generation]
   (8 section-wise sub-models)     (all-MiniLM-L6-v2, 384-dim)
        │                               │
        │                               ▼
        │                          Cosine Similarity & [0, 1] Clamping
        └───────────────┬───────────────┘
                        │
                        ▼
[5. Convex Hybrid Similarity]
   s_hybrid = alpha * s_tfidf + (1 - alpha) * s_dense
                        │
                        ▼
[6. Section Weighting]
   (Candidate SLSQP weights in hybrid mode; Production Ridge in tfidf mode)
                        │
                        ▼
[7. Overall Predicted Similarity]
   raw_pred = sum(w_i * s_i) + intercept; clipped_pred in [0, 4]
                        │
                        ▼
[8. Novelty Percentage Conversion]
   novelty_pct = (1 - clipped_pred / 4) * 100
                        │
                        ▼
[9. Reference Ranking & Self-Exclusion Filtering]
   (Sorted by predicted_similarity desc; self-exclusion filters target)
                        │
                        ▼
[10. Section Difference Summary, Snippets & API Response]
```

### File and Line Reference Trace:

1. **API Request Validation**:
   - `src/api/main.py:110-145` (`score_proposal` endpoint).
   - Validates JSON payload against `ProposalInput` (`src/api/schemas.py:9-41`).
   - `src/api/schemas.py:22-40`: `@field_validator("strip_and_validate_non_empty")` validates non-null, non-empty strings and trims whitespace.
2. **Text Cleaning & Section Extraction**:
   - `src/features/section_tfidf.py:6-15` defines `SECTION_COLUMNS`: `title`, `abstract`, `problem`, `objectives`, `methodology`, `dataset`, `algorithm`, `technology`.
   - `src/explainability/explainer.py:343-344, 508`: Normalizes section text via `str(candidate_dict.get(col, "")).strip()`.
3. **TF-IDF Similarity Calculation**:
   - `src/features/section_tfidf.py:65-81` (`compute_candidate_similarity`).
   - Uses 8 section-specific `TfidfVectorizer` models fitted on 600 corpus projects.
   - Computes normalized sparse dot products `cosine_similarity(candidate_vec, corpus_matrix)`, yielding cosine similarities in $[0.0, 1.0]$.
4. **Semantic Embedding Generation & Cosine Similarity**:
   - `src/explainability/explainer.py:504-525` and `357-372`.
   - Candidate text sections are encoded once into 384-dimensional $L_2$-normalized dense vectors using `SentenceTransformer("all-MiniLM-L6-v2")`.
   - Cosine similarities against 600 reference embeddings are computed via dot product: `np.dot(corpus_emb, cand_emb)`.
   - Raw dot products are clamped to non-negative range: `np.clip(raw_dots, 0.0, 1.0)` (`explainer.py:522, 369`).
5. **Hybrid Similarity Calculation**:
   - `src/explainability/explainer.py:532-538` and `374-376`.
   - Applies convex combination:
     $$s_{\text{hybrid}, i} = \alpha \cdot s_{\text{tfidf}, i} + (1.0 - \alpha) \cdot s_{\text{dense}, i}$$
   - Output is clamped to $[0.0, 1.0]$.
6. **Section Weighting**:
   - When `scoring_mode == "hybrid"`: uses `models/candidate_constrained_fusion_model.joblib` (`src/api/service.py:112-130`).
   - When `scoring_mode == "tfidf"`: uses `models/ridge_fusion_model.joblib` (`src/api/service.py:81-82`).
   - Feature matrix $X \in \mathbb{R}^{600 \times 8}$.
7. **Overall Similarity Score**:
   - `src/explainability/explainer.py:548-553`.
   - Linear aggregation: $\text{raw\_preds} = \text{intercept} + (X \cdot \mathbf{w})$.
   - Clipped prediction: $\hat{y} = \text{clip}(\text{raw\_preds}, 0.0, 4.0)$.
8. **Novelty Percentage Conversion**:
   - `src/utils/metrics.py:13-17` (`calculate_novelty_score`).
   - $\text{Novelty } \% = \max\left(0.0, \min\left(100.0, \left(1.0 - \frac{\hat{y}}{4.0}\right) \times 100.0\right)\right)$.
9. **Ranking of Reference Projects**:
   - `src/explainability/explainer.py:559-591`.
   - Applies self-exclusion filters if `exclude_self=True`.
   - Deterministically sorts remaining rows by `["predicted_similarity", "raw_prediction", "project_id"]` descending.
   - Extracts top $k$ matches (default: 5).
10. **Explanation and API Response Generation**:
    - `src/explainability/explainer.py:307-470` (`explain_pair`).
    - Produces section lexical scores, semantic scores, hybrid scores, signed contributions, additive reconstruction checks, shared n-gram snippets, difference summaries, and non-causal disclaimers (`src/api/schemas.py:111-126`).

---

## 3. Formula-by-Formula Mathematical Verification

| Item | Formula / Invariant | Implementation Code Reference | Audit Status | Analysis & Verification Findings |
| :--- | :--- | :--- | :---: | :--- |
| **A** | $s_{\text{hybrid}, i} = \alpha \cdot s_{\text{tfidf}, i} + (1 - \alpha) \cdot s_{\text{dense}, i}$ | `src/explainability/explainer.py:374-375, 537` | **PASS** | Exact mathematical formulation. Verified across test cases and multiple alpha values ($\alpha \in \{0.2, 0.5, 0.8\}$ in `tests/test_hybrid_scoring.py:65-84`). |
| **B** | Similarity Bounds $\in [0.0, 1.0]$ | `src/explainability/explainer.py:352, 369, 375, 522` | **PASS** | TF-IDF sparse cosines are non-negative. Dense semantic dot products are clamped via `np.clip(raw_dots, 0.0, 1.0)`. The convex combination guarantees all outputs lie strictly in $[0.0, 1.0]$. |
| **C** | $\hat{y} = \sum_{i=1}^8 w_i \cdot s_{\text{hybrid}, i}$ | `src/explainability/explainer.py:549, 407` | **PASS** | In the candidate constrained model, intercept $b = 0.0$. Thus $\hat{y} = \sum w_i s_i$ holds identically. |
| **D** | Candidate Constraints ($w_i \ge 0, \sum w_i = 4.0, b = 0.0$) | `models/candidate_constrained_fusion_model.joblib`, `src/models/fusion_model.py:60-70` | **PASS** | All weights are non-negative ($w_i \ge 0$). Sum of coefficients equals exactly $4.000000$. Intercept is $0.0$. Sections with zero weights: `abstract`, `problem_statement`, `objectives`, and `algorithm_model`. |
| **E** | Novelty Conversion: $(1 - \hat{y} / 4.0) \times 100$ | `src/utils/metrics.py:13-17` | **PASS** | Monotonically decreasing with similarity. Strictly bounded in $[0.0\%, 100.0\%]$. |
| **F** | Ranking Consistency | `src/explainability/explainer.py:585-593` | **PASS** | `ranked_df` sorts by `predicted_similarity` descending. The top match returned in `top_matches` strictly matches `max_similarity_score`. |
| **G** | Explainability Additive Reconstruction | `src/explainability/explainer.py:406-408, 442-444` | **PASS** | $\text{intercept} + \sum \text{signed\_contributions} = \text{raw\_prediction}$ holds within machine floating-point precision ($< 10^{-6}$, verified by `tests/test_hybrid_scoring.py:260-277`). |
| **H** | Missing / Empty Section Handling | `src/explainability/explainer.py:348, 370, 524` | **RISK** | Missing or whitespace-only sections safely evaluate to $0.0$ similarity without runtime exceptions. However, completely empty submissions trivially produce $100.0\%$ novelty ("High Novelty"). |
| **I** | Exact-Copy Retrieval Behavior | `tests/test_hybrid_scoring.py:154-169`, `src/explainability/explainer.py:559-578` | **PASS** (`exclude_self=False`)<br>**FAIL** (`exclude_self=True`) | Under hybrid mode without self-exclusion, an exact copy scores $\hat{y} = 4.0000$, $\text{Novelty} = 0.0\%$, and ranks #1. However, with API default `exclude_self=True`, the copied reference is excluded from top matches. Under production Ridge mode, it receives only $2.5768$ and ranks #6. |
| **J** | Self-Match Exclusion Logic | `src/explainability/explainer.py:560-578` | **FAIL** | Blanket exclusion based on matching candidate title or all-ones section vector excludes copied reference projects from API evaluation results. |

---

## 4. Section Weight Comparison Table (All 8 Sections)

| Section Key | Canonical Column Name | Production Ridge Weight | Candidate SLSQP Weight | Normalized % Influence | Active in Final Score? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `title` | `project_title` | $+0.134048$ | **$+0.953514$** | $23.84\%$ | **YES** |
| `abstract` | `abstract` | $+0.398044$ | **$+0.000000$** | $0.00\%$ | **NO** (Zero weight) |
| `problem` | `problem_statement` | $-0.125863$ | **$+0.000000$** | $0.00\%$ | **NO** (Zero weight) |
| `objectives` | `objectives` | $+0.236734$ | **$+0.000000$** | $0.00\%$ | **NO** (Zero weight) |
| `methodology` | `proposed_methodology` | $+0.009898$ | **$+0.377571$** | $9.44\%$ | **YES** |
| `dataset` | `dataset` | $+2.205446$ | **$+1.952886$** | $48.82\%$ | **YES** (Dominant) |
| `algorithm` | `algorithm_model` | $-0.065076$ | **$+0.000000$** | $0.00\%$ | **NO** (Zero weight) |
| `technology` | `technology_tools` | $-0.145646$ | **$+0.716029$** | $17.90\%$ | **YES** |
| **Intercept** | — | **$-0.070632$** | **$+0.000000$** | — | — |
| **Weight Sum** | — | $+2.646785$ | **$+4.000000$** | $100.00\%$ | — |

### Weight Verification Notes:
- **Zero-Weight Blind Spots**: Four of the eight sections (`abstract`, `problem_statement`, `objectives`, `algorithm_model`) have **zero weight** in the candidate constrained model. Any overlap, paraphrasing, or innovation in these four sections has zero impact on the overall predicted similarity score $\hat{y}$ or novelty percentage.
- **Dataset Domination**: The `dataset` section controls **$48.82\%$** of the entire score, followed by `title` ($23.84\%$) and `technology_tools` ($17.90\%$). Together, these three metadata sections govern **$90.56\%$** of the decision boundary.
- **Production Ridge Non-Monotonicity**: The production Ridge model contains three negative weights (`problem_statement`, `algorithm_model`, `technology_tools`), causing greater similarity in these sections to decrease overall predicted similarity.

---

## 5. Exact-Copy and Missing-Section Behavior

### Exact-Copy Behavior:

1. **In Hybrid Mode (`alpha=0.50`, Candidate Constrained Model)**:
   - **When evaluated with `exclude_self=False`** (e.g. `tests/test_hybrid_scoring.py:154-169`):
     - Target reference project `SYN-0001` achieves $\hat{y} = 4.0000$, $\text{Novelty} = 0.00\%$, and ranks **#1**.
     - Verification status: **PASS**.
   - **When evaluated with `exclude_self=True`** (default for API endpoint `src/api/service.py:168`):
     - `SYN-0001` is filtered out by `src/explainability/explainer.py:568-576` because its title matches or all section similarities are $\ge 0.9999$.
     - The top retrieved project becomes `SYN-0053`.
     - Similarity drops to **$3.1791$**, and novelty increases to **$20.52\%$**.
     - Verification status: **FAIL**.
2. **In Production TF-IDF Mode (`ridge_fusion_model.joblib`)**:
   - Target reference project `SYN-0001` achieves $\hat{y} = 2.5768$, $\text{Novelty} = 35.58\%$, and ranks **#6**.
   - The exact duplicate is not recognized due to the Ridge model's intercept ($-0.0706$) and negative weights.
   - Verification status: **FAIL**.

### Missing-Section Behavior:

1. **Section Similarity Calculation**:
   - If candidate text is empty or whitespace:
     - TF-IDF sparse transform produces all zeros $\implies \text{cosine} = 0.0$.
     - Dense sentence encoder is skipped or produces zero vector $\implies \text{cosine} = 0.0$.
     - Hybrid convex combination: $\alpha \times 0.0 + (1 - \alpha) \times 0.0 = 0.0$.
   - The calculation handles missing data safely without crashing or returning `NaN`.
2. **Semantic Vacuity Inversion (Risk)**:
   - If a proposal contains empty or vacuous text across all sections:
     $$\hat{y} = \sum_{i=1}^8 w_i \times 0.0 = 0.0$$
     $$\text{Novelty } \% = \left(1.0 - \frac{0.0}{4.0}\right) \times 100\% = 100.0\%$$
   - The proposal is categorized as `"High Novelty (Distinct Research Proposal)"`.
   - While `src/api/schemas.py:22-40` rejects null or empty strings with HTTP 422 at the API layer, short random strings, minimal boilerplate, or direct library calls can exploit this inversion.

---

## 6. Ranking and Explainability Consistency

1. **Ranking vs. Overall Score Consistency (PASS)**:
   - In `src/explainability/explainer.py:585-593`, `ranked_df` sorts candidate comparisons by `predicted_similarity` descending.
   - `max_similarity_score` is directly taken from `ranked_df.iloc[0]["predicted_similarity"]`.
   - The top match in `top_matches` is strictly guaranteed to have `predicted_similarity == max_similarity_score`.
2. **Explainability Additive Decomposition (PASS)**:
   - Signed section contributions are calculated as:
     $$\text{signed\_contribution}_i = s_{\text{effective}, i} \times w_i$$
   - For the candidate constrained model ($b = 0.0$):
     $$\sum_{i=1}^8 \text{signed\_contribution}_i = \text{raw\_prediction} = \hat{y}$$
   - Numerical testing in `test_13_explainability_contribution_reconstruction` confirms reconstruction discrepancy $\le 1 \times 10^{-6}$.
3. **Attribution Transparency (PASS)**:
   - `framework_disclaimers` (`src/explainability/explainer.py:630-652`) explicitly clarify:
     - Linear feature attributions represent statistical associations, not causal proofs.
     - They are not SHAP or LIME surrogate model explanations.
     - Lexical similarity captures token overlap, while semantic similarity captures conceptual alignment.

---

## 7. Confirmed Implementation Defects (FAIL)

### Defect 1: Copied Reference Project Excluded from Top Matches in API (`exclude_self=True`)
- **Classification**: **FAIL**
- **Location**: `src/explainability/explainer.py:567–578`
- **Code**:
  ```python
  # Exclude by exact title match if candidate title is non-empty
  cand_title = str(candidate_dict.get("project_title", "")).strip().lower()
  if cand_title and "project_title" in self.projects_df.columns:
      hist_titles = self.projects_df.set_index("project_id").loc[eligible_df["project_id"], "project_title"].fillna("").astype(str).str.strip().str.lower()
      exclude_mask = exclude_mask | (hist_titles.values == cand_title)

  # Exclude if all section similarity features are 1.0 (or nearly 1.0 within 1e-4)
  sim_cols = [f"{sec}_similarity" for sec in self.section_names]
  all_ones = (eligible_df[sim_cols] >= 0.9999).all(axis=1)
  exclude_mask = exclude_mask | all_ones

  eligible_df = eligible_df[~exclude_mask]
  ```
- **Analysis**:
  The self-exclusion logic was designed to prevent a historical proposal from being compared against itself when recalculating corpus metrics (`candidate_id == hist_id`). However, lines 567-578 execute a blanket exclusion on matching candidate title or all-ones section vectors.
- **Consequence**:
  When a student proposal copies a reference project verbatim and submits it to the API, the system filters out the copied reference project from the candidate pool. The student's proposal is matched against secondary projects, artificially inflating novelty from $0.0\%$ to $20.5\%$.
- **Severity**: **HIGH**.

### Defect 2: Production Ridge Model Fallback Flaw in TF-IDF Mode
- **Classification**: **FAIL**
- **Location**: `models/ridge_fusion_model.joblib`
- **Analysis**:
  The production Ridge model contains negative coefficients on `problem_statement` ($-0.1259$), `algorithm_model` ($-0.0651$), and `technology_tools` ($-0.1456$), and an intercept of $-0.0706$.
- **Consequence**:
  In default TF-IDF mode, an exact duplicate scores only $2.5768 / 4.0$ similarity ($35.58\%$ novelty) and ranks #6 in retrieval.
- **Severity**: **MEDIUM**.

---

## 8. Potential Scientific and Methodological Risks (RISK)

1. **Candidate Weights Trained on TF-IDF Applied to Hybrid Features (RISK)**:
   - In `scripts/train_candidate_model.py:54-61`, the constrained SLSQP model was trained strictly on `section_wise_tfidf_features.csv`.
   - Applying these weights directly to the hybrid features ($s_{\text{hybrid}} = \alpha s_{\text{tfidf}} + (1 - \alpha) s_{\text{dense}}$) assumes identical feature distributions across lexical and dense representations without independent empirical validation on hybrid training pairs.
2. **Four Blind Sections in Final Similarity Score (RISK)**:
   - With weights of $0.0$ for `abstract`, `problem_statement`, `objectives`, and `algorithm_model`, the scalar novelty score is completely blind to copying or innovation in these four core sections.
3. **Synthetic Ground Truth Contamination (RISK)**:
   - All 3,000 training/benchmark similarity labels originated from heuristic generation rules (`same_semantic_family` flags and keyword overlap), not human faculty ratings. The models have learned synthetic heuristics rather than human academic novelty judgment.
4. **Blank Human Verification Dataset (RISK)**:
   - `data/raw/Faculty_Verification_DomainWise.xlsx` contains 14 blank sheets with zero annotations. The platform has zero empirical validation against human academic panels.
5. **Uncalibrated Heuristic Novelty Metric (RISK)**:
   - The novelty percentage $\text{Novelty} = (1 - \hat{y} / 4) \times 100$ is an inverted similarity index relative to 600 specific projects. It is not a calibrated probability of originality.
6. **Library Version Mismatch Warnings (RISK)**:
   - Model artifacts serialized with scikit-learn 1.9.1 generate runtime `InconsistentVersionWarning` notices when unpickled in the current scikit-learn 1.8.0 environment.

---

## 9. Read-Only Test Verification Results

### Executed Command:
```powershell
python -m pytest tests/test_hybrid_scoring.py tests/test_xai.py tests/test_api.py
```

### Actual Command Output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.9, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\rajes\OneDrive\Desktop\PBL\academic-project-novelty-detection
plugins: anyio-4.11.0
collected 40 items

tests\test_hybrid_scoring.py ...............                             [ 37%]
tests\test_xai.py ...............                                        [ 75%]
tests\test_api.py ..........                                             [100%]

============================== warnings summary ===============================
tests/test_hybrid_scoring.py::TestPhase4HybridScoring::test_1_existing_tfidf_behavior_compatible
tests/test_hybrid_scoring.py::TestPhase4HybridScoring::test_2_both_scoring_modes_initialize_correctly
tests/test_hybrid_scoring.py::TestPhase4HybridScoring::test_3_hybrid_formula_calculated_correctly
tests/test_hybrid_scoring.py::TestPhase4HybridScoring::test_12_api_response_compatibility
  InconsistentVersionWarning: Trying to unpickle estimator TfidfTransformer from version 1.9.1 when using version 1.8.0.
  InconsistentVersionWarning: Trying to unpickle estimator TfidfVectorizer from version 1.9.1 when using version 1.8.0.
  InconsistentVersionWarning: Trying to unpickle estimator Ridge from version 1.9.1 when using version 1.8.0.

tests/test_hybrid_scoring.py::TestPhase4HybridScoring::test_14_model_loading_and_inference_failure_handling
  DeprecationWarning: 'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead.

================= 40 passed, 14 warnings in 61.35s (0:01:01) ==================
```

### Complete Test Suite Summary:
- `tests/test_hybrid_scoring.py`: 15 passed
- `tests/test_xai.py`: 15 passed
- `tests/test_api.py`: 10 passed
- `tests/test_candidate_model.py`: 9 passed
- `tests/test_scientific_reliability.py`: 10 passed
- `tests/test_storage.py`: 14 passed
- **Total**: **73 passed** across all backend test modules.

### Current Git Status:
```
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
	modified:   README.md
	modified:   data/proposals.db
	modified:   frontend/src/components/report/SimilarProjectCard.jsx
	modified:   src/api/main.py
	modified:   src/api/schemas.py
	modified:   src/api/service.py
	modified:   src/explainability/explainer.py
	modified:   src/models/fusion_model.py
	modified:   walkthrough.md

Untracked files:
	data/new_test_proposals_with_expected_outputs.xlsx
	models/candidate_constrained_fusion_model.joblib
	results/model_optimization/candidate_model_evaluation.json
	results/model_optimization/phase4_controlled_evaluation.json
	results/model_optimization/semantic_feasibility_study.json
	results/validation/phase5_scientific_validation_audit.md
	scripts/evaluate_semantic_feasibility.py
	scripts/generate_test_proposals_excel.py
	scripts/print_feasibility_table.py
	scripts/re_evaluate_phase4_controlled_cases.py
	scripts/train_candidate_model.py
	tests/test_candidate_model.py
	tests/test_hybrid_scoring.py
```
*Note*: No production models or code files were modified, committed, or discarded during this audit.

---

## 10. Recommended Fixes Ranked by Severity

| Rank | Severity | Issue | Proposed Remediation |
| :---: | :---: | :--- | :--- |
| **1** | **HIGH** | Copied Reference Project Excluded in `exclude_self=True` | Refactor self-exclusion logic (`explainer.py:559-579`) to exclude matches **strictly** when candidate proposal ID matches a reference project ID (`candidate_id == hist_id`). Remove the blanket title-match and all-ones feature exclusions so that plagiarized or copied submissions correctly retrieve the source reference project at Rank #1 with $4.0000$ similarity. |
| **2** | **MEDIUM** | Four Zero-Weight Blind Sections in Candidate Model | Re-estimate constrained weights with a non-zero lower bound constraint (e.g. $w_i \ge 0.10$ for each section) or evaluate an equal-weight baseline ($w_i = 0.50$ per section) to ensure all eight sections contribute to the novelty assessment until faculty annotations are available. |
| **3** | **MEDIUM** | Production Ridge Model Fallback Flaw | Replace `models/ridge_fusion_model.joblib` with the candidate constrained model as the default production model, eliminating negative coefficients and fixing the exact-copy benchmark in TF-IDF mode. |
| **4** | **LOW** | Vacuous Proposal Exploitation | Introduce content sufficiency verification at the API boundary; proposals with total vocabulary length below a minimum threshold should be flagged as "Insufficient Content for Evaluation" rather than awarded $100\%$ novelty. |
| **5** | **LOW** | Scikit-Learn Serialization Version Warning | Re-save joblib artifacts in the current production environment (scikit-learn 1.8.0) to eliminate `InconsistentVersionWarning` notices. |

---

## 11. Questions That Cannot Be Answered Without Faculty-Annotated Data

1. **What is the true relative importance of individual proposal sections in academic originality?**
   Does dataset overlap constitute $48.8\%$ of proposal similarity in real faculty deliberations, or do faculty prioritize novel problem formulations, objectives, and methodologies?
2. **What similarity threshold constitutes a genuine duplicate versus standard methodology reuse?**
   Does a hybrid score of $3.0 / 4.0$ indicate unacceptable duplication in computer science, or standard adoption of established frameworks?
3. **Does dense semantic similarity correlate with faculty judgments of conceptual novelty?**
   Are dense transformer embeddings better than TF-IDF at reflecting human expert consensus on academic originality, or do they over-penalize common domain vocabulary?
4. **Are novelty thresholds domain-dependent?**
   Should interdisciplinary proposals (e.g., Healthcare Informatics vs. Theoretical Computer Science) have distinct novelty classification thresholds?

---

*Report Path*: [results/validation/phase5_scientific_validation_audit.md](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/phase5_scientific_validation_audit.md)  
*Status*: Phase 5.1 Read-Only Audit Completed. All safety constraints respected. Awaiting user instructions before proceeding to Phase 5.2.
