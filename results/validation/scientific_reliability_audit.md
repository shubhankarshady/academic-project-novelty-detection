# Scientific Reliability & Model Behavior Audit Report

**Project:** Academic Project Novelty Detection & Explainable AI (XAI)  
**Audit Date:** October 22, 2026  
**Audited Artifacts:**
- `data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx`
- `data/raw/Faculty_Verification_DomainWise.xlsx`
- `data/raw/Project_Similarity_Pairs_0_to_4.xlsx`
- `data/processed/similarity_pairs_clean.csv`
- `models/section_tfidf_extractor.joblib`
- `models/ridge_fusion_model.joblib`
- `src/explainability/explainer.py`
- `src/api/service.py`
- `src/utils/metrics.py`
- `results/model_optimization/`
- `results/final_evaluation/`
- `tests/test_scientific_reliability.py`

---

## 1. Executive Summary

This independent scientific audit evaluated whether the novelty scores, similarity predictions, and explainability attributions produced by the system are scientifically reliable indicators of academic novelty or merely plausible-looking heuristic outputs.

### Overall Verdict: **WARNING — Rigorous Heuristic Prototype, Not Independently Validated AI**
The application architecture, API pipeline, storage layer, and mathematical reconstruction of linear feature attributions are implemented correctly and reproducibly. However, from a scientific and machine learning standpoint, **the novelty predictions cannot be claimed as validated measures of academic originality**. 

### Primary Findings:
1. **100% Synthetic Ground Truth:** Across all 3,000 training/benchmark pairs and all 13 sheets of the faculty verification package (`Faculty_Verification_DomainWise.xlsx`), there are **zero (0)** real-world faculty ratings. The regression model was trained to predict synthetic rule-based labels, not actual human novelty judgements.
2. **Mathematical Scale Disconnect ($[0, 4]$ vs $[0, 1]$):** The model predicts similarity on a **0.0 to 4.0 scale**, not 0.0 to 1.0. A reported maximum similarity of `1.137` corresponds to a normalized similarity of $1.137 / 4.0 = 28.4\%$, which reproduces the displayed novelty of $71.6\%$. However, without explicit $[0, 4]$ scale indicators in the UI, users are easily misled into assuming similarity exceeds 100%.
3. **Severe Theoretical Ceiling:** Because the fitted Ridge model has an intercept of $-0.0706$ and a sum of coefficients of $+2.6476$, an identical $100\%$ duplicate of a project across all eight sections achieves a maximum raw prediction of only **$2.5768$** (out of $4.0$). Consequently, an exact copy receives a novelty score of **$35.58\%$** ("Low Novelty") and can **never** reach the "Very Low Novelty / Duplicate" tier ($< 25\%$).
4. **Section Non-Monotonicity (Negative Coefficients):** The Ridge model assigned negative weights to `technology_tools` ($-0.1456$), `problem_statement` ($-0.1259$), and `algorithm_model` ($-0.0651$). As a result, sharing identical technologies or algorithms penalizes predicted similarity, causing an exact duplicate of a project to rank **#6** in the corpus while near-peers with differing tools rank higher.
5. **Pure Lexical Vulnerability:** The feature pipeline relies on TF-IDF word n-grams. When a proposal is paraphrased into alternative academic terminology, predicted similarity drops to $0.0175$, yielding an unwarranted **$99.56\%$ novelty score** for a conceptual duplicate.
6. **Disproportionate Dataset Sensitivity:** The `dataset` feature has an outsized weight of **$+2.2054$** ($>55\%$ of the total scale). Merely sharing an open dataset name (e.g. `PlantVillage`) causes an unrelated marine underwater acoustic sensor project to receive a similarity score of $2.1254$ and drops its novelty to $46.87\%$.

---

## 2. Data Provenance & Leakage Audit

| Provenance Audit Dimension | Finding | Status |
|---|---|:---:|
| **Data Realism** | All 600 projects are marked `SYNTHETIC` in the raw dataset. | **WARNING** |
| **Label Source** | 100% of the 3,000 pairwise similarity ratings are synthetic heuristics (`annotation_note: "SYNTHETIC BENCHMARK LABEL"`). | **FAIL** |
| **Faculty Verification** | `Faculty_Verification_DomainWise.xlsx` contains 0 completed faculty annotations across all sheets. | **FAIL** |
| **Synthetic Rule Influence** | Labels 2, 3, and 4 correlate 100% with `same_semantic_family == 1`. Label 0 correlates 100% with `same_semantic_family == 0`. | **WARNING** |
| **Pair Deduplication** | 4 redundant rows (1 exact duplicate, 3 reversed pairs) were identified and removed in `data/processed/`. | **PASS** |
| **Project Disjointness** | Splits were partitioned by unique project IDs to avoid direct pair leakage across train and test sets. | **PASS** |
| **Holdout Independence** | The "final holdout" corresponds exactly to **Fold 5** of the 5-fold CV, which was part of the $\alpha=10.0$ hyperparameter tuning run. | **WARNING** |

### Detailed Provenance Assessment:
- **How similarity labels were generated:** Inspection of `Academic_Project_Similarity_Synthetic_Dataset.xlsx` reveals that similarity labels were assigned by rule-based templates based on whether two projects belong to the same domain/family and their baseline lexical token overlap.
- **Influence on evaluation metrics:** Because both the training labels and evaluation labels were generated using the same synthetic rules, high regression metrics ($R^2 \approx 0.77$, $\text{RMSE} \approx 0.65$) merely indicate that the Ridge regression model successfully learned the synthetic generation function. They do **not** prove that the model detects real-world academic novelty.
- **Reporting Limitations:** It is scientifically invalid to report these metrics as "Novelty Detection Accuracy" on academic projects. They must be accurately reported as *"Performance on a Synthetic Benchmark Rule Dataset."*

---

## 3. Novelty Calculation Verification

### Formula Audit:
In `src/utils/metrics.py` and `src/explainability/explainer.py`:
$$\text{sim\_norm} = \text{clip}\left(\frac{\text{predicted\_similarity}}{4.0}, 0.0, 1.0\right)$$
$$\text{Novelty (\%)} = (1.0 - \text{sim\_norm}) \times 100.0$$

### Numerical Verification of User Example:
- **Input Maximum Similarity:** $1.137$
- **Step 1 (Normalization):** $1.137 / 4.0 = 0.28425$
- **Step 2 (Novelty calculation):** $(1.0 - 0.28425) \times 100\% = 71.575\%$
- **Step 3 (Rounding):** Displayed as **$71.6\%$**
- **Assessment Tier:** $71.6\% \in [50.0\%, 75.0\%) \implies$ *"Moderate Novelty (Partial Overlap)"*

### Status: **PASS (Mathematically Consistent) / WARNING (User Communication Risk)**
The calculation is mathematically consistent and correctly implements the coded formula. However:
1. The regression target was an integer scale from **0 to 4**, whereas users and frontend consumers expect cosine similarity in $[0, 1]$.
2. Presenting `Max Similarity: 1.137` alongside `Novelty: 71.6%` without clarifying the $[0, 4]$ denominator creates confusion and can lead observers to assume a calculation error or broken bounding.

---

## 4. Controlled Test Cases & Model Behavior

All seven controlled cases were executed through the actual model pipeline via `tests/test_scientific_reliability.py`.

| Case # | Candidate Description | Expected Behavior | Actual Raw Sim ($[0, 4]$) | Actual Novelty % | Top Retrieved Match | Expected Behavior Occurred? |
|:---:|---|---|:---:|:---:|---|:---:|
| **1a** | Exact copy of `SYN-0001` (`exclude_self=False`) | Maximum possible similarity ($4.0$), novelty $< 10\%$ | $2.5768$ | $35.58\%$ | `SYN-0157` (Rank #1, sim $2.6551$); `SYN-0001` (Rank #6) | **FAIL** (Model ceiling and non-monotonicity) |
| **1b** | Exact copy of `SYN-0001` (production `exclude_self=True`) | Exclude `SYN-0001`, find closest peer | $2.6551$ | $33.62\%$ | `SYN-0157` ("AI-Assisted Crop Disease Detection Using CNN") | **PASS** (Self-exclusion worked) |
| **2** | Copy with minor wording edits | High similarity ($> 2.5$), Low novelty ($< 40\%$) | $2.6352$ | $34.12\%$ | `SYN-0157` ("AI-Assisted Crop Disease Detection Using CNN") | **PASS** |
| **3** | Paraphrased version of `SYN-0001` (same concepts, altered phrasing) | High conceptual similarity ($> 2.0$) | $0.0175$ | $99.56\%$ | `SYN-0156` (sim $0.0175$) | **FAIL** (Lexical blindness: false 99.6% novelty) |
| **4** | Same title as `SYN-0001`, different content (Drone sensing) | Moderate/Low similarity ($< 1.5$) | $1.1100$ | $72.25\%$ | `SYN-0275` ("Industrial Defect Detection Using YOLO") | **PASS** (Filtered `SYN-0001`, title weight $+0.13$) |
| **5** | Completely unrelated topic (Quantum Key Distribution) | Zero similarity ($0.0$), novelty $100\%$ | $0.0000$ | $100.00\%$ | `SYN-0001` (sim $0.0000$) | **PASS** |
| **6** | Single section overlap: `dataset = PlantVillage` (Underwater sensor) | Low similarity ($< 0.5$), high novelty ($> 85\%$) | $2.1254$ | $46.87\%$ | `SYN-0208` ("Crop Disease Detection Using CNN") | **FAIL** (Excessive dataset weight $+2.2054$) |
| **7** | Generic academic boilerplate ("Data analysis application") | Low similarity ($< 0.5$), high novelty ($> 85\%$) | $0.0985$ | $97.54\%$ | `SYN-0378` (sim $0.0985$) | **PASS** (Stop words and generic filter active) |

---

## 5. Top-Match Correctness & Explainable AI (XAI) Verification

| XAI Verification Dimension | Finding | Status |
|---|---|:---:|
| **Historical ID Existence** | All returned top match IDs (`SYN-XXXX`) exist in `projects_df`. | **PASS** |
| **Record Alignment** | Titles, domains, and section texts align with source records. | **PASS** |
| **Snippet Substring Integrity** | All snippets verified as exact physical substrings of candidate and historical texts (`verify_snippet == True`). | **PASS** |
| **Shared Keyword Verification** | All displayed terms verified to occur with whole-word boundaries in both source sections. | **PASS** |
| **Linear Model Reconstruction** | $\text{Intercept} + \sum \text{Signed Contributions} = \text{Raw Prediction}$ holds within $< 10^{-5}$. | **PASS** |
| **Database Segregation** | User submissions in `data/proposals.db` never contaminate the reference corpus. | **PASS** |
| **Explanation Misinterpretation Risk** | Ridge contributions reflect statistical associations in the synthetic set, **not** causal proof of copying or plagiarism. | **WARNING** |

---

## 6. Evaluation Leakage Risks

1. **Pre-Model Selection Holdout Exposure:**
   - In `scripts/optimize_models.py`, 5-fold cross-validation was run over all 600 projects to compare features and tune $\alpha \in [0.01, 100.0]$.
   - In `scripts/evaluate_holdout.py`, the holdout split was created using the exact same random seed (`seed=42`), making the holdout set identical to Fold 5.
   - Because the decision to select $\alpha = 10.0$ was based on the 5-fold average (which included Fold 5), the holdout set was not an untouched, sealed holdout.
2. **Global TF-IDF Vocabulary in Production:**
   - While cross-validation properly fitted vectorizers inside training folds, the production artifact `models/section_tfidf_extractor.joblib` was fitted on all 600 projects.
   - This is standard for serving a fixed reference database, but any evaluation against external datasets must fit vectorizers only on the reference set.

---

## 7. Claims Assessment

### Claims Currently Justified:
1. **Explainable Additive Attribution:** The system successfully decomposes a linear Ridge prediction into transparent, section-by-section and keyword-by-keyword evidence with exact mathematical reconstruction.
2. **Lexical Corpus Retrieval:** The system accurately identifies exact lexical overlaps, shared datasets, and identical technical terminology relative to a fixed 600-project reference corpus.
3. **Database Segregation:** New user proposal submissions are stored exclusively in SQLite and do not contaminate or mutate the historical baseline reference database.
4. **Reproducible Pipeline:** The feature extraction, scoring API, and SQLite database layers operate deterministically without runtime crashes.

### Claims NOT Yet Justified:
1. **"Detects Academic Novelty":** The system cannot detect conceptual, structural, or algorithmic originality. It only detects TF-IDF lexical overlap.
2. **"Trained on Expert Faculty Evaluations":** The ground-truth ratings are 100% synthetic heuristic rules. Not a single faculty evaluation has been recorded in the benchmark.
3. **"Robust Plagiarism / Duplicate Detector":** Paraphrased proposals completely evade detection ($99.6\%$ novelty). Exact copies cannot reach the duplicate threshold due to model under-prediction ceiling ($35.6\%$ novelty).
4. **"Balanced Section Weighting":** The model over-indexes on dataset names ($w = +2.2054$) and penalizes identical algorithms/tools ($w < 0$), distorting academic novelty intuition.

---

## 8. Prioritized Next Steps for Scientific Validity

### Priority 1: Acquire Real Faculty Annotations (Highest Priority)
- Distribute `data/raw/Faculty_Verification_DomainWise.xlsx` to department faculty panels.
- Obtain at least 200–300 independent, double-annotated pair ratings on real student proposals.
- Compute inter-annotator agreement (Cohen's Kappa / Krippendorff's Alpha) before model fitting.

### Priority 2: Constrain Coefficients to Be Non-Negative (Monotonicity Fix)
- Replace unconstrained Ridge regression with **Non-Negative Least Squares (NNLS)** or bounded Ridge regression ($w_i \ge 0$).
- Enforce that higher similarity in any section must never reduce overall predicted similarity.

### Priority 3: Integrate Dense Semantic Embeddings (Paraphrase Defense)
- Reintroduce hybrid dense semantic embeddings (e.g. `all-MiniLM-L6-v2` or `Specter2` for scientific papers) alongside TF-IDF.
- This ensures paraphrased proposals (Case 3) are caught through semantic cosine similarity rather than slipping through lexical filters.

### Priority 4: Re-scale or Normalize Model Outputs to $[0, 100\%]$
- Calibrate the model output so that an identical project ($X = \vec{1}$) maps to $4.0$ ($0\%$ novelty) and completely disjoint projects ($X = \vec{0}$) map to $0.0$ ($100\%$ novelty).
- Clearly document in the frontend that similarity is measured on the $[0, 4]$ rubric and display normalized percentages.

### Priority 5: True Sealed Holdout Protocol
- Partition an independent, untouched test set of real project pairs *before* any feature extraction or model tuning.
- Do not inspect test performance until the final model configuration is frozen.

---

## 9. Single Most Important Limitation

> [!CAUTION]
> **The Single Most Important Limitation:**  
> **100% of the similarity ground-truth labels are synthetic heuristic rules, and not a single faculty annotation has been collected.**  
> Consequently, the system is currently a demonstration of **synthetic rule fitting and transparent linear attribution**, rather than an empirically validated detector of real-world academic novelty. Until real faculty ratings are collected and evaluated on an untouched test split, all reported scores must be treated as illustrative prototype estimates.
