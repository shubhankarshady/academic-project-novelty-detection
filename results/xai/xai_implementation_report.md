# Explainable AI (XAI) Implementation & Quality Review Report

**Project:** `academic-project-novelty-detection`  
**Phase:** Explainable AI (XAI) for Academic Proposal Novelty Detection — Quality Review  
**Primary Candidate:** Ridge Regression with 8-Section TF-IDF Similarities  
**Date:** October 22, 2026  
**Status:** Completed, Verified, and Quality Reviewed  

---

## 1. Executive Summary

This report documents the design, implementation, and quality review of the Explainable AI (XAI) engine for the Academic Project Novelty Detection System.
The objective of this module is to transform opaque numeric similarity and novelty predictions into transparent, evidence-based explanations that academic evaluators can inspect, verify, and critique.

Following the initial XAI implementation, a focused quality review was conducted to resolve precision, wording, and robustness issues before frontend development:
1. **Corpus-Relative Novelty Wording:** Replaced misleading phrasing such as `"High Novelty (Original Work)"` with `"High estimated novelty relative to the current database"`. Explicitly emphasized in reports and disclaimers that the score is corpus-relative and does not establish academic originality or prove plagiarism.
2. **High-Information Keyword Filtering:** Eliminated generic academic filler words (such as `"using"`, `"based"`, `"proposed"`, `"project"`, and `"mini project"`) from evidence extraction using the fitted TF-IDF vocabulary combined with stop-word and low-information filtering. Legitimate domain and technical terms (e.g., algorithm names, datasets, and frameworks like `"U-Net"`, `"BraTS"`, `"PyTorch"`, `"CNN"`, `"Solidity"`) are strictly preserved. When no informative shared terms exist, an empty list is returned with an explanatory message rather than inventing evidence.
3. **Word-Boundary-Aware Snippets:** Fixed context window slicing so that snippets never start or end mid-word. Verbatim source text wording is preserved, ellipses (`...`) are added if and only if text was omitted before or after, and every snippet is verified against the original source text.
4. **Scoring Correctness & Non-Causal Attribution:** Verified that the intercept plus signed section contributions reconstructs the raw Ridge prediction. Distinctly tracked raw predictions from predictions clipped to the supported $[0.0, 4.0]$ scale. Described model contributions strictly as linear statistical feature associations rather than causal explanations.
5. **Deterministic Historical Matching & Self-Exclusion:** Implemented self-exclusion covering project IDs, exact titles, and identical section content, and enforced deterministic ranking using tie-breaking (`predicted_similarity` descending, `raw_prediction` descending, `project_id` ascending).
6. **Full Test Suite:** Expanded the automated test suite from 7 tests to 15 comprehensive unit and integration tests, all passing.

All changes were implemented without retraining, overwriting, or mutating any existing model checkpoints, raw datasets, or holdout evaluation outputs.

---

## 2. Architecture & File Inventory

The following files constitute the XAI subsystem:

| File / Component | Path | Description |
|---|---|---|
| **XAI Package Init** | [`src/explainability/__init__.py`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/src/explainability/__init__.py) | Package initialization exposing `NoveltyExplainer` |
| **XAI Explainer Engine** | [`src/explainability/explainer.py`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/src/explainability/explainer.py) | Core engine for section ranking, word-boundary snippet extraction, informative keyword filtering, and additive attribution |
| **Upgraded Scoring CLI** | [`scripts/score_novelty.py`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/scripts/score_novelty.py) | CLI with `--explain`, `--json_output`, `--model_type`, and self-exclusion support |
| **Novelty Metrics Utility** | [`src/utils/metrics.py`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/src/utils/metrics.py) | Central metric definitions and updated corpus-relative novelty tiers |
| **XAI Unit & Integration Tests** | [`tests/test_xai.py`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/tests/test_xai.py) | 15 comprehensive automated tests covering all edge cases and quality improvements |
| **XAI Implementation Report** | [`results/xai/xai_implementation_report.md`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/xai/xai_implementation_report.md) | Technical report and quality review documentation |

---

## 3. Explanation Fields Returned by the Engine

The engine produces both human-readable terminal reports and structured JSON output. The JSON schema contains:

### Proposal-Level Summary Fields:
- `candidate_proposal_title`: Submitted project title.
- `candidate_project_id`: Unique proposal identifier (or `NEW_PROPOSAL`).
- `corpus_size_evaluated`: Total historical projects evaluated (excluding self-matches).
- `max_similarity_score`: Highest predicted similarity score ($[0.0, 4.0]$) among all historical projects.
- `overall_novelty_score_pct`: $100 \times (1 - \text{max\_similarity} / 4.0)$ ($[0.0\%, 100.0\%]$).
- `novelty_assessment_tier`: Human-readable category:
  - $\ge 75\%$: *High estimated novelty relative to the current database*
  - $\ge 50\%$: *Moderate Novelty (Partial Overlap)*
  - $\ge 25\%$: *Low Novelty (Significant Prior Work)*
  - $< 25\%$: *Very Low Novelty / Potential Duplicate*
- `top_matches`: Ordered list of top $K$ historical project matches.
- `framework_disclaimers`: Formal methodological notices clarifying corpus relativity, heuristic thresholds, synthetic labels, and non-causal attribution.

### Match-Level Detailed Breakdown (for each top match):
1. **Metadata:** `historical_project_id`, `historical_project_title`, `historical_domain`.
2. **Predicted Scores:**
   - `predicted_similarity`: Model prediction clipped to $[0.0, 4.0]$.
   - `raw_similarity_prediction`: Unclipped raw Ridge regression output.
   - `clipped_prediction`: Explicit clipped score for UI bounding.
   - `novelty_score_pct`: Percentage novelty score based on clipped similarity.
   - `novelty_category`: Risk level category.
3. **Additive Model Attribution:**
   - `intercept`: Model bias term ($b = -0.0706$).
   - `sum_of_contributions`: $\sum_{s=1}^8 (\text{sim}_s \times w_s)$.
   - `reconstructed_raw_prediction`: $\text{intercept} + \sum \text{contributions}$ (verified identical to `raw_similarity_prediction`).
4. **Section Ranking (`ranked_sections`):**
   List of sections sorted descending by raw lexical cosine similarity:
   - `section`: Name (`dataset`, `abstract`, `title`, etc.)
   - `raw_similarity`: Cosine similarity score in $[0.0, 1.0]$.
   - `model_weight`: Fitted Ridge coefficient $w_s$.
   - `signed_contribution`: Points contributed toward predicted similarity ($\text{sim}_s \times w_s$).
5. **Lexical Keyword & Snippet Evidence (`section_evidence`):**
   For each section:
   - `shared_terms`: High-information n-grams appearing with whole-word boundaries in both source texts (no generic filler).
   - `term_cosine_contribution`: Term's mathematical share of the section cosine similarity.
   - `candidate_snippet`: Word-boundary-aware context window centered on the term from candidate text.
   - `historical_snippet`: Word-boundary-aware context window centered on the term from historical text.
   - `message`: Explanatory message (`"Shared lexical evidence identified."` or `"No informative shared terms identified for this section."`).
6. **Transparent Methodology Disclaimers:**
   Explicit notes on lexical limitations, non-causal linear attribution, and synthetic benchmark ground truth.

---

## 4. Test Suite & Verification Results

The automated test suite in [`tests/test_xai.py`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/tests/test_xai.py) was expanded to 15 unit and integration tests using Python's standard `unittest` framework:

```
Ran 15 tests in 3.051s
OK
```

### Detailed Test Coverage:

| # | Test Case | Description | Result |
|---|---|---|:---:|
| 1 | `test_exact_keyword_evidence_from_source_text` | Asserts `term in candidate_text.lower()` and `term in historical_text.lower()` for every returned term. Prevents synthetic hallucination of keywords. | **PASSED** |
| 2 | `test_correct_section_to_score_mapping` | Asserts all 8 sections are present, correctly mapped to feature names, and bounded in $[0.0, 1.0]$. | **PASSED** |
| 3 | `test_contribution_sum_matches_ridge_prediction` | Asserts that $\text{intercept} + \sum \text{contributions} == \text{raw\_prediction}$ within floating-point tolerance. | **PASSED** |
| 4 | `test_empty_or_missing_sections` | Tests proposals with empty strings across sections. Asserts 0.0 scores, 0.0 contributions, empty keyword lists, and graceful degradation. | **PASSED** |
| 5 | `test_no_self_matching` | Submits a proposal identical to an existing corpus project. Asserts the project is excluded from historical ranking and corpus count decrements by 1. | **PASSED** |
| 6 | `test_deterministic_ranking` | Runs the pipeline twice on the same candidate. Asserts identical rankings, similarities, and contributions across calls. | **PASSED** |
| 7 | `test_correct_json_output_from_scoring_cli` | Executes `scripts/score_novelty.py --json_output` via subprocess. Asserts valid JSON, required top-level keys, and nested explanation schema. | **PASSED** |
| 8 | `test_generic_keyword_filtering` | Verifies that generic filler terms such as `"using"`, `"based"`, `"proposed"`, `"project"`, and `"using cnn"` are rejected as meaningful evidence. | **PASSED** |
| 9 | `test_technical_terms_preserved` | Verifies that legitimate technical terms like `"U-Net"`, `"BraTS"`, `"PyTorch"`, `"CNN"`, `"Solidity"`, `"Ethereum"`, and `"ROS"` are preserved. | **PASSED** |
| 10 | `test_snippet_word_boundaries` | Verifies that extracted snippets never start or end mid-word (no partial words). | **PASSED** |
| 11 | `test_empty_evidence_with_explanatory_message` | Verifies that when sections share no informative terms, an empty list is returned with an explanatory message rather than invented evidence. | **PASSED** |
| 12 | `test_source_text_verification` | Verifies that every extracted snippet's body is an exact substring of the source text and that ellipses are present if and only if text was omitted. | **PASSED** |
| 13 | `test_score_reconstruction_and_clipping` | Verifies that `intercept + sum(contributions)` reconstructs the raw Ridge prediction, and that raw predictions are kept distinct from clipped $[0, 4]$ values. | **PASSED** |
| 14 | `test_self_exclusion_comprehensive` | Verifies self-exclusion by project ID, by exact title, and by identical text across sections. | **PASSED** |
| 15 | `test_novelty_wording_and_disclaimers` | Verifies that score $\ge 75\%$ returns `"High estimated novelty relative to the current database"` and disclaimers state the score is corpus-relative and does not prove plagiarism. | **PASSED** |

---

## 5. Sample Explanation Demonstration

Below is an authentic explanation generated from the CLI for a test proposal after the quality review enhancements:

**Candidate Proposal:**  
*Title:* "Federated Learning for Privacy-Preserving Medical Diagnostics"  
*Abstract:* "We propose a federated learning framework using convolutional neural networks to train diagnostic models across distributed hospital datasets without centralizing sensitive patient records."

### Generated Output:
```
==============================================================================
         ACADEMIC PROJECT NOVELTY INSPECTION & XAI EXPLANATION REPORT
==============================================================================
Candidate Title       : Federated Learning for Privacy-Preserving Medical Diagnostics
Reference Corpus Size : 600 historical projects
Overall Novelty Score : 98.7%
Novelty Assessment    : High estimated novelty relative to the current database
Max Similarity Score  : 0.05 / 4.00
Notice                : Estimated relative to the current database.
                        Does not establish originality or prove plagiarism.
==============================================================================

Top 2 Most Similar Prior Projects in Corpus:
------------------------------------------------------------------------------

[Rank 1] Project ID: SYN-0584 | Domain: Education Technology
  Title               : Privacy-Aware Decentralized Identity Verification Using Hyperledger Fabric
  Predicted Similarity: 0.05 / 4.00 (Raw: 0.05) | Novelty: 98.7% (High estimated novelty relative to the current database)
  Additive Model Attribution : Intercept (-0.071) + Contributions (+0.124) = Raw 0.05 (Clipped: 0.05)
  Section Overlap Ranking:
    - objectives  : Lexical Sim = 0.325 | Model Weight = +0.237 | Score Contrib = +0.077
    - title       : Lexical Sim = 0.184 | Model Weight = +0.134 | Score Contrib = +0.025
    - problem     : Lexical Sim = 0.150 | Model Weight = -0.126 | Score Contrib = -0.019
    - abstract    : Lexical Sim = 0.112 | Model Weight = +0.398 | Score Contrib = +0.045
    - methodology : Lexical Sim = 0.083 | Model Weight = +0.010 | Score Contrib = +0.001
    - technology  : Lexical Sim = 0.029 | Model Weight = -0.146 | Score Contrib = -0.004
    - dataset     : Lexical Sim = 0.000 | Model Weight = +2.205 | Score Contrib = +0.000
    - algorithm   : Lexical Sim = 0.000 | Model Weight = -0.065 | Score Contrib = -0.000
  Key Lexical Evidence (Shared N-Grams):
    - Title      : 'privacy' (sim +0.184)
        Candidate : "Federated Learning for Privacy-Preserving Medical Diagnostics"
        Historical: "Privacy-Aware Decentralized Identity Verification Using..."
    - Abstract   : 'records' (sim +0.104)
        Candidate : "...without centralizing sensitive patient records."
        Historical: "...Fabric to analyze synthetic identity records and generate useful predictions or classifications...."
    - Problem    : 'data' (sim +0.150)
        Candidate : "Data privacy regulations and GDPR restrict sharing..."
        Historical: "...and may provide inconsistent results when data volume or operational conditions increase. The..."
    - Objectives : 'decentralized' (sim +0.325)
        Candidate : "Train decentralized models while maintaining patient..."
        Historical: "...relevant data; 2) develop a model for decentralized identity verification; 3) evaluate the model..."
    - Methodology: 'framework' (sim +0.083)
        Candidate : "Deploy Flower federated learning framework with PyTorch."
        Historical: "Design and evaluate a research framework with baseline and proposed models, hyperparameter..."
    - Technology : 'python' (sim +0.028)
        Candidate : "Python, PyTorch, Flower, CUDA."
        Historical: "Python, scikit-learn, React"
------------------------------------------------------------------------------

Methodology Disclaimers (Corpus-Relative Heuristic - Does Not Establish Originality or Prove Plagiarism):
  - Novelty score is an estimated heuristic relative to the current database, not proof of absolute academic originality, and does not prove plagiarism.
  - Novelty assessment tiers are heuristic benchmark rules relative to the indexed database and require faculty panel review and domain-specific validation.
  - All benchmark similarity ratings in this system originate from synthetic heuristic labels, not verified faculty adjudications.
  - Linear model feature attributions represent statistical associations rather than causal evidence.
==============================================================================
```

Noticeable quality improvements:
- Generic word `"using"` is cleanly excluded from the abstract evidence even though both texts contain the word.
- Snippets start and end at full word boundaries (e.g., `"...without centralizing sensitive patient records."` rather than cut-off fragments like `"...asets..."` or `"classific..."`).
- The report explicitly labels model attribution as non-causal linear contributions and states that the score is corpus-relative and does not prove plagiarism.

---

## 6. Known Limitations & Communication Caveats

1. **Corpus-Relative Novelty:** Novelty scores are calculated relative to the currently indexed reference corpus (600 projects). A proposal scoring 90% novelty is novel relative to this specific database, but could duplicate external literature outside the repository. It does not establish absolute academic originality or prove plagiarism.
2. **Synthetic Ground Truth:** All similarity ratings ($0$ to $4$) originate from heuristic synthetic benchmarks. These metrics demonstrate the internal mathematical consistency, feature relevance, and generalization capability of the TF-IDF Ridge pipeline, but **do not represent real-world faculty-validated novelty detection**.
3. **Lexical Blind Spots (TF-IDF):** TF-IDF captures exact word and phrase overlap. High section similarity indicates shared vocabulary. However, low lexical overlap may miss paraphrased or conceptually related work using different terminology (e.g., "deep learning" vs "artificial neural network").
4. **Non-Causal Statistical Attribution:** Model contributions ($X_s \times w_s$) are linear statistical feature associations learned by Ridge regression. They must **not** be presented as causal explanations of academic misconduct or plagiarism.
5. **Threshold Validation Required:** The novelty tiers (High $\ge 75\%$, Moderate $\ge 50\%$, Low $\ge 25\%$, Potential Duplicate $< 25\%$) are benchmark heuristics and require formal validation by an academic faculty committee.
