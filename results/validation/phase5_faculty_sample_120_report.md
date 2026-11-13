# Phase 5.2 Completion Report: Expansion of Faculty Manual Verification Dataset (120 Project Pairs)

**Document Reference**: `results/validation/phase5_faculty_sample_120_report.md`  
**Execution Date**: November 13, 2026  
**Status**: Completed / Ready for Double-Blind Human Faculty Verification  
**Primary Output Artifacts**:
- `data/annotation/faculty_evaluation_sample_120.xlsx` (4-sheet formatted Excel workbook)
- `data/annotation/faculty_evaluation_sample_120.csv` (combined 120-pair programmatic CSV)
- `data/annotation/faculty_evaluation_sample_120_master.csv` (research master file with ground truth metadata)

---

## 1. Executive Summary

As part of Phase 5 (Scientific Validation) of the Academic Project Novelty Detection system, the manual human verification sample has been expanded from 60 to **120 unique project pairs**.

To ensure manageable review workloads and facilitate double-blind inter-rater evaluation, the 120 pairs are structured into **two separate evaluation worksheets of 60 pairs each** (`Evaluation_Batch_1` and `Evaluation_Batch_2`). The dataset guarantees:
- **Zero Duplicates**: No project pair appears in both batches, and no pair is duplicated within either batch.
- **Stratified Balance**: Exactly 24 pairs per similarity level across the full cohort, with each batch containing exactly 12 pairs per level (0, 1, 2, 3, 4).
- **Controlled Benchmark Integration**: All 8 documented controlled benchmark test cases (verbatim copy, paraphrases, domain/problem divergences, cross-domain tools) are preserved and evenly distributed (4 in Batch 1, 4 in Batch 2).
- **Double-Blind Safeguards**: Faculty-facing sheets contain **no** synthetic similarity labels, category names, selection rationales, or model prediction scores.
- **Side-by-Side Comparison**: All 8 canonical project sections are grouped side by side for direct comparative evaluation.
- **Professional Formatting**: Excel features include dark headers, freeze panes, auto-filters, text wrapping, zebra striping, light yellow reviewer inputs, light green adjudication fields, light orange pending status, and 0–4 ordinal conditional formatting.
- **Preservation of Baseline Artifacts**: The original 60-pair verification sample (`faculty_evaluation_sample_60.xlsx` and `faculty_evaluation_sample_60.csv`) remains intact and unmodified.

---

## 2. Sample Distribution & Stratification

### 2.1 Similarity Level Distribution

| Ordinal Similarity Level | Category Description | Batch 1 Count | Batch 2 Count | Total Pairs | Target Requirement | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Level 0** | Not Similar (Unrelated) | 12 | 12 | **24** | 24 | Satisfied |
| **Level 1** | Slightly Similar (Jargon/Tools) | 12 | 12 | **24** | 24 | Satisfied |
| **Level 2** | Moderately Similar (Shared Component) | 12 | 12 | **24** | 24 | Satisfied |
| **Level 3** | Highly Similar (Core Paraphrase) | 12 | 12 | **24** | 24 | Satisfied |
| **Level 4** | Nearly Duplicate / Duplicate | 12 | 12 | **24** | 24 | Satisfied |
| **Total** | **All Levels (0 – 4)** | **60** | **60** | **120** | **120** | **Satisfied** |

### 2.2 Calibration vs. Held-Out Evaluation Subsets

To preserve the experimental design of Phase 5 while eliminating data leakage:
- **Development / Calibration Set (`DEV_CALIBRATION`)**: 30 pairs total (15 in Batch 1, 15 in Batch 2; 3 pairs per similarity level per batch). Used for initial reviewer alignment and pilot review.
- **Held-Out Evaluation Set (`HELD_OUT_EVALUATION`)**: 90 pairs total (45 in Batch 1, 45 in Batch 2; 9 pairs per similarity level per batch). Kept strictly held-out for double-blind human consensus scoring and final model benchmark validation.

```
+-------------------------------------------------------------------------+
|                        TOTAL SAMPLE (120 Pairs)                         |
+------------------------------------+------------------------------------+
|     Evaluation_Batch_1 (60 Pairs)  |     Evaluation_Batch_2 (60 Pairs)  |
+------------------+-----------------+------------------+-----------------+
| Dev / Calib (15) | Held-Out (45)   | Dev / Calib (15) | Held-Out (45)   |
| (3 per level 0-4)| (9 per level 0-4)| (3 per level 0-4)| (9 per level 0-4)|
+------------------+-----------------+------------------+-----------------+
```

### 2.3 Balanced Distribution of Controlled Benchmark Test Cases

All 8 documented controlled test cases from `scripts/evaluate_semantic_feasibility.py` are integrated and distributed across the two batches:

| Case ID | Candidate ID | Level | Subset | Batch Assignment | Benchmark Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `CASE_A_EXACT_COPY` | `TEST-CASE-A` | 4 | `DEV_CALIBRATION` | **Batch 1** | Verbatim copy across all 8 text sections |
| `CASE_D_SAME_DOMAIN_DIFF_PROBLEM` | `TEST-CASE-D` | 1 | `DEV_CALIBRATION` | **Batch 1** | Same domain (AI/ML), completely different problem (fraud vs crops) |
| `CASE_E_DIFF_DOMAIN_COMMON_TECH` | `TEST-CASE-E` | 0 | `DEV_CALIBRATION` | **Batch 1** | Cross-domain project sharing only common technology (SDN vs crop disease) |
| `CASE_F_SAME_DATASET_DIFF_OBJECTIVES` | `TEST-CASE-F` | 2 | `HELD_OUT_EVALUATION` | **Batch 1** | Shared PlantVillage dataset with divergent goal (inpainting vs diagnosis) |
| `CASE_B_LIGHT_PARAPHRASE` | `TEST-CASE-B` | 4 | `HELD_OUT_EVALUATION` | **Batch 2** | Verbatim copy with light grammatical & lexical edits |
| `CASE_C_STRONG_PARAPHRASE` | `TEST-CASE-C` | 3 | `HELD_OUT_EVALUATION` | **Batch 2** | Deep conceptual paraphrase with disjoint vocabulary |
| `CASE_G_SAME_ALGO_DIFF_APPLICATION` | `TEST-CASE-G` | 1 | `DEV_CALIBRATION` | **Batch 2** | Same algorithm (CNN), different domain (brain tumor vs crops) |
| `CASE_H_SUPERFICIAL_WORDING_DIFF_OBJECTIVES` | `TEST-CASE-H` | 0 | `DEV_CALIBRATION` | **Batch 2** | Superficial keyword overlap with socio-legal policy objective |

---

## 3. Double-Blind Protocol & Reviewer Bias Prevention

To ensure scientific validity, the faculty-facing evaluation sheets adhere strictly to double-blind guardrails:
1. **No Synthetic Labels**: The `provisional_synthetic_label` (0–4) column is completely omitted from `Evaluation_Batch_1` and `Evaluation_Batch_2`.
2. **No Provisional Categories**: Category names (e.g. *"Not Similar"*, *"Nearly Duplicate"*) are omitted.
3. **No Selection Rationale**: Notes indicating whether a pair is a controlled benchmark, exact copy, or paraphrase are omitted.
4. **No Automated Scores**: Neither TF-IDF cosine, semantic dense embeddings, nor hybrid fusion model scores appear anywhere in the faculty sheets.
5. **No Ordering Clues**: Within each batch, pairs are shuffled deterministically with a random seed so that similarity levels are non-sequential. Reviewers cannot infer similarity ratings based on row position.
6. **Isolated Research Master File**: Research metadata is stored exclusively in `data/annotation/faculty_evaluation_sample_120_master.csv` for post-adjudication statistical validation.

---

## 4. Excel Workbook Structure and Visual Formatting

The Excel workbook `data/annotation/faculty_evaluation_sample_120.xlsx` contains four sheets:

### 4.1 Evaluation Sheets (`Evaluation_Batch_1` & `Evaluation_Batch_2`)
Each sheet contains 60 evaluation rows and 28 columns:

- **Identification Metadata (Cols A–E, Navy Header `#1E293B`)**:
  - `pair_id` (`EVAL-001` to `EVAL-060` in Batch 1; `EVAL-061` to `EVAL-120` in Batch 2)
  - `project_candidate_id`
  - `project_reference_id`
  - `domain_candidate`
  - `domain_reference`
- **Side-by-Side Project Sections (Cols F–U, Deep Blue `#1E3A8A` & Charcoal `#374151`)**:
  - `candidate_title` | `reference_title`
  - `candidate_abstract` | `reference_abstract`
  - `candidate_problem` | `reference_problem`
  - `candidate_objectives` | `reference_objectives`
  - `candidate_methodology` | `reference_methodology`
  - `candidate_dataset` | `reference_dataset`
  - `candidate_algorithm` | `reference_algorithm`
  - `candidate_technology` | `reference_technology`
- **Reviewer 1 Input (Cols V–W, Amber Header `#854D0E`, Light Yellow Cells `#FEF9C3` / `#FEF08A`)**:
  - `reviewer_1_rating` (empty)
  - `reviewer_1_comments` (empty)
- **Reviewer 2 Input (Cols X–Y, Amber Header `#854D0E`, Light Yellow Cells `#FEF9C3` / `#FEF08A`)**:
  - `reviewer_2_rating` (empty)
  - `reviewer_2_comments` (empty)
- **Adjudication Section (Cols Z–AA, Forest Green Header `#14532D`, Light Green Cells `#F0FDF4` / `#DCFCE7`)**:
  - `adjudicated_rating` (empty)
  - `adjudication_comments` (empty)
- **Annotation Status (Col AB, Slate Header `#1E293B`, Light Orange Cells `#FFEDD5`)**:
  - `annotation_status` (initialized to `"PENDING_HUMAN_REVIEW"`)

### 4.2 Formatting Specifications
- **Header Row**: White bold text on dark backgrounds; frozen via `freeze_panes = "A2"`.
- **Auto-Filters**: Enabled across all 28 columns (`ws.auto_filter.ref = "A1:AB61"`).
- **Text Alignment**: Top-aligned vertically with text wrapping enabled on all descriptive sections.
- **Zebra Striping**: Alternating `#FFFFFF` and `#F8FAFC` for high readability.
- **Section Dividers**: Medium-weight vertical borders separate metadata, project sections, reviewer sections, and adjudication.
- **Conditional Formatting for Ratings (0 – 4 Scale)**:
  - Rating `0`: Soft Slate (`#F1F5F9`, dark text `#334155`)
  - Rating `1`: Pale Cyan (`#CFFAFE`, dark text `#0E7490`)
  - Rating `2`: Light Yellow (`#FEF08A`, dark text `#854D0E`)
  - Rating `3`: Light Orange (`#FED7AA`, dark text `#C2410C`)
  - Rating `4`: Soft Coral/Rose (`#FECACA`, dark text `#B91C1C`)
- **Conditional Formatting for Status**:
  - `PENDING_HUMAN_REVIEW`: Light orange fill (`#FFEDD5`, text `#9A3412`)
  - `COMPLETED` / `ADJUDICATED_FINAL`: Light green fill (`#DCFCE7`, text `#166534`)
  - `REQUIRES_DISCUSSION`: Light red fill (`#FEE2E2`, text `#991B1B`)

### 4.3 Supporting Sheets
- **`Rating_Rubric`**: Contains the full 0–4 ordinal rating rubric with operational definitions, primary indicators, and actionable guidance for faculty.
- **`Instructions`**: Provides reviewers with step-by-step guidance, visual color legends, principles for disentangling lexical overlap from conceptual similarity, edge case advice, and double-blind protocol reminders.

---

## 5. Artifacts Created & Modified

| File Path | Action | Description |
| :--- | :---: | :--- |
| `scripts/generate_faculty_sample.py` | Modified | Updated generator supporting 120-pair dual-batch output, professional formatting, and reproducible sampling |
| `data/annotation/faculty_evaluation_sample_120.xlsx` | Created | Faculty-facing 4-sheet evaluation workbook (blind, formatted) |
| `data/annotation/faculty_evaluation_sample_120.csv` | Created | Programmatic 120-pair CSV with `batch` identifier column |
| `data/annotation/faculty_evaluation_sample_120_master.csv` | Created | Research master CSV with ground-truth labels and sampling rationales |
| `data/annotation/faculty_evaluation_sample_60.xlsx` | Preserved | Original 60-pair verification workbook (intact, untouched) |
| `data/annotation/faculty_evaluation_sample_60.csv` | Preserved | Original 60-pair verification CSV (intact, untouched) |
| `docs/FACULTY_ANNOTATION_GUIDELINES.md` | Modified | Updated guidelines (v2.0.0) reflecting 120 pairs across two batches |
| `docs/PHASE5_SCIENTIFIC_EVALUATION_PLAN.md` | Modified | Updated scientific validation plan reflecting 120 pairs (30 Dev, 90 Eval) |
| `tests/test_faculty_sample_120.py` | Created | Automated test suite (17 comprehensive test cases) |
| `results/validation/phase5_faculty_sample_120_report.md` | Created | This formal completion and verification report |

---

## 6. Verification & Test Execution Results

The test suite was executed via pytest:

```
tests/test_api.py ......................... [10 passed]
tests/test_candidate_model.py ............. [9 passed]
tests/test_faculty_sample_120.py .......... [17 passed]
tests/test_hybrid_scoring.py .............. [16 passed]
tests/test_scientific_reliability.py ...... [10 passed]
tests/test_storage.py ..................... [14 passed]
tests/test_xai.py ......................... [15 passed]
======================= 91 passed in 31.37s =======================
```

### Specific Validations Passed by `tests/test_faculty_sample_120.py`:
1. `test_sample_120_files_exist`: Confirms all 120-pair files exist and are non-empty.
2. `test_original_60_pair_files_preserved`: Confirms original 60-pair sample was not overwritten.
3. `test_excel_sheet_names_and_count`: Confirms presence of `Evaluation_Batch_1`, `Evaluation_Batch_2`, `Rating_Rubric`, and `Instructions`.
4. `test_batch_sizes_in_excel`: Confirms each evaluation sheet has exactly 60 data rows.
5. `test_csv_batch_counts_and_structure`: Confirms 120 rows in CSV with 60 pairs per batch.
6. `test_pair_ids_unique_and_monotonic`: Confirms sequential, unique IDs `EVAL-001` through `EVAL-120`.
7. `test_no_duplicate_pairs_across_or_within_batches`: Confirms zero duplicate pairs (directed and undirected).
8. `test_faculty_sheets_contain_no_biasing_columns`: Confirms zero synthetic labels or model scores in Excel.
9. `test_combined_csv_contains_no_biasing_columns`: Confirms combined CSV is strictly blind.
10. `test_reviewer_ratings_are_unannotated_and_empty`: Confirms all rating columns are blank.
11. `test_annotation_status_is_pending_across_all_rows`: Confirms status is `'PENDING_HUMAN_REVIEW'`.
12. `test_all_eight_sections_present_and_non_empty`: Confirms all 8 project sections are populated.
13. `test_similarity_levels_stratified_in_master`: Confirms 24 pairs per level (12 per batch).
14. `test_development_and_held_out_subsets_preserved`: Confirms 15 Dev and 45 Eval pairs per batch.
15. `test_all_eight_controlled_cases_included`: Confirms all 8 controlled benchmarks are included (4 per batch).
16. `test_excel_formatting_features`: Confirms freeze panes, auto-filters, and conditional formatting.
17. `test_reproducibility_with_seed`: Confirms deterministic reproducibility with seed 42.

---

## 7. Assumptions, Governance, and Next Steps

1. **Faculty Reviewer Assignment**:
   - `Evaluation_Batch_1` (Pairs `EVAL-001` to `EVAL-060`) and `Evaluation_Batch_2` (Pairs `EVAL-061` to `EVAL-120`) can be assigned to two independent reviewer pairs, or sequentially to a single reviewer pair.
2. **Zero-Annotation Compliance**:
   - In strict compliance with the Phase 5 protocol, no empirical validation metrics (MAE, RMSE, Cohen's Kappa, Pearson $r$) are reported at this stage. Ratings must be independently completed by human faculty before running evaluation pipelines.
3. **Distribution of Workbooks**:
   - Distribute `data/annotation/faculty_evaluation_sample_120.xlsx` to participating faculty alongside `docs/FACULTY_ANNOTATION_GUIDELINES.md`. Reviewers should review the embedded `Instructions` and `Rating_Rubric` sheets prior to beginning annotations.

