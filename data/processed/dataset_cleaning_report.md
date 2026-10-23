# Dataset Cleaning & Quality Validation Report

**Dataset:** Academic Project Similarity Benchmark  
**Date:** October 22, 2026  
**Cleaned Output File:** `data/processed/similarity_pairs_clean.csv`  
**Manifest:** `data/processed/dataset_cleaning_manifest.json`  

---

## 1. Executive Summary

An audit and deduplication of the academic project similarity pair dataset was completed.
The original raw dataset (`data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx`) was left completely untouched.
A cleaned pair dataset of **2,996 rows** was produced by removing exactly **4 redundant rows** (1 exact duplicate and 3 reversed duplicate pairs).

> [!WARNING]
> **Synthetic Label Limitation:**
> All 3,000 original pairs carry synthetic benchmark labels (`annotation_note: "SYNTHETIC BENCHMARK LABEL — must be replaced/validated by faculty annotation before publication."`).
> Evaluation metrics obtained on this dataset reflect algorithmic consistency and synthetic benchmark performance, **not** real-world faculty-validated novelty detection.

---

## 2. Row Count Summary

| Metric | Count | Percentage |
|---|:---:|:---:|
| Original Rows | **3,000** | 100.00% |
| Removed Rows | **4** | 0.13% |
| **Cleaned Rows** | **2,996** | **99.87%** |

---

## 3. Detailed Audit of Removed Rows

| Pair ID Removed | Project A | Project B | Label | Category | Retained Counterpart | Removal Reason |
|---|---|---|:---:|---|---|---|
| `PAIR-1819` | SYN-0524 | SYN-0451 | 0 | Not Similar | `PAIR-1324` | Exact duplicate pair (identical direction and projects) |
| `PAIR-2023` | SYN-0130 | SYN-0097 | 0 | Not Similar | `PAIR-0432` | Reversed duplicate of PAIR-0432 (SYN-0097 vs SYN-0130) |
| `PAIR-2195` | SYN-0165 | SYN-0249 | 0 | Not Similar | `PAIR-1401` | Reversed duplicate of PAIR-1401 (SYN-0249 vs SYN-0165) |
| `PAIR-2957` | SYN-0538 | SYN-0174 | 0 | Not Similar | `PAIR-2588` | Reversed duplicate of PAIR-2588 (SYN-0174 vs SYN-0538) |

### Label Consistency Verification
Prior to removal, all equivalent pairs were evaluated for label concordance.
- `PAIR-1324` vs `PAIR-1819`: Both label `0` (`Not Similar`), proxy `0.7838`.
- `PAIR-0432` vs `PAIR-2023`: Both label `0` (`Not Similar`), proxy `0.7632`.
- `PAIR-1401` vs `PAIR-2195`: Both label `0` (`Not Similar`), proxy `0.7436`.
- `PAIR-2588` vs `PAIR-2957`: Both label `0` (`Not Similar`), proxy `0.6829`.

**Conclusion:** 100% agreement. No label ambiguity or contradictory annotations exist.

---

## 4. Label Distribution (Before vs After)

| Similarity Label | Category | Before Cleaning | % (Before) | After Cleaning | % (After) | Change |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **0** | Not Similar | 1,336 | 44.53% | 1,332 | 44.46% | -4 |
| **1** | Slightly Similar | 334 | 11.13% | 334 | 11.15% | 0 |
| **2** | Moderately Similar | 514 | 17.13% | 514 | 17.16% | 0 |
| **3** | Highly Similar | 581 | 19.37% | 581 | 19.39% | 0 |
| **4** | Nearly Duplicate | 235 | 7.83% | 235 | 7.84% | 0 |
| **Total** | | **3,000** | **100.00%** | **2,996** | **100.00%** | **-4** |

---

## 5. Dataset Quality Validation Checks

| Validation Check | Expected | Observed | Status |
|---|:---:|:---:|:---:|
| Row Count | 2,996 | 2,996 | **PASSED** |
| Missing / NaN values | 0 | 0 | **PASSED** |
| Self-pairs ($A == B$) | 0 | 0 | **PASSED** |
| Undirected duplicate pairs | 0 | 0 | **PASSED** |
| Invalid / Dangling Project IDs | 0 | 0 | **PASSED** |
| Project ID space coverage | 600 unique projects | 600 unique projects | **PASSED** |
| Target label range | [0, 4] integers | [0, 4] integers | **PASSED** |

---

## 6. Cross-Validation Compatibility

The cleaned dataset was verified with the project-disjoint cross-validation protocol:
- **Project Disjointness:** Splits are made at the unique project level (600 projects).
- **Leakage Prevention:** Cross-boundary pairs (where project A is in train and project B is in test) are discarded from both folds, ensuring zero data leakage.
- **TF-IDF Fitting:** TF-IDF vectorizers must be fitted strictly inside training folds, not globally across the entire corpus.
