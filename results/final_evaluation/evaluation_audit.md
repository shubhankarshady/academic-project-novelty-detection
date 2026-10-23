# Focused Audit of the Final Holdout Evaluation

**Project:** `academic-project-novelty-detection`  
**Date:** October 22, 2026  
**Audited Artifacts:**
- `scripts/evaluate_holdout.py`
- `results/final_evaluation/final_evaluation_report.md`
- `results/final_evaluation/holdout_metrics.json`
- `results/final_evaluation/holdout_predictions.csv`
- `results/final_evaluation/split_manifest.json`
- `results/model_optimization/fold_results.csv`
- `results/model_optimization/summary.csv`

---

## 1. Executive Summary & Audit Conclusions

This audit independently investigated the integrity, leakage safety, reproducibility, and partition history of the final holdout evaluation.

### Key Audit Findings:
1. **Mathematical Identity with Fold 5:** The holdout evaluation score ($\text{RMSE} = 0.7734$) is mathematically identical to Fold 5 of the 5-fold cross-validation ($\text{RMSE} = 0.773352$). This is because `KFold(n_splits=5, shuffle=True, random_state=42)` in `optimize_models.py` and `RandomState(42).shuffle()` in `evaluate_holdout.py` partition the sorted project list into identical index slices, placing the exact same 120 projects (`shuffled[480:600]`) into the test set.
2. **Explanation of Discrepancy with 5-Fold Summary:** The earlier 5-fold cross-validation summary reported $\text{RMSE} = 0.6511 \pm 0.0910$, which is the **unweighted arithmetic mean** across all 5 folds. The individual fold RMSEs ranged widely:
   - Fold 1: $0.5854$ (117 test pairs)
   - Fold 2: $0.6035$ (100 test pairs)
   - Fold 3: $0.5707$ (127 test pairs)
   - Fold 4: $0.7227$ (123 test pairs)
   - Fold 5: $0.7734$ (120 test pairs)  
   Fold 5 is the most challenging fold (the maximum of the distribution), which fully accounts for why the holdout RMSE ($0.7734$) is higher than the 5-fold mean ($0.6511$).
3. **Partition Timing & "Untouched" Status Disclosure:** Because 5-fold cross-validation was run across all 600 projects to compare feature sets and tune $\alpha \in [0.01, 0.1, 1.0, 10.0, 100.0]$ *before* `evaluate_holdout.py` was executed, the 120 projects in Fold 5 were part of the tuning decision to select $\alpha = 10.0$. Therefore, while test pair labels were strictly isolated from Ridge weight fitting, this partition was **not an a priori isolated holdout** in the strictest sense.
4. **Data Leakage Verification:** All pair construction, project partitioning, and TF-IDF fitting checks passed with **zero data leakage**.
5. **Exact Metric Reproducibility:** Recomputing RMSE, MAE, $R^2$, Pearson $r$, and Spearman $\rho$ directly from `holdout_predictions.csv` reproduces `holdout_metrics.json` with an error $< 10^{-9}$.

---

## 2. Investigation 1: Holdout vs Cross-Validation Metrics

### The Observation
- 5-Fold CV Summary (`results/model_optimization/summary.csv`):
  $\text{RMSE} = 0.6511 \pm 0.0910$, $\text{MAE} = 0.4731 \pm 0.0680$, $R^2 = 0.7737 \pm 0.0695$, $\text{Spearman} = 0.7928 \pm 0.0391$.
- Final Holdout Evaluation (`results/final_evaluation/holdout_metrics.json`):
  $\text{RMSE} = 0.7734$, $\text{MAE} = 0.5648$, $R^2 = 0.6677$, $\text{Spearman} = 0.7508$.

### The Technical Explanation
In `scripts/optimize_models.py`, `KFold(n_splits=5, shuffle=True, random_state=42)` partitions the 600 project IDs into 5 folds of 120 projects each.
In `scripts/evaluate_holdout.py`, `np.random.RandomState(42).shuffle(all_projects)` partitions the same 600 project IDs into an 80% train set (indices 0 to 479) and a 20% test set (indices 480 to 599).

Under scikit-learn's `KFold`, the 5th fold assigns indices `480:600` to the test split and `0:480` to the train split. Therefore:
$$\text{Train Projects (Holdout)} \equiv \text{Train Projects (Fold 5)}$$
$$\text{Test Projects (Holdout)} \equiv \text{Test Projects (Fold 5)}$$

Comparing the per-fold records in `results/model_optimization/fold_results.csv`:

| Fold | Eligible Train Pairs | Held-out Test Pairs | RMSE | MAE | $R^2$ | Spearman $\rho$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 1,876 | 117 | 0.5854 | 0.4524 | 0.8351 | 0.8195 |
| **Fold 2** | 1,940 | 100 | 0.6035 | 0.4124 | 0.8040 | 0.8130 |
| **Fold 3** | 1,908 | 127 | 0.5707 | 0.4133 | 0.8213 | 0.8305 |
| **Fold 4** | 1,935 | 123 | 0.7227 | 0.5223 | 0.7402 | 0.7501 |
| **Fold 5** | 1,916 | 120 | **0.7734** | **0.5648** | **0.6677** | **0.7508** |
| **5-Fold Mean** | **1,915** | **117.4** | **0.6511** | **0.4731** | **0.7737** | **0.7928** |
| **5-Fold Std** | — | — | **±0.0910** | **±0.0680** | **±0.0695** | **±0.0391** |

**Conclusion:** The holdout score is higher than the cross-validation mean because Fold 5 represents an adversarial/challenging partition with higher error ($\approx 1.34\sigma$ above the mean). The earlier report's statement that metrics "align closely" was misleading without this explicit breakdown. The report has been updated to show both the 5-fold mean and the fold range $[0.5707, 0.7734]$.

---

## 3. Investigation 2: Status of the Holdout Partition

### Did Test Labels Influence Decisions?
1. **Model Architecture Selection:** During cross-validation, Ridge Regression was chosen over Gradient Boosting and Random Forest based on 5-fold metrics (which included Fold 5).
2. **Hyperparameter Selection:** The hyperparameter $\alpha = 10.0$ was selected because its 5-fold average RMSE ($0.6511$) was superior to $\alpha=1.0$ ($0.6548$) and $\alpha=100.0$ ($0.6703$).
3. **Weight Fitting:** When training the Ridge model on the 1,916 training pairs, **no test labels were seen**. The weights were fitted strictly on $X_{\text{train}}, y_{\text{train}}$.

### Verdict: Can it be considered "Untouched"?
- **Strict ML Standpoint:** **No, not in the strictest sense.** An untouched holdout must be partitioned *prior* to any cross-validation or hyperparameter tuning and kept sealed until the final model configuration is locked down. Because $\alpha=10.0$ was selected using 5-fold CV that aggregated across all 600 projects, the holdout partition had indirect exposure during hyperparameter selection.
- **Practical Standpoint:** Because $\alpha=10.0$ performed consistently across all folds (and did not specifically overfit Fold 5—in fact, Fold 5 had the highest error under all alphas), the selection of $\alpha=10.0$ was not tailored to exploit Fold 5.
- **Recommended Corrective Action:** For academic publication or faculty verification, reserve a fresh, unseeded partition of faculty-annotated real projects that has never been involved in any synthetic benchmark cross-validation.

---

## 4. Investigation 3: Verification of Pair Construction & Disjointness

An automated audit verified the 2,996 cleaned pairs against the partition sets:

| Verification Condition | Formula / Rule | Result | Status |
|---|---|:---:|:---:|
| Project Partition Disjointness | $\text{train\_projects} \cap \text{test\_projects} = \emptyset$ | $480 \cap 120 = \emptyset$ | **PASSED** |
| Pair Training Project Disjointness | $\text{train\_pair\_projects} \subseteq \text{train\_projects}$ | $480 \subseteq 480$ | **PASSED** |
| Pair Testing Project Disjointness | $\text{test\_pair\_projects} \subseteq \text{test\_projects}$ | $120 \subseteq 120$ | **PASSED** |
| Pair Project Separation | $\text{train\_pair\_projects} \cap \text{test\_pair\_projects} = \emptyset$ | $\emptyset$ | **PASSED** |
| Cross-Boundary Pair Exclusion | Pairs where one project $\in \text{Train}$ and one $\in \text{Test}$ | 960 pairs excluded | **PASSED** |
| Pair Count Additivity | $N_{\text{train}} + N_{\text{test}} + N_{\text{cross}} = N_{\text{total}}$ | $1916 + 120 + 960 = 2996$ | **PASSED** |
| No Undirected Duplicates | Unique $(\min(A,B), \max(A,B))$ | 0 duplicates found | **PASSED** |
| No Self Pairs | $A \neq B$ for all rows | 0 self pairs found | **PASSED** |

---

## 5. Investigation 4: Leakage-Free TF-IDF Fitting

The vectorization code in `scripts/evaluate_holdout.py` (lines 114–135) was audited:
```python
train_projects_df = projects_df[projects_df["project_id"].isin(train_proj_set)].copy()
for sec_name, col_name in SECTIONS.items():
    train_texts = train_projects_df[col_name].fillna("").astype(str).tolist()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vec.fit(train_texts)  # Strictly 480 training texts
    matrix = vec.transform(all_texts)
```

### Empirical Check:
- Evaluated vocabulary learned by `vec.fit(train_texts)` on `project_title`: exactly 450 n-grams.
- Evaluated n-grams present in the 120 test project titles: **27 n-grams** appear exclusively in the test projects (e.g., domain-specific phrasing unique to test projects).
- **Confirmation:** None of these 27 test n-grams exist in the fitted vocabulary. When transforming test project titles, these n-grams are ignored by the vectorizer. Zero vocabulary or IDF weight leakage occurred.

---

## 6. Investigation 5: Independent Reproduction of Holdout Metrics

All evaluation metrics were independently recomputed directly from the saved CSV file (`results/final_evaluation/holdout_predictions.csv`):

| Metric | Value in `holdout_metrics.json` | Recomputed from `holdout_predictions.csv` | Absolute Difference |
|---|:---:|:---:|:---:|
| **RMSE** | 0.773352420164 | 0.773352420165 | $< 10^{-11}$ |
| **MAE** | 0.564822027336 | 0.564822027336 | $< 10^{-11}$ |
| **$R^2$ Score** | 0.667685402564 | 0.667685402564 | $< 10^{-11}$ |
| **Pearson $r$** | 0.823703981077 | 0.823703981077 | $< 10^{-11}$ |
| **Spearman $\rho$** | 0.750841241515 | 0.750841241515 | $0.0000$ |

*(Note: In the initial execution, `holdout_predictions.csv` rounded predictions to 4 decimal places, which caused a minor rank-tie difference of $0.00012$ in Spearman $\rho$. `scripts/evaluate_holdout.py` was updated to save full float precision in `holdout_predictions.csv`, achieving exact floating-point parity.)*

---

## 7. Investigation 6: Section Weight Directionality

The learned Ridge coefficients were examined for physical interpretability:
- `dataset_similarity`: **+1.9352**
- `abstract_similarity`: **+0.4984**
- `objectives_similarity`: **+0.2234**
- `title_similarity`: **+0.1446**
- `problem_similarity`: **+0.1288**
- `methodology_similarity`: **+0.0013**
- `algorithm_similarity`: **-0.0694**
- `technology_similarity`: **-0.1453**

**Correction in Report:** An earlier draft claimed all section coefficients were non-negative. In reality, `technology_similarity` (-0.1453) and `algorithm_similarity` (-0.0694) have small negative coefficients. This occurs because tools (e.g., "Python", "TensorFlow") and standard algorithm terms appear broadly across many dissimilar projects, acting as partial suppressor variables under Ridge regression. The report text in `final_evaluation_report.md` has been corrected.

---

## 8. Summary of Corrective Actions Taken

1. **Updated `scripts/evaluate_holdout.py`:**
   - Preserves full float precision in `holdout_predictions.csv` to ensure exact reproduction of metrics.
   - Clarified the section weights description in report generation.
   - Added explicit explanation of the relationship between the holdout and Fold 5 of cross-validation.
2. **Updated `results/final_evaluation/final_evaluation_report.md`:**
   - Now includes the 5-fold cross-validation range $[0.5707, 0.7734]$ alongside the 5-fold mean.
   - Discloses that the holdout partition matches Fold 5 and was derived after cross-validation rather than as an a priori isolated vault.
3. **Generated `results/final_evaluation/evaluation_audit.md`:**
   - Complete documentation of all six audit investigations.
