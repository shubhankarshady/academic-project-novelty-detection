# Phase 5.4: Fast, Controlled Model Tuning and Verification Report

**Document Reference**: `results/validation/phase5_tuning_report.md`  
**Execution Date**: November 13, 2026  
**Document Status**: **TUNED PRELIMINARY HYBRID BENCHMARK COMPLETED**  
**Tuned Model Configuration**: `models/phase5_tuned_hybrid_config.json`  
**Dataset Reference**: `data/annotation/faculty_evaluation_sample_120.xlsx`  

---

> [!CAUTION]
> **PRELIMINARY EVALUATION NOTICE — NOT COMPLETED FACULTY VALIDATION**  
> This tuning experiment used the **30 development/calibration pairs** evaluated by a single human reviewer (`Reviewer 1`).  
> - **Reviewer 2 independent ratings remain pending.**  
> - **Adjudicated consensus ratings remain pending.**  
> - **Inter-reviewer agreement (IRR) cannot yet be measured.**  
> - **The results herein must NOT be represented as completed faculty validation, independent double-review, or statistically established ground truth.**  
> All tuning was conducted under strict scientific safeguards to assess whether controlled parameter tuning on calibration data generalizes to held-out human ratings.

---

## 1. Executive Summary & Objective

The primary objective of Phase 5.4 was to test whether the existing **Hybrid Lexical-Semantic Fusion Model** could be improved using **strictly the 30 `DEV_CALIBRATION` pairs**, while keeping the **90 `HELD_OUT_EVALUATION` pairs completely untouched** until final verification.

### Core Experimental Findings:
1. **Optimal Alpha Selection ($\alpha = 0.60$)**:  
   5-Fold Cross-Validation on the 30 calibration pairs demonstrated that increasing the lexical weight from $\alpha = 0.50$ to $\alpha = 0.60$ achieved the minimum out-of-fold RMSE ($0.3921$), improved calibration exact agreement from $76.7\%$ to $80.0\%$, and raised calibration Macro $F_1$ from $48.0\%$ to $50.6\%$, while maintaining balanced bias ($\text{MBE} = -0.0356$).
2. **Section Weights Frozen (No Retraining)**:  
   Investigation into section weights confirmed that fitting 8 free parameters on only 30 observations carries severe risk of sample-specific overfitting. The structurally robust candidate constrained weights (non-negative, sum to 4.0, intercept 0.0) were **frozen without modification**.
3. **Generalization to Unseen Held-Out Pairs ($N = 90$)**:  
   When evaluated ONCE on the completely untouched 90 held-out pairs, the tuned model ($\alpha = 0.60$) delivered substantial improvements over the Phase 5.3 baseline:
   - **Exact Agreement leaped from $74.44\%$ to $84.44\%$ ($+10.00\%$ absolute improvement)**, outperforming the TF-IDF Ridge model ($83.33\%$).
   - **Macro $F_1$ Score surged from $57.35\%$ to $69.72\%$ ($+12.37\%$ absolute improvement)**.
   - **Class 0 (Unrelated) Recall doubled from $36.8\%$ to $73.7\%$** (14/19 pairs correct).
   - **Class 2 (Moderate) Recall achieved $100.0\%$** (5/5 pairs correct).
   - **Within-One Rating Agreement was maintained at $98.89\%$** (89/90 pairs).
   - **Continuous RMSE improved from $0.4421$ to $0.4410$**, and Spearman rank correlation increased to **$0.8514$**.

---

## 2. Baseline Reference (Phase 5.3 Preliminary Benchmark)

In Phase 5.3, the baseline models evaluated against the 90 held-out pairs yielded:
- **TF-IDF Ridge Model**: MAE $0.4713$, RMSE $0.6379$, Mean Bias $-0.4130$, Exact $83.33\%$, Macro $F_1$ $50.62\%$. (Suffers from negative bias and zero predictions for Classes 1 and 4).
- **Dense Semantic Model**: MAE $0.4563$, RMSE $0.5869$, Mean Bias $+0.3362$, Exact $65.56\%$, Macro $F_1$ $40.00\%$. (Suffers from positive background similarity drift).
- **Baseline Hybrid ($\alpha=0.50$)**: MAE $0.3155$, RMSE $0.4421$, Mean Bias $+0.0164$, Exact $74.44\%$, Macro $F_1$ $57.35\%$. (Substantially outperformed individual channels, but slightly overpredicted Level 0 as Level 1).

---

## 3. Step 2: Hybrid Alpha Tuning on Calibration Cohort ($N = 30$)

To optimize the convex mixture:
$$s_{\text{hybrid}, i}(\alpha) = \alpha \cdot s_{\text{tfidf}, i} + (1.0 - \alpha) \cdot s_{\text{dense}, i}$$

A controlled grid search was conducted across $\alpha \in \{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8\}$ using **5-Fold Cross-Validation within the 30 calibration pairs** (6 pairs per fold).

### Calibration Alpha Grid Search Results

| Candidate $\alpha$ | CV MAE | CV RMSE | CV Mean Bias (MBE) | CV Pearson $r$ | CV Spearman $\rho$ | CV Exact Agree (%) | CV Within-1 Agree (%) | CV Macro $F_1$ (%) | Full Dev MAE | Full Dev RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.2** | $0.3595$ | $0.4747$ | $+0.2106$ | $0.9222$ | $0.8506$ | $70.00\%$ | $100.0\%$ | $36.95\%$ | $0.3595$ | $0.4994$ |
| **0.3** | $0.3345$ | $0.4423$ | $+0.1490$ | $0.9329$ | $0.8506$ | $73.33\%$ | $100.0\%$ | $41.49\%$ | $0.3345$ | $0.4629$ |
| **0.4** | $0.3221$ | $0.4171$ | $+0.0875$ | $0.9406$ | $0.8398$ | $73.33\%$ | $100.0\%$ | $43.39\%$ | $0.3221$ | $0.4335$ |
| **0.5 (Baseline)**| **$0.3209$** | $0.4001$ | $+0.0259$ | $0.9445$ | $0.8398$ | $76.67\%$ | $100.0\%$ | $47.96\%$ | $0.3209$ | $0.4128$ |
| **0.6 (Selected)**| $0.3272$ | **$0.3921$** | **$-0.0356$** | **$0.9447$** | **$0.8398$** | **$80.00\%$** | **$100.0\%$** | **$50.63\%$** | $0.3272$ | **$0.4022$** |
| **0.7** | $0.3346$ | $0.3934$ | $-0.0972$ | $0.9413$ | $0.8398$ | $83.33\%$ | $100.0\%$ | $53.30\%$ | $0.3346$ | $0.4024$ |
| **0.8** | $0.3485$ | $0.4041$ | $-0.1588$ | $0.9344$ | $0.8398$ | $86.67\%$ | $100.0\%$ | $54.63\%$ | $0.3485$ | $0.4135$ |

### Selection Rationale for $\alpha = 0.60$:
1. **Minimum Cross-Validation RMSE**: $\alpha = 0.60$ achieved the lowest out-of-fold RMSE ($0.3921$), representing a $2.0\%$ error reduction over baseline $\alpha = 0.50$ ($0.4001$).
2. **Balanced Directional Bias**: Mean Bias Error is near zero ($\text{MBE} = -0.0356$), preventing both semantic overprediction (prevalent at $\alpha \le 0.40$) and lexical underprediction (prevalent at $\alpha \ge 0.70$).
3. **Higher Discrete Classification Fidelity**: In cross-validation, exact agreement increased from $76.67\%$ to $80.00\%$, and Macro $F_1$ increased from $47.96\%$ to $50.63\%$.
4. **Avoidance of Overfitting Boundary**: While $\alpha = 0.70$ and $0.80$ produced higher exact agreement on the calibration split, their continuous MAE rose steadily ($0.3272 \to 0.3346 \to 0.3485$) and negative bias drifted toward $-0.16$, indicating excessive reliance on lexical matching.

---

## 4. Step 3: Investigation of Section Weights

The current candidate section weights were audited:
- Title: $+0.9535$ ($23.84\%$)
- Methodology: $+0.3776$ ($9.44\%$)
- Dataset: $+1.9529$ ($48.82\%$)
- Technology: $+0.7160$ ($17.90\%$)
- Abstract, Problem, Objectives, Algorithm: $0.0000$ ($0.0\%$)
- Sum of weights: $4.0000$, Intercept: $0.0000$.

### Regularization and Refitting Experiments on Calibration Data:
1. **Unregularized SLSQP Refit on 30 Pairs**:  
   Refitting 8 free weights on $N=30$ observations resulted in extreme weight redistribution (Title spiked to $39.8\%$, Technology dropped to $0.3\%$). This represents severe sample noise memorization ($p/n = 8/30 \approx 0.27$).
2. **Uniform Smoothing Regularization ($\lambda \in [0.05, 0.20]$)**:  
   Shrinking candidate weights toward uniform section weights ($\vec{w}_{\text{smooth}} = (1 - \lambda)\vec{w}_{\text{cand}} + \lambda (0.5)$) gave non-zero weights to the four inactive sections. While this yielded a tiny nominal CV MAE improvement ($0.3209 \to 0.3142$), it diluted the proven discriminant power of Dataset and Title on real controlled benchmark cases (such as dataset reuse detection).
3. **Decision: Retain Frozen Candidate Constrained Weights**:  
   In compliance with the project's zero-leakage and anti-overfitting directives, **no section weights were modified**. The candidate weights were preserved intact, and model tuning was restricted strictly to the single hyperparameter $\alpha = 0.60$.

---

## 5. Step 4: Model Freezing

The tuned configuration was frozen and saved to [`models/phase5_tuned_hybrid_config.json`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/models/phase5_tuned_hybrid_config.json):
- **Version**: `phase5_tuned_hybrid_v1.0`
- **Tuned Alpha**: `0.60`
- **Section Weights**: Retained candidate constrained weights (`models/candidate_constrained_fusion_model.joblib`)
- **Status**: Frozen prior to held-out evaluation.

---

## 6. Step 5: Final Held-Out Evaluation ($N = 90$ Pairs)

The frozen configuration was evaluated **ONCE** against the 90 held-out pairs. Neither weights nor hyperparameters were adjusted after unblinding the held-out split.

### 4-Model Comparative Benchmark (Held-Out Evaluation Split, $N = 90$)

```
=============================================================================================================
                              FINAL HELD-OUT EVALUATION BENCHMARK (N = 90 PAIRS)
=============================================================================================================
Metric                             TF-IDF Ridge      Dense Semantic   Baseline Hybrid (0.50)   Tuned Hybrid (0.60)
-------------------------------------------------------------------------------------------------------------
Continuous Error Metrics:
   Mean Absolute Error (MAE)             0.4713              0.4563                   0.3155               0.3270
   Root Mean Squared Error (RMSE)        0.6379              0.5869                   0.4421               0.4410
   Mean Bias Error (MBE)                -0.4130             +0.3362                  +0.0164              -0.0476
Correlation Metrics:
   Pearson Correlation (r)               0.9292              0.9380                   0.9462               0.9463
   Pearson p-value                     8.32e-40            2.88e-42                 6.69e-45             6.21e-45
   Spearman Rank Correlation (rho)       0.7716              0.8299                   0.8502               0.8514
   Spearman p-value                    5.61e-19            5.12e-24                 3.04e-26             2.14e-26
Discrete Classification Metrics:
   Exact Agreement (%)                   83.33%              65.56%                   74.44%               84.44%
   Within-One Rating Agreement (%)       97.78%              98.89%                   98.89%               98.89%
   Model-Reviewer QWK (kappa_w)          0.9186              0.8644                   0.9028               0.9287
   Macro F1 Score (%)                    50.62%              40.00%                   57.35%               69.72%
   Weighted F1 Score (%)                 77.86%              61.99%                   74.55%               84.28%
=============================================================================================================
```

### Confusion Matrix Comparison (Held-Out Set, $N = 90$)

#### Baseline Hybrid ($\alpha = 0.50$)
```
            Pred 0    Pred 1    Pred 2    Pred 3    Pred 4    Total
Actual 0         7        12         0         0         0       19
Actual 1         2         5         0         0         0        7
Actual 2         0         0         4         1         0        5
Actual 3         0         1         1        50         2       54
Actual 4         0         0         0         4         1        5
Total            9        18         5        55         3       90
```

#### Tuned Hybrid ($\alpha = 0.60$)
```
            Pred 0    Pred 1    Pred 2    Pred 3    Pred 4    Total
Actual 0        14         5         0         0         0       19  <-- 14/19 correct (+7 pairs!)
Actual 1         2         5         0         0         0        7  <-- 5/7 correct
Actual 2         0         0         5         0         0        5  <-- 5/5 correct (100% recall!)
Actual 3         0         1         1        51         1       54  <-- 51/54 correct
Actual 4         0         0         0         4         1        5  <-- 1/5 correct
Total           16        11         6        55         2       90
```

### Per-Class Performance Breakdown: Baseline vs. Tuned

| Ordinal Class | Support ($N=90$) | Baseline Hybrid ($\alpha=0.50$) | Tuned Hybrid ($\alpha=0.60$) | Delta / Improvement |
| :---: | :---: | :---: | :---: | :--- |
| **Class 0 (Unrelated)** | 19 | $P=77.8\%, R=36.8\%, F_1=50.0\%$ | **$P=87.5\%, R=73.7\%, F_1=80.0\%$** | **$+36.9\%$ Recall, $+30.0\%$ $F_1$** |
| **Class 1 (Slight)** | 7 | $P=27.8\%, R=71.4\%, F_1=40.0\%$ | **$P=45.5\%, R=71.4\%, F_1=55.6\%$** | **$+17.7\%$ Precision, $+15.6\%$ $F_1$** |
| **Class 2 (Moderate)** | 5 | $P=80.0\%, R=80.0\%, F_1=80.0\%$ | **$P=83.3\%, R=100.0\%, F_1=90.9\%$** | **$+20.0\%$ Recall, $+10.9\%$ $F_1$** |
| **Class 3 (High)** | 54 | $P=90.9\%, R=92.6\%, F_1=91.7\%$ | **$P=92.7\%, R=94.4\%, F_1=93.6\%$** | **$+1.8\%$ Recall, $+1.9\%$ $F_1$** |
| **Class 4 (Duplicate)** | 5 | $P=33.3\%, R=20.0\%, F_1=25.0\%$ | **$P=50.0\%, R=20.0\%, F_1=28.6\%$** | **$+16.7\%$ Precision, $+3.6\%$ $F_1$** |

---

## 7. Before-versus-After Comparison Summary

| Metric | Phase 5.3 Baseline ($\alpha=0.50$) | Phase 5.4 Tuned ($\alpha=0.60$) | Absolute Difference | Relative Change | Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Exact Agreement** | $74.44\%$ | **$84.44\%$** | **$+10.00\%$** | $+13.4\%$ | **Major Improvement** (76 vs 67 pairs exact) |
| **Macro $F_1$ Score** | $57.35\%$ | **$69.72\%$** | **$+12.37\%$** | $+21.6\%$ | **Substantial Leap** across all 5 classes |
| **Weighted $F_1$ Score** | $74.55\%$ | **$84.28\%$** | **$+9.73\%$** | $+13.1\%$ | Substantial population-weighted gain |
| **Model QWK ($\kappa_w$)** | $0.9028$ | **$0.9287$** | **$+0.0259$** | $+2.9\%$ | Higher ordinal concordance |
| **Root Mean Squared Error**| $0.4421$ | **$0.4410$** | **$-0.0011$** | $-0.2\%$ | Slightly lower variance |
| **Spearman Rank ($\rho$)** | $0.8502$ | **$0.8514$** | **$+0.0012$** | $+0.1\%$ | Maintained strong rank monotonicity |
| **Pearson Correlation ($r$)** | $0.9462$ | **$0.9463$** | **$+0.0001$** | $+0.0\%$ | Unchanged high linear correlation |
| **Mean Absolute Error** | $0.3155$ | $0.3270$ | $+0.0115$ | $+3.6\%$ | Minor trade-off ($0.01$ similarity points) |
| **Within-1 Agreement** | $98.89\%$ | **$98.89\%$** | $0.00\%$ | $0.0\%$ | Maintained exceptional adjacent fidelity |

---

## 8. Verification and Safeguards Audit

The experimental protocol adhered strictly to scientific safeguards:
- **Zero Label Tampering**: The original 120 ratings entered by Reviewer 1 in `faculty_evaluation_sample_120.xlsx` were unchanged.
- **Strict Data Segregation**: Held-out evaluation ratings were unblinded only after `phase5_tuned_hybrid_config.json` was generated and frozen.
- **Production Immutability**: Production models (`models/ridge_fusion_model.joblib`) and TF-IDF extractors were untouched.
- **API Continuity**: API defaults (`NOVELTY_SCORING_MODE="tfidf"`) were preserved. The tuned model exists as a verified candidate artifact.

---

## 9. Generated Artifacts

1. [`models/phase5_tuned_hybrid_config.json`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/models/phase5_tuned_hybrid_config.json) — Frozen tuned configuration.
2. [`results/validation/phase5_tuning_results.csv`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/phase5_tuning_results.csv) — Cross-validation and full Dev results across all 7 candidate alphas.
3. [`results/validation/phase5_tuned_model_comparison.csv`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/phase5_tuned_model_comparison.csv) — Comparative benchmark across all 4 models on Held-Out and Dev splits.
4. [`results/validation/phase5_tuned_model_predictions.csv`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/phase5_tuned_model_predictions.csv) — Pair-by-pair predictions across all 120 rows with baseline delta tracking.
5. [`scripts/run_phase5_tuning.py`](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/scripts/run_phase5_tuning.py) — End-to-end tuning and evaluation script.

