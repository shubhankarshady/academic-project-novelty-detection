# Model Optimization & Hyperparameter Tuning Report

**Dataset:** `data/processed/similarity_pairs_clean.csv` (2,996 clean undirected pairs)  
**Evaluation Protocol:** 5-Fold Project-Disjoint Cross-Validation (Zero Data Leakage)  
**Date:** October 22, 2026  
**Status:** Completed  

---

## 1. Executive Summary & Best Model

- **Best Overall Model:** **GradientBoosting**
- **Optimal Feature Channel:** **TF-IDF only**
- **Optimal Hyperparameters:** `n_estimators=100, max_depth=3, lr=0.1`
- **Performance:**
  - **RMSE:** **0.6487 ± 0.6487**
  - **MAE:** **0.4632 ± 0.0731**
  - **$R^2$:** **0.7750 ± 0.0715**
  - **Spearman $\rho$:** **0.7905 ± 0.0307**

> [!WARNING]
> **Synthetic Benchmark Limitation:**
> The target similarity labels are derived from synthetic benchmark rules. These metrics measure internal mathematical consistency and model generalization against benchmark patterns, not faculty-validated real-world performance.

---

## 2. Complete Model Comparison Leaderboard

Models are sorted by primary selection metric (**Mean RMSE** ascending):

| Model | Feature Set | Hyperparameters | RMSE | MAE | $R^2$ | Spearman $\rho$ |
|---|---|---|:---:|:---:|:---:|:---:|
| GradientBoosting | TF-IDF only | `n_estimators=100, max_depth=3, lr=0.1` | 0.6487 ± 0.0966 | 0.4632 ± 0.0731 | 0.7750 ± 0.0715 | 0.7905 ± 0.0307 |
| Ridge | TF-IDF only | `alpha=10.0` | 0.6511 ± 0.0910 | 0.4731 ± 0.0680 | 0.7737 ± 0.0695 | 0.7928 ± 0.0391 |
| Ridge | Hybrid | `alpha=10.0` | 0.6545 ± 0.0900 | 0.4734 ± 0.0680 | 0.7714 ± 0.0692 | 0.7863 ± 0.0386 |
| Ridge | TF-IDF only | `alpha=1.0` | 0.6548 ± 0.0900 | 0.4709 ± 0.0673 | 0.7710 ± 0.0703 | 0.7864 ± 0.0368 |
| Ridge | TF-IDF only | `alpha=0.01` | 0.6550 ± 0.0894 | 0.4715 ± 0.0676 | 0.7709 ± 0.0700 | 0.7857 ± 0.0358 |
| Ridge | TF-IDF only | `alpha=0.1` | 0.6553 ± 0.0896 | 0.4710 ± 0.0674 | 0.7706 ± 0.0702 | 0.7845 ± 0.0350 |
| Ridge | Hybrid | `alpha=1.0` | 0.6555 ± 0.0900 | 0.4714 ± 0.0680 | 0.7705 ± 0.0708 | 0.7839 ± 0.0329 |
| Ridge | Hybrid | `alpha=0.01` | 0.6559 ± 0.0895 | 0.4727 ± 0.0691 | 0.7702 ± 0.0707 | 0.7839 ± 0.0351 |
| Ridge | Hybrid | `alpha=0.1` | 0.6563 ± 0.0897 | 0.4719 ± 0.0686 | 0.7699 ± 0.0709 | 0.7851 ± 0.0325 |
| GradientBoosting | Dense only | `n_estimators=100, max_depth=3, lr=0.1` | 0.6628 ± 0.0784 | 0.4772 ± 0.0620 | 0.7671 ± 0.0597 | 0.7859 ± 0.0190 |
| Ridge | Hybrid | `alpha=100.0` | 0.6660 ± 0.0877 | 0.5030 ± 0.0653 | 0.7641 ± 0.0665 | 0.7951 ± 0.0324 |
| GradientBoosting | Hybrid | `n_estimators=100, max_depth=3, lr=0.1` | 0.6697 ± 0.0979 | 0.4756 ± 0.0751 | 0.7608 ± 0.0740 | 0.7824 ± 0.0218 |
| Ridge | TF-IDF only | `alpha=100.0` | 0.6703 ± 0.0872 | 0.5232 ± 0.0617 | 0.7612 ± 0.0662 | 0.7967 ± 0.0325 |
| Ridge | Dense only | `alpha=0.01` | 0.6807 ± 0.0792 | 0.5060 ± 0.0624 | 0.7541 ± 0.0627 | 0.7921 ± 0.0310 |
| Ridge | Dense only | `alpha=0.1` | 0.6807 ± 0.0792 | 0.5061 ± 0.0624 | 0.7541 ± 0.0627 | 0.7921 ± 0.0309 |
| Ridge | Dense only | `alpha=1.0` | 0.6808 ± 0.0793 | 0.5063 ± 0.0623 | 0.7541 ± 0.0627 | 0.7918 ± 0.0309 |
| Ridge | Dense only | `alpha=10.0` | 0.6815 ± 0.0805 | 0.5088 ± 0.0629 | 0.7536 ± 0.0629 | 0.7938 ± 0.0319 |
| Ridge | Dense only | `alpha=100.0` | 0.7009 ± 0.0793 | 0.5430 ± 0.0593 | 0.7401 ± 0.0610 | 0.7975 ± 0.0278 |

---

## 3. Project-Disjoint Fold Pair Statistics

In project-disjoint cross-validation, 600 unique projects are partitioned into 5 folds (480 train / 120 test projects per fold).
To prevent representation leakage, pairs where one project belongs to train and the other to test (cross-boundary pairs) are strictly excluded from both training and testing.

| Fold | Eligible Training Pairs | Held-out Testing Pairs |
|---|:---:|:---:|
| Fold 1 | 1876 | 117 |
| Fold 2 | 1940 | 100 |
| Fold 3 | 1908 | 127 |
| Fold 4 | 1935 | 123 |
| Fold 5 | 1916 | 120 |

---

## 4. Key Experimental Findings

### 4.1 Ridge Regularization Tuning ($\alpha$)
- Ridge regression was evaluated across $\alpha \in [0.01, 0.1, 1.0, 10.0, 100.0]$.
- Across TF-IDF and Hybrid channels, performance is highly stable across $\alpha \in [0.1, 1.0, 10.0]$.
- Extreme values ($\alpha=100.0$) introduce underfitting with elevated RMSE.

### 4.2 Semantic Embeddings vs TF-IDF Alone
- TF-IDF section features alone achieve lower RMSE and higher $R^2$ than dense embeddings alone.
- Combining TF-IDF with dense embeddings (Hybrid channel) produces competitive results, but does not provide a statistically decisive margin over pure TF-IDF alone on these synthetic benchmark labels.

### 4.3 Gradient Boosting vs Linear Ridge
- Gradient Boosting Regressor (GBR) captures potential non-linear interactions across project sections.
- While GBR performs strongly, Ridge regression provides equivalent or superior generalization with negligible computational overhead and complete transparency.

---

## 5. Methodological & Test Set Reservations

- **No Premature Model Freezing:** The final production model artifact was **not** overwritten.
- **Untouched Test Reservation Recommendation:** While 5-fold project-disjoint CV provides a cross-validation estimate, a dedicated, permanently held-out project test set (e.g., 20% / 120 projects reserved completely untouched before final training) should be locked down before certifying production deployment.
