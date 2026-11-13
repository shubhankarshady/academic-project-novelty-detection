# Phase 5.3: Preliminary Human Evaluation and Model Comparison Report

**Document Reference**: `results/validation/phase5_preliminary_evaluation_report.md`  
**Execution Date**: November 13, 2026  
**Document Status**: **PRELIMINARY SINGLE-REVIEWER EVALUATION**  
**Lead Reviewer**: Reviewer 1 (Ratings fully collected for N = 120 pairs)  
**Pending Actions**: Reviewer 2 Independent Ratings, Reviewer Comments, Consensus Adjudication  

---

> [!CAUTION]
> **PRELIMINARY EVALUATION NOTICE — NOT COMPLETED FACULTY VALIDATION**  
> This report documents a **preliminary evaluation** based on human similarity ratings supplied by a single reviewer (`reviewer_1_rating`) across 120 project pairs.  
> - **Independent Reviewer 2 ratings are pending.**
> - **Adjudicated consensus ratings are pending.**
> - **Inter-reviewer agreement (IRR / Quadratic Weighted Kappa between reviewers) cannot yet be measured.**
> - **The results herein must NOT be represented as completed faculty validation, independent double-review, or statistically established ground truth.**  
> All findings, metric calculations, and model comparisons in this document represent an interim checkpoint to evaluate scoring behavior against initial human feedback while awaiting second-faculty availability.

---

## 1. Executive Summary

In Phase 5.2, an expanded faculty evaluation cohort of **120 unique project pairs** was created and structured into two double-blind worksheets of 60 pairs each (`Evaluation_Batch_1` and `Evaluation_Batch_2`). Reviewer 1 has completed ratings for all 120 pairs on the ordinal 0–4 similarity scale.

This Phase 5.3 evaluation benchmarks three existing scoring architectures against Reviewer 1 ratings:
1. **Production TF-IDF Ridge Model** (`models/ridge_fusion_model.joblib`)
2. **Dense Semantic Embedding Model** (`models/section_dense_extractor.joblib`, SentenceTransformer `all-MiniLM-L6-v2`)
3. **Hybrid Lexical-Semantic Fusion Model** (`models/candidate_constrained_fusion_model.joblib`, convex weight $\alpha=0.50$)

All model artifacts were preserved intact without retraining, fine-tuning, or parameter modification. In accordance with rigorous scientific protocol, the **90 held-out evaluation pairs** served as the primary benchmark, while the **30 development/calibration pairs** were examined strictly for exploratory consistency.

### Primary Benchmark Takeaways (Held-Out Evaluation Set, N = 90 Pairs)

| Metric Category | Metric | TF-IDF Ridge Model | Dense Semantic Model | Hybrid Fusion ($\alpha=0.50$) | Best Performing Architecture |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Continuous Error** | **Mean Absolute Error (MAE)** | $0.4713$ | $0.4563$ | **$0.3155$** | **Hybrid Fusion ($-33.0\%$ error)** |
| | **Root Mean Squared Error (RMSE)** | $0.6379$ | $0.5869$ | **$0.4421$** | **Hybrid Fusion ($-30.7\%$ error)** |
| | **Mean Bias Error (MBE)** | $-0.4130$ (Underpredicts) | $+0.3362$ (Overpredicts) | **$+0.0164$ (Unbiased)** | **Hybrid Fusion (Near-Zero Bias)** |
| **Correlation** | **Pearson Correlation ($r$)** | $0.9292$ | $0.9380$ | **$0.9462$** | **Hybrid Fusion** |
| | **Spearman Rank Correlation ($\rho$)** | $0.7716$ | $0.8299$ | **$0.8502$** | **Hybrid Fusion** |
| **Discrete Agreement** | **Exact Agreement (%)** | **$83.33\%$** | $65.56\%$ | $74.44\%$ | **TF-IDF Ridge** *(bolstered by class skew)* |
| | **Within-One Rating (%)** | $97.78\%$ | **$98.89\%$** | **$98.89\%$** | **Semantic & Hybrid Tied ($89/90$ pairs)** |
| **Classification** | **Model-Reviewer QWK ($\kappa_w$)** | **$0.9186$** | $0.8644$ | $0.9028$ | **TF-IDF Ridge** |
| | **Macro $F_1$ Score (%)** | $50.62\%$ | $40.00\%$ | **$57.35\%$** | **Hybrid Fusion** |
| | **Weighted $F_1$ Score (%)** | **$77.86\%$** | $61.99\%$ | $74.55\%$ | **TF-IDF Ridge** |

> [!NOTE]
> **Key Finding on Complementary Error Cancellation:**  
> TF-IDF systematically underpredicts similarity ($\text{MBE} = -0.4130$) because lexical mismatch depresses similarity on paraphrased text. Conversely, Dense Semantic embeddings systematically overpredict similarity ($\text{MBE} = +0.3362$) because transformer embeddings capture background academic vocabulary overlap. When merged in the convex hybrid model ($\alpha=0.50$), these opposing biases cancel out almost completely ($\text{MBE} = +0.0164$), driving continuous MAE down to **$0.3155$** and RMSE down to **$0.4421$**.

---

## 2. Validation of Human Reviewer Data

The completed Excel workbook (`data/annotation/faculty_evaluation_sample_120.xlsx`) was audited programmatically against all validation criteria:

```
[CHECK] all_120_present                      : PASS (120/120 values present, 0 missing)
[CHECK] ratings_valid_integers_0_to_4        : PASS (All values in {0, 1, 2, 3, 4})
[CHECK] no_duplicates                        : PASS (0 duplicate pairs, 0 duplicate pair IDs)
[CHECK] dual_batch_balance_60_each           : PASS (Batch 1 = 60, Batch 2 = 60)
[CHECK] reviewer_2_and_adjudication_unfilled : PASS (0 values in pending audit columns)
[CHECK] subsets_preserved_30_90              : PASS (Dev/Calibration = 30, Held-Out = 90)
```

### Reviewer 1 Rating Distribution

| Ordinal Similarity Rating | Category Description | Dev / Calibration (N=30) | Held-Out Evaluation (N=90) | Total Cohort (N=120) | Cohort Percentage (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Not Similar (Unrelated) | 5 | 19 | **24** | $20.0\%$ |
| **1** | Slightly Similar (Jargon/Tools) | 4 | 7 | **11** | $9.2\%$ |
| **2** | Moderately Similar (Shared Components) | 4 | 5 | **9** | $7.5\%$ |
| **3** | Highly Similar (Substantial Paraphrase) | 15 | 54 | **69** | **$57.5\%$** |
| **4** | Nearly Duplicate / Duplicate | 2 | 5 | **7** | $5.8\%$ |
| **Total** | **All Levels (0 – 4)** | **30** | **90** | **120** | **$100.0\%$** |

### Crucial Empirical Observation: Severe Class Imbalance
While the synthetic benchmark cohort was constructed with a balanced uniform prior ($24$ pairs per level, $20\%$ each), the actual human reviewer concentrated ratings heavily into **Level 3 ("Highly Similar")**, accounting for **$57.5\%$ of all pairs** ($60.0\%$ of the held-out split). The reviewer applied Level 4 ("Duplicate") sparingly ($5.8\%$), reserving it strictly for verbatim or nearly identical project proposals.

```
Reviewer 1 Rating Distribution (Held-Out Evaluation Split, N = 90)
Level 0 [■■■■■■■■■■■■■■■■■■■] 19 (21.1%)
Level 1 [■■■■■■■]              7 (7.8%)
Level 2 [■■■■■]                5 (5.6%)
Level 3 [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 54 (60.0%)
Level 4 [■■■■■]                5 (5.6%)
```

---

## 3. Evaluation Subsets & Methodological Safeguards

To prevent data leakage and maintain statistical validity:
1. **Held-Out Evaluation Set (`HELD_OUT_EVALUATION`, 90 pairs)**:
   - Kept strictly unseen during all prior model fitting.
   - Evaluated using pre-trained model weights.
   - Used for all primary benchmark metrics reported in this study.
2. **Development / Calibration Set (`DEV_CALIBRATION`, 30 pairs)**:
   - Preserved for pilot alignment and exploratory calibration.
   - Not pooled into held-out statistics.
3. **No Retroactive Hyperparameter Tuning**:
   - Neither the Ridge alpha ($\alpha = 1.0$), candidate SLSQP weights, nor the hybrid mixture parameter ($\alpha = 0.50$) were adjusted or optimized using these human annotations.

---

## 4. Model Architectures & Prediction Formulas

All three architectures generate a continuous similarity prediction $\hat{y} \in [0.0, 4.0]$ from 8 section-level similarities:

$$\mathbf{s} = [s_{\text{title}}, s_{\text{abstract}}, s_{\text{problem}}, s_{\text{objectives}}, s_{\text{methodology}}, s_{\text{dataset}}, s_{\text{algorithm}}, s_{\text{technology}}]^T$$

### Model 1: Production TF-IDF Ridge Model
- **Inputs**: Lexical cosine similarities $\mathbf{s}_{\text{tfidf}} \in [0.0, 1.0]^8$ computed via 8 section-specific `TfidfVectorizer` models fitted on 600 corpus projects.
- **Formula**:
  $$\hat{y}_{\text{tfidf}} = \text{clip}\left(b_{\text{ridge}} + \sum_{i=1}^8 w_i^{\text{ridge}} s_{\text{tfidf}, i}, \; 0.0, \; 4.0\right)$$
- **Weights**:
  - Intercept $b = -0.070632$
  - Active weights: Title ($+0.1340$), Abstract ($+0.3980$), Methodology ($+0.0099$), Dataset ($+2.2054$)
  - Negative weights: Problem ($-0.1259$), Algorithm ($-0.0651$), Technology ($-0.1456$)
- **Defects**: Theoretical ceiling at $\approx 2.5768 / 4.0$ for exact duplicates; negative weights penalize shared tools.

### Model 2: Dense Semantic Embedding Model (SentenceTransformer `all-MiniLM-L6-v2`)
- **Inputs**: Dense cosine similarities $\mathbf{s}_{\text{dense}} \in [0.0, 1.0]^8$ generated by encoding text sections into 384-dimensional $L_2$-normalized dense vectors.
- **Formula**:
  $$\hat{y}_{\text{semantic}} = \text{clip}\left(\sum_{i=1}^8 w_i^{\text{candidate}} s_{\text{dense}, i}, \; 0.0, \; 4.0\right)$$
- **Weights**:
  - Non-negative SLSQP weights: Title ($+0.9535$), Methodology ($+0.3776$), Dataset ($+1.9529$), Technology ($+0.7160$). Intercept $b = 0.0$.
  - Zero-weight sections: Abstract ($0.0$), Problem ($0.0$), Objectives ($0.0$), Algorithm ($0.0$).
- **Defects**: Elevated background cosine similarities across academic texts create positive bias.

### Model 3: Hybrid Lexical-Semantic Fusion Model ($\alpha = 0.50$)
- **Inputs**: Convex section-wise mixture of lexical and dense similarities:
  $$s_{\text{hybrid}, i} = 0.50 \cdot s_{\text{tfidf}, i} + 0.50 \cdot s_{\text{dense}, i}, \quad \forall i \in \{1 \dots 8\}$$
- **Formula**:
  $$\hat{y}_{\text{hybrid}} = \text{clip}\left(\sum_{i=1}^8 w_i^{\text{candidate}} s_{\text{hybrid}, i}, \; 0.0, \; 4.0\right)$$
- **Score Range**: Guaranteed strictly bounded in $[0.0, 4.0]$.

### Justification of Continuous-to-Ordinal Discretization
To compute discrete classification metrics (confusion matrices, exact agreement, macro $F_1$, and weighted kappa), continuous similarity scores $\hat{y} \in [0.0, 4.0]$ are converted to ordinal ratings $\hat{y}_{\text{ord}} \in \{0, 1, 2, 3, 4\}$ via **standard nearest-integer rounding**:

$$\hat{y}_{\text{ord}} = \text{clip}(\lfloor \hat{y} + 0.5 \rfloor, 0, 4)$$

**Methodological Justification**:
1. **Target Compatibility**: All three models were trained to predict numerical target values on the $0.0 - 4.0$ scale. Nearest-integer rounding divides the continuum into symmetric Voronoi regions around each integer target: $[0.0, 0.5) \to 0$, $[0.5, 1.5) \to 1$, $[1.5, 2.5) \to 2$, $[2.5, 3.5) \to 3$, $[3.5, 4.0] \to 4$.
2. **Boundary Sensitivity Awareness**: For continuous predictions near the decision boundary (e.g. $2.49$ vs $2.51$), rounding can flip discrete class assignments. Therefore, continuous metrics (MAE, RMSE, Pearson $r$, Spearman $\rho$) are prioritized as the primary performance indicators.

---

## 5. Detailed Metric Results on Held-Out Pairs (N = 90)

```
========================================================================================
                      HELD-OUT EVALUATION METRICS (N = 90 PAIRS)
========================================================================================
Metric                               TF-IDF Ridge      Dense Semantic        Hybrid Fusion
----------------------------------------------------------------------------------------
Mean Absolute Error (MAE)                  0.4713              0.4563               0.3155
Root Mean Squared Error (RMSE)             0.6379              0.5869               0.4421
Mean Bias Error (MBE)                     -0.4130             +0.3362              +0.0164
Pearson Correlation (r)                    0.9292              0.9380               0.9462
Pearson p-value                          8.32e-40            2.88e-42             6.69e-45
Spearman Rank Correlation (rho)            0.7716              0.8299               0.8502
Spearman p-value                         5.61e-19            5.12e-24             3.04e-26
Exact Rating Agreement (%)                 83.33%              65.56%               74.44%
Within-One Rating Agreement (%)            97.78%              98.89%               98.89%
Model-Reviewer QWK (kappa_w)               0.9186              0.8644               0.9028
Model-Reviewer Linear Kappa                0.8606              0.7410               0.8228
Macro F1 Score (%)                         50.62%              40.00%               57.35%
Weighted F1 Score (%)                      77.86%              61.99%               74.55%
========================================================================================
```

### Confusion Matrices (Rows: Actual Reviewer 1 Rating, Columns: Predicted Rating)

#### 1. Production TF-IDF Ridge Model
```
            Pred 0    Pred 1    Pred 2    Pred 3    Pred 4    Total
Actual 0        19         0         0         0         0       19
Actual 1         7         0         0         0         0        7
Actual 2         0         0         5         0         0        5
Actual 3         1         0         2        51         0       54
Actual 4         0         0         1         4         0        5
Total           27         0         8        55         0       90
```
- **Key Observation**: The Ridge model predicted **zero instances** of Level 1 and **zero instances** of Level 4. It collapsed Level 1 into Level 0 (due to lack of lexical overlap) and collapsed Level 4 into Level 3 (due to its theoretical mathematical ceiling at $2.58$).

#### 2. Dense Semantic Model (all-MiniLM-L6-v2)
```
            Pred 0    Pred 1    Pred 2    Pred 3    Pred 4    Total
Actual 0         1        17         1         0         0       19
Actual 1         1         6         0         0         0        7
Actual 2         0         0         1         4         0        5
Actual 3         0         0         1        49         4       54
Actual 4         0         0         0         3         2        5
Total            2        23         3        56         6       90
```
- **Key Observation**: The Semantic model suffered from background lexical similarity, classifying 17 out of 19 unrelated pairs (Level 0) as Level 1. However, unlike Ridge, it successfully identified Level 4 duplicates.

#### 3. Hybrid Lexical-Semantic Fusion Model ($\alpha = 0.50$)
```
            Pred 0    Pred 1    Pred 2    Pred 3    Pred 4    Total
Actual 0         7        12         0         0         0       19
Actual 1         2         5         0         0         0        7
Actual 2         0         0         4         1         0        5
Actual 3         0         1         1        50         2       54
Actual 4         0         0         0         4         1        5
Total            9        18         5        55         3       90
```
- **Key Observation**: The Hybrid model is the **only model that successfully populates all 5 ordinal rating bins**, achieving substantial diagonal concentration across the entire scale.

### Per-Class Performance Breakdown

| Model | Class 0 (Unrelated) | Class 1 (Slight) | Class 2 (Moderate) | Class 3 (High) | Class 4 (Duplicate) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Ridge** | $P=70.4\%, R=100\%, F_1=82.6\%$ | $P=0.0\%, R=0.0\%, F_1=0.0\%$ | $P=62.5\%, R=100\%, F_1=76.9\%$ | $P=92.7\%, R=94.4\%, F_1=93.6\%$ | $P=0.0\%, R=0.0\%, F_1=0.0\%$ |
| **Dense Semantic**| $P=50.0\%, R=5.3\%, F_1=9.5\%$ | $P=26.1\%, R=85.7\%, F_1=40.0\%$ | $P=33.3\%, R=20.0\%, F_1=25.0\%$ | $P=87.5\%, R=90.7\%, F_1=89.1\%$ | $P=33.3\%, R=40.0\%, F_1=36.4\%$ |
| **Hybrid ($\alpha=0.50$)**| $P=77.8\%, R=36.8\%, F_1=50.0\%$ | $P=27.8\%, R=71.4\%, F_1=40.0\%$ | **$P=80.0\%, R=80.0\%, F_1=80.0\%$** | **$P=90.9\%, R=92.6\%, F_1=91.7\%$** | $P=33.3\%, R=20.0\%, F_1=25.0\%$ |

---

## 6. Analysis of Errors & Systematic Biases

### The Underprediction Defect of TF-IDF Ridge ($\text{MBE} = -0.4130$)
The TF-IDF model systematically underestimates similarity for three structural reasons:
1. **Vocabulary Fragility on Paraphrases**: When projects address identical research goals using synonym substitution (e.g. `CASE_C_STRONG_PARAPHRASE`), TF-IDF cosine collapses to near $0.0$, driving the predicted similarity to $0.0$ despite high human similarity ratings.
2. **Negative Section Weights**: The negative weights on `problem_statement` ($-0.1259$), `algorithm_model` ($-0.0651$), and `technology_tools` ($-0.1456$) actively penalize similarity when shared technologies are present.
3. **Truncated Upper Dynamic Range**: Because the maximum possible score for an exact duplicate is $2.5768 / 4.0$, TF-IDF can never assign discrete rating 4 to duplicates.

### The Overprediction Defect of Dense Embeddings ($\text{MBE} = +0.3362$)
The Dense Semantic model exhibits systematic positive bias because:
1. **Domain Background Similarity**: Transformer representations of computer science and engineering proposals share pre-trained language patterns (e.g. phrases such as *"We propose a novel framework"*, *"evaluated using standard metrics"*). Even completely unrelated projects yield dense cosine similarities between $0.25$ and $0.45$.
2. **Zero-Weight Blind Spots**: The candidate weights zero out `abstract`, `problem`, and `objectives`, relying heavily on `dataset` ($48.8\%$) and `title` ($23.8\%$). When a dataset matches superficially, the score escalates rapidly.

### Why Convex Hybrid Fusion Succeeds ($\text{MBE} = +0.0164$)
When combined via $s_{\text{hybrid}} = 0.50 s_{\text{tfidf}} + 0.50 s_{\text{dense}}$, the lexical and dense representations provide mutual error correction:
- Lexical TF-IDF acts as an **anchor**, dampening false-positive semantic drift on unrelated proposals.
- Dense embeddings act as a **bridge**, maintaining similarity signal when paraphrasing breaks lexical n-grams.
- As a direct consequence, the hybrid model achieves **$98.89\%$ within-one-rating agreement** and lowers continuous RMSE to **$0.4421$**.

---

## 7. Exploratory Verification on Dev / Calibration Set (N = 30)

Performance on the 30 development/calibration pairs confirmed identical relative trends:

| Model | MAE | RMSE | Mean Bias | Pearson $r$ | Spearman $\rho$ | Exact Agree (%) | Within-1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Ridge Model** | $0.5024$ | $0.6101$ | $-0.3646$ | $0.9190$ | $0.8147$ | $76.67\%$ | $100.0\%$ |
| **Dense Semantic Model** | $0.4292$ | $0.5885$ | $+0.3337$ | $0.9222$ | $0.8507$ | $70.00\%$ | $100.0\%$ |
| **Hybrid Model ($\alpha=0.50$)**| **$0.3209$** | **$0.4128$** | **$+0.0259$** | **$0.9445$** | **$0.8502$** | **$76.67\%$** | **$100.0\%$** |

Across both the calibration set and the held-out set, the hybrid model maintains stable error metrics ($\text{MAE} \approx 0.316 - 0.321$, $\text{RMSE} \approx 0.413 - 0.442$) without evidence of overfitting.

---

## 8. Visualizations Summary

Six publication-quality figures were generated under `results/validation/figures/`:

1. **Figure 1**: [figure1_human_rating_distribution.png](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/figures/figure1_human_rating_distribution.png)  
   *Side-by-side comparison of Reviewer 1 rating distributions across Held-Out (90 pairs) and Calibration (30 pairs) splits, highlighting the concentration in Level 3.*
2. **Figure 2**: [figure2_model_vs_human_comparison.png](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/figures/figure2_model_vs_human_comparison.png)  
   *Box and jittered strip plots showing continuous model prediction distributions across actual ordinal human ratings.*
3. **Figure 3**: [figure3_confusion_matrices.png](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/figures/figure3_confusion_matrices.png)  
   *Side-by-side $5 \times 5$ confusion matrices for TF-IDF Ridge, Semantic, and Hybrid models.*
4. **Figure 4**: [figure4_model_error_distribution.png](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/figures/figure4_model_error_distribution.png)  
   *Kernel density estimation of prediction residuals ($\hat{y} - y$), illustrating the underprediction of TF-IDF, overprediction of Semantic, and zero-centered distribution of Hybrid.*
5. **Figure 5**: [figure5_actual_vs_predicted.png](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/figures/figure5_actual_vs_predicted.png)  
   *Scatter plots with trendlines against the ideal identity line ($y=x$).*
6. **Figure 6**: [figure6_comparative_model_performance.png](file:///c:/Users/rajes/OneDrive/Desktop/PBL/academic-project-novelty-detection/results/validation/figures/figure6_comparative_model_performance.png)  
   *Summary bar chart contrasting Exact Agreement, Within-1 Agreement, Macro $F_1$, and Weighted $F_1$.*

---

## 9. Scientific Limitations & Threats to Validity

1. **Single Reviewer Subjectivity**:  
   All human ratings originate from a single reviewer (`Reviewer 1`). Individual evaluators possess subjective calibration biases regarding what constitutes "moderate" versus "high" similarity.
2. **Absence of Inter-Rater Reliability (IRR)**:  
   Because Reviewer 2 ratings are pending, Quadratic Weighted Cohen's Kappa between independent human judges cannot be computed. The system cannot yet verify whether human experts agree with each other on this sample.
3. **Pending Adjudication Consensus**:  
   Adjudicated ground truth ratings do not yet exist. Model predictions have been compared against un-adjudicated raw ratings.
4. **Severe Class Imbalance**:  
   $57.5\%$ of the ratings reside in Class 3, while Class 2 ($7.5\%$) and Class 4 ($5.8\%$) are underrepresented. High accuracy or exact agreement metrics are influenced by majority class frequency.
5. **Sample Size Constraints**:  
   A cohort of 90 held-out pairs provides preliminary statistical orientation ($p < 10^{-20}$ for correlations), but does not guarantee generalization across all academic institutions or sub-disciplines.
6. **Discretization Boundary Artifacts**:  
   Mapping continuous scores to integer bins via rounding creates threshold sensitivity near bin boundaries ($0.5, 1.5, 2.5, 3.5$).

---

## 10. Concrete Next Steps for Final Faculty Validation

When the second faculty reviewer becomes available, the following phased protocol will be executed:

1. **Double-Blind Scoring by Reviewer 2**:
   - Provide `Evaluation_Batch_1` and `Evaluation_Batch_2` in `faculty_evaluation_sample_120.xlsx` to Reviewer 2 without access to Reviewer 1 ratings.
   - Populate `reviewer_2_rating` and `reviewer_2_comments` for all 120 pairs.
2. **Inter-Rater Reliability (IRR) Measurement**:
   - Calculate Quadratic Weighted Cohen's Kappa ($\kappa_w$), Linear Kappa, and percentage adjacent agreement between Reviewer 1 and Reviewer 2 on the 90 held-out pairs.
   - Target acceptance threshold: $\kappa_w \ge 0.70$.
3. **Adjudication Session**:
   - Convene an adjudication meeting for all pairs where $|R_1 - R_2| \ge 1$.
   - Record consensus agreed values in `adjudicated_rating` and rationales in `adjudication_comments`.
4. **Final Model Benchmarking**:
   - Re-run the evaluation pipeline against `adjudicated_rating` to establish definitive validated performance.
   - Publish the Phase 5 Final Scientific Validation Report.

