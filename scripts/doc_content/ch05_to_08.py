"""
Chapters 5 to 8 of the Complete Technical Machine Learning Documentation.
"""

CHAPTER_5_TITLE = "Chapter 5: What the ML Model Learns"
CHAPTER_5_TEXT = """================================================================================
CHAPTER 5: WHAT THE MACHINE LEARNING MODEL LEARNS
================================================================================

5.1 The Two-Stage Architecture Overview
---------------------------------------
The system uses a two-stage hybrid machine learning architecture:
  - Stage 1 (Lexical Representation): Section-wise TF-IDF (Term Frequency-Inverse Document Frequency) vectorization with unigram and bigram n-grams and stop-word filtering.
  - Stage 2 (Multivariate Fusion & Calibration): L2-regularized Ridge Regression that takes the eight section cosine similarities as input and predicts an overall similarity score on a continuous [0.0, 4.0] scale.

5.2 Deep Dive into TF-IDF (Term Frequency-Inverse Document Frequency)
---------------------------------------------------------------------
TF-IDF is an established natural language processing technique that converts text strings into sparse numerical vectors reflecting the statistical importance of words.

Key Definitions for Beginners:
  - Token: A single word or sub-word unit extracted from text (e.g., "convolutional", "neural", "plantvillage").
  - Document: In this system, each individual section of a project is treated as a separate document (e.g., the `abstract` of Project SYN-0001 is Document 1 for the Abstract Vectorizer).
  - Corpus: The collection of all 600 documents for that section across the entire reference dataset.
  - Vocabulary: The set of all unique recognized unigrams and bigrams learned by the vectorizer during fitting.

Mathematical Formulation (as implemented in scikit-learn):
1. Term Frequency ($TF(t, d)$):
   Measures how frequently term $t$ appears in document $d$:
                 TF(t, d) = count of term t in document d

2. Inverse Document Frequency ($IDF(t, D)$):
   Measures how rare or informative term $t$ is across all $N$ documents in the corpus:
                 IDF(t) = ln( (1 + N) / (1 + DF(t)) ) + 1.0
   Where:
     * $N = 600$ (total number of projects in the reference corpus).
     * $DF(t)$ is the document frequency (the number of projects whose section contains term $t$).
     * The `+1` terms represent Laplace smoothing (`smooth_idf=True`), preventing division by zero.
   Interpretation: Words appearing in almost every project (e.g., "model", "system") receive a low IDF weight. Rare technical terms appearing in only three projects (e.g., "plantvillage", "vit") receive a high IDF weight.

3. Raw TF-IDF Weight:
                 raw_tfidf(t, d) = TF(t, d) * IDF(t)

4. L2 Normalization (Euclidean Length Normalization):
   Each document vector is divided by its Euclidean norm ($L_2$ norm) so that longer texts do not produce artificially inflated cosine similarities:
                 v_norm = v / sqrt( sum( v_i^2 ) )

5. N-Grams: Unigrams vs. Bigrams:
   The configuration specifies `ngram_range=(1, 2)`.
   - Unigram (1-gram): Single words (e.g., "neural", "network").
   - Bigram (2-gram): Contiguous sequences of two words (e.g., "neural network", "disease detection").
   Bigrams capture contextual domain phrasing that unigrams lose.

5.3 Exact Learned TF-IDF Parameters & Vocabulary Sizes
-------------------------------------------------------
Inspection of `models/section_tfidf_extractor.joblib` reveals the exact vocabulary sizes learned by the 8 vectorizers:

+----------------------+--------------------+-----------------+------------------+
| Section Key          | Source Column      | N-Gram Range    | Vocabulary Size  |
+----------------------+--------------------+-----------------+------------------+
| title                | project_title      | (1, 2)          | 477 n-grams      |
| abstract             | abstract           | (1, 2)          | 454 n-grams      |
| problem              | problem_statement  | (1, 2)          | 199 n-grams      |
| objectives           | objectives         | (1, 2)          | 188 n-grams      |
| methodology          | proposed_method    | (1, 2)          | 227 n-grams      |
| dataset              | dataset            | (1, 2)          | 83 n-grams       |
| algorithm            | algorithm_model    | (1, 2)          | 19 n-grams       |
| technology           | technology_tools   | (1, 2)          | 89 n-grams       |
+----------------------+--------------------+-----------------+------------------+
| Total Combined       | All 8 Sections     |                 | 1,736 n-grams    |
+----------------------+--------------------+-----------------+------------------+

Stop Words Configuration:
All eight vectorizers use scikit-learn's built-in English stop-word dictionary (318 common English filler words such as "the", "is", "at", "which").

5.4 Deep Dive into Ridge Regression (L2-Regularized Linear Fusion)
-----------------------------------------------------------------
Why Ridge Regression?
Why not use standard Ordinary Least Squares (OLS) or a Deep Neural Network?
  1. Interpretability: A linear model produces exact, human-understandable additive weights for each section. If the predicted similarity is 2.12, the system can prove that 1.93 came from the Dataset, 0.20 came from the Abstract, and so forth. Deep neural networks are opaque black boxes.
  2. Multicollinearity: The eight section similarities are correlated (projects sharing an abstract often share methodology and algorithms). Standard OLS regression becomes unstable under correlated features. Ridge regression stabilizes weights using L2 regularization.

Mathematical Objective of Ridge Regression:
Given training feature matrix $X \in \mathbb{R}^{n \times 8}$ and target similarity labels $y \in [0, 4]^n$, Ridge regression solves:

              min_w { || y - (Xw + b) ||_2^2 + alpha * || w ||_2^2 }

Where:
  - $|| y - (Xw + b) ||_2^2 = \sum_{j=1}^n (y_j - (\sum_{i=1}^8 w_i x_{ji} + b))^2$ is the sum of squared prediction errors.
  - $alpha * || w ||_2^2 = alpha * \sum_{i=1}^8 w_i^2$ is the L2 penalty that shrinks coefficients toward zero to prevent overfitting.
  - Closed-form solution:
              w = (X^T X + alpha * I)^{-1} X^T (y - mean(y))

5.5 Actual Fitted Model Parameters
----------------------------------
Inspection of `models/ridge_fusion_model.joblib` reveals:
  - Model Class: `sklearn.linear_model.Ridge`
  - Regularization Parameter: `alpha = 1.0`
  - Intercept ($b$ or $\beta_0$): `-0.07063171692234005`
  - Number of Features: Exactly 8 (one per proposal section)
  - Fitted Coefficients ($w_i$):
      * `title_similarity`:       +0.13404783
      * `abstract_similarity`:    +0.39804354
      * `problem_similarity`:     -0.12586323
      * `objectives_similarity`:  +0.23673393
      * `methodology_similarity`: +0.00989785
      * `dataset_similarity`:     +2.20544639
      * `algorithm_similarity`:   -0.06507632
      * `technology_similarity`:  -0.14564642
  - Sum of Coefficients ($\sum w_i$): +2.64758357 (~ +2.6476)
"""

CHAPTER_6_TITLE = "Chapter 6: How the Eight Sections Are Combined"
CHAPTER_6_TEXT = """================================================================================
CHAPTER 6: HOW THE EIGHT SECTIONS ARE COMBINED
================================================================================

6.1 Section Independence & Specialized Vectorizers
--------------------------------------------------
A crucial architectural decision in this system is that sections are NOT pooled into a single combined text blob prior to feature extraction.

Why?
In Notebook 02 (`02_tfidf_baseline.ipynb`), all eight sections were concatenated into a single string. This produced poor diagnostic resolution: if two projects shared a few technology words ("Python", "TensorFlow"), the global TF-IDF score could not distinguish whether the overlap occurred in the core scientific methodology or merely in incidental programming tools.

Therefore, in Notebook 03 (`03_section_wise_tfidf.ipynb`) and production (`src/features/section_tfidf.py`), eight independent `TfidfVectorizer` instances are maintained:
  - Vectorizer 1: Fitted ONLY on `project_title` texts.
  - Vectorizer 2: Fitted ONLY on `abstract` texts.
  - Vectorizer 3: Fitted ONLY on `problem_statement` texts.
  - Vectorizer 4: Fitted ONLY on `objectives` texts.
  - Vectorizer 5: Fitted ONLY on `proposed_methodology` texts.
  - Vectorizer 6: Fitted ONLY on `dataset` texts.
  - Vectorizer 7: Fitted ONLY on `algorithm_model` texts.
  - Vectorizer 8: Fitted ONLY on `technology_tools` texts.

6.2 Computing Section-by-Section Cosine Similarities
----------------------------------------------------
When comparing Project A and Project B, the system transforms each text section into an L2-normalized TF-IDF vector and computes cosine similarity:

              s_i = cos( v_A^{(i)}, v_B^{(i)} ) = ( v_A^{(i)} . v_B^{(i)} ) / ( ||v_A^{(i)}|| * ||v_B^{(i)}|| )

Because vectors are already L2-normalized ($||v|| = 1.0$), this simplifies to the dot product:
              s_i = sum_k ( v_{A,k}^{(i)} * v_{B,k}^{(i)} )

Each section similarity $s_i$ is bounded in the interval $[0.0, 1.0]$:
  - $s_i = 1.0$: Identical lexical vocabulary and proportional frequency in that section.
  - $s_i = 0.0$: Zero shared vocabulary (completely disjoint n-grams).

6.3 The 8-Dimensional Feature Representation
---------------------------------------------
The eight scalar section similarities are concatenated into a single feature vector $\\vec{x} \in [0.0, 1.0]^8$:

    \\vec{x} = [ s_title, s_abstract, s_problem, s_objectives, s_methodology, s_dataset, s_algorithm, s_technology ]^T

6.4 Distinction: Manually Configured vs. Machine-Learned Section Weights
------------------------------------------------------------------------
In literature and simple heuristic projects, researchers often assign ad-hoc manual weights (e.g., Title = 10%, Abstract = 30%, Dataset = 20%, etc.).

In this system:
  - NO MANUAL WEIGHTS ARE USED IN SCORING.
  - The weights are LEARNED REGRESSION COEFFICIENTS ($w_i$) computed by scikit-learn's `Ridge(alpha=1.0)` on the training dataset.
  - The model decided its own weights based entirely on which section similarities best minimized prediction error against the target labels.

Comparison of Weight Concepts:
+------------------------+-----------------------------------------------------------------------------+
| Weight Type            | Origin and Function                                                         |
+------------------------+-----------------------------------------------------------------------------+
| Manually Assigned      | Fixed constants chosen by human intuition (e.g., w_title = 0.15).           |
| Weight                 | NOT USED in this system's scoring engine.                                   |
+------------------------+-----------------------------------------------------------------------------+
| Learned Ridge          | Statistically optimized coefficients ($w_i$) learned by Ridge regression    |
| Coefficient            | to minimize squared error on pairwise labels. (e.g., w_dataset = +2.2054).  |
+------------------------+-----------------------------------------------------------------------------+
| Signed Feature         | The product of a specific section's cosine similarity and its Ridge weight: |
| Contribution           | contribution_i = s_i * w_i. Used in Explainable AI (XAI) reports.           |
+------------------------+-----------------------------------------------------------------------------+
"""

CHAPTER_7_TITLE = "Chapter 7: Training Process, Data Splits & Cross-Validation"
CHAPTER_7_TEXT = """================================================================================
CHAPTER 7: TRAINING PROCESS, DATA SPLITS & CROSS-VALIDATION
================================================================================

7.1 Labeled Data Scale
-----------------------
- Total unique projects in reference database: 600 projects (`SYN-0001` to `SYN-0600`).
- Total cleaned pairwise comparisons: 2,996 pairs (after deduplication).
- Target variable: `expert_similarity_label` $\in \{0, 1, 2, 3, 4\}$.

7.2 The Project-Disjoint Partitioning Protocol
----------------------------------------------
The most vital principle in pairwise similarity modeling is avoiding project identity leakage. If an experiment randomly splits the 2,996 pair rows into 80% train and 20% test:
  - Project SYN-0015 will appear in the training pairs (e.g., SYN-0015 vs SYN-0040).
  - Project SYN-0015 will ALSO appear in the test pairs (e.g., SYN-0015 vs SYN-0112).
  - The model can memorize specific project vocabulary and characteristics during training, creating falsely optimistic evaluation metrics.

To ensure scientific honesty, `scripts/optimize_models.py` and `scripts/evaluate_holdout.py` implement `project_disjoint_split` in `src/utils/metrics.py`:
  1. The 600 unique projects are randomly shuffled using `np.random.RandomState(42)`.
  2. The projects are partitioned into:
       * Training Projects: 480 projects (80.0%)
       * Test Projects: 120 projects (20.0%)
  3. Every pair in the dataset is filtered:
       * Train Pairs: Both Project A AND Project B must be in the 480 training projects.
         Result: Exactly 1,916 pairs.
       * Test Pairs: Both Project A AND Project B must be in the 120 test projects.
         Result: Exactly 120 pairs.
       * Cross-Boundary Pairs: Pairs where one project is in Train and the other is in Test are DISCARDED.
         Result: Exactly 960 pairs discarded.
  4. Total accountability: $1,916 + 120 + 960 = 2,996$ clean pairs.

7.3 Leakage-Free TF-IDF Fitting
-------------------------------
In rigorous machine learning, feature extractors must be fitted ONLY on training documents.
Auditing `scripts/evaluate_holdout.py` (lines 114–135) confirms:
  - `TfidfVectorizer.fit()` is called ONLY on the 480 training projects.
  - Vocabulary learned on `project_title`: exactly 450 n-grams.
  - Exactly 27 technical n-grams that occur exclusively in the 120 test projects were omitted from the vocabulary.
  - Zero vocabulary or inverse document frequency leakage occurred from test projects into the feature pipeline.

7.4 Five-Fold Project-Disjoint Cross-Validation
------------------------------------------------
In `scripts/optimize_models.py`, hyperparameter optimization was conducted using 5-fold project-disjoint cross-validation:
  - The 600 project IDs were split into 5 disjoint folds of 120 projects each (`n_splits=5`, `seed=42`).
  - In each fold, 4 folds (480 projects, ~1,915 pairs) formed the training set; 1 fold (120 projects, ~117 pairs) formed the validation set.
  - Evaluated candidate alphas for Ridge regression: $\alpha \in [0.01, 0.1, 1.0, 10.0, 100.0]$.
  - Cross-validation results across alphas:
      * $\alpha = 0.01$:  Mean RMSE = 0.6558, Mean MAE = 0.4770, Mean R2 = 0.7705
      * $\alpha = 0.1$:   Mean RMSE = 0.6557, Mean MAE = 0.4768, Mean R2 = 0.7706
      * $\alpha = 1.0$:   Mean RMSE = 0.6548, Mean MAE = 0.4761, Mean R2 = 0.7712
      * $\alpha = 10.0$:  Mean RMSE = 0.6511, Mean MAE = 0.4731, Mean R2 = 0.7737 (Best CV)
      * $\alpha = 100.0$: Mean RMSE = 0.6703, Mean MAE = 0.4901, Mean R2 = 0.7602

7.5 Audit Disclosure: The Relationship Between the Holdout and Fold 5
---------------------------------------------------------------------
As documented in the scientific audit (`results/validation/scientific_reliability_audit.md`):
  - In `scripts/evaluate_holdout.py`, the 80/20 train/test project partition used `RandomState(42)`.
  - In `scripts/optimize_models.py`, `KFold(n_splits=5, shuffle=True, random_state=42)` partitioned the identical sorted list of projects.
  - Under scikit-learn's `KFold`, Fold 5 assigns the exact same project index slice (`480:600`) as the test split!
  - Therefore, the holdout test set is mathematically identical to Fold 5 of the cross-validation.
  - Because $\alpha = 10.0$ was selected using the 5-fold average (which included Fold 5), the holdout partition was NOT an untouched holdout in the strictest machine learning sense. It had indirect exposure during model selection.
  - Furthermore, in the final production artifact `models/ridge_fusion_model.joblib`, the saved model was trained with `alpha=1.0`.

7.6 Per-Fold Performance Breakdown (Why Holdout RMSE is 0.7734)
---------------------------------------------------------------
The holdout evaluation reported $\text{RMSE} = 0.7734$, which is higher than the 5-fold mean ($\text{RMSE} = 0.6511$).
Inspecting `results/model_optimization/fold_results.csv` explains this discrepancy completely:

+---------+-------------------+------------------+----------+----------+----------+-----------------+
| Fold    | Train Pairs Count | Held-out Pairs   | RMSE     | MAE      | R2 Score | Spearman rho    |
+---------+-------------------+------------------+----------+----------+----------+-----------------+
| Fold 1  | 1,876 pairs       | 117 pairs        | 0.5854   | 0.4524   | 0.8351   | 0.8195          |
| Fold 2  | 1,940 pairs       | 100 pairs        | 0.6035   | 0.4124   | 0.8040   | 0.8130          |
| Fold 3  | 1,908 pairs       | 127 pairs        | 0.5707   | 0.4133   | 0.8213   | 0.8305          |
| Fold 4  | 1,935 pairs       | 123 pairs        | 0.7227   | 0.5223   | 0.7402   | 0.7501          |
| Fold 5  | 1,916 pairs       | 120 pairs        | 0.7734   | 0.5648   | 0.6677   | 0.7508          |
+---------+-------------------+------------------+----------+----------+----------+-----------------+
| Mean    | 1,915 pairs       | 117.4 pairs      | 0.6511   | 0.4731   | 0.7737   | 0.7928          |
| Std Dev |                   |                  | +-0.0910 | +-0.0680 | +-0.0695 | +-0.0391        |
+---------+-------------------+------------------+----------+----------+----------+-----------------+

Conclusion: Fold 5 happens to be the most challenging partition in the dataset ($1.34\sigma$ above the mean). The holdout test result ($0.7734$) reflects evaluating on Fold 5.
"""

CHAPTER_8_TITLE = "Chapter 8: All Calculations and Mathematical Formulas"
CHAPTER_8_TEXT = """================================================================================
CHAPTER 8: ALL CALCULATIONS AND MATHEMATICAL FORMULAS
================================================================================

8.1 Complete Mathematical Inventory
-----------------------------------
Every calculation executed by the system from text input to the final novelty display is documented below with exact formulas.

8.2 Formula 1: TF-IDF Vectorization
-----------------------------------
For each section $k \in \{1, \dots, 8\}$ and term $t$ in text $d$:
              TF(t, d) = count(t, d)
              IDF(t) = ln( (1 + 600) / (1 + DF(t)) ) + 1.0
              w_raw(t, d) = TF(t, d) * IDF(t)
              v_k = w_raw / sqrt( sum_j (w_raw_j^2) )

8.3 Formula 2: Section-Level Cosine Similarity
----------------------------------------------
Between candidate vector $v_{cand}^{(k)}$ and reference vector $v_{ref}^{(k)}$:
              s_k = v_{cand}^{(k)} . v_{ref}^{(k)} = sum_j ( v_{cand, j}^{(k)} * v_{ref, j}^{(k)} )
Where $s_k \in [0.0, 1.0]$.

8.4 Formula 3: Additive Ridge Regression Raw Prediction
-------------------------------------------------------
              y_raw = beta_0 + sum_{k=1}^8 ( w_k * s_k )

Using the exact fitted model parameters from `models/ridge_fusion_model.joblib`:
  y_raw = -0.07063172
        + (0.13404783 * s_title)
        + (0.39804354 * s_abstract)
        - (0.12586323 * s_problem)
        + (0.23673393 * s_objectives)
        + (0.00989785 * s_methodology)
        + (2.20544639 * s_dataset)
        - (0.06507632 * s_algorithm)
        - (0.14564642 * s_technology)

8.5 Formula 4: Section Signed Model Contribution (XAI)
------------------------------------------------------
For each section $k$:
              contrib_k = w_k * s_k

Reconstruction Invariant (checked in unit tests):
              y_raw == beta_0 + sum_{k=1}^8 contrib_k  (within 1e-5 floating point tolerance)

8.6 Formula 5: Prediction Bounding / Clipping
---------------------------------------------
Because linear regression is unconstrained and can theoretically output values $< 0.0$ or $> 4.0$, predictions are clipped to the valid target interval:
              y_clipped = clip( y_raw, 0.0, 4.0 )
              y_clipped = max( 0.0, min( 4.0, y_raw ) )

8.7 Formula 6: Maximum Historical Similarity Selection
------------------------------------------------------
Across all $M$ eligible reference corpus projects (strictly excluding candidate self-matches):
              s_max = max_{j in {1, ..., M}} { y_{clipped, j} }

8.8 Formula 7: Estimated Novelty Percentage
-------------------------------------------
In `src/utils/metrics.py` (lines 88–95):
              sim_norm = clip( s_max / 4.0, 0.0, 1.0 )
              Novelty (%) = (1.0 - sim_norm) * 100.0

Equivalently:
              Novelty (%) = 100.0 * ( 1.0 - (s_max / 4.0) )

8.9 Why 4.0 Is Used as the Denominator
--------------------------------------
The denominator `4.0` is NOT an arbitrary constant. It represents the maximum possible ground-truth integer rating in the benchmark scale:
  - 0 = Not Similar
  - 1 = Slightly Similar
  - 2 = Moderately Similar
  - 3 = Highly Similar
  - 4 = Nearly Duplicate (Identical)

Dividing by 4.0 normalizes the 0-to-4 regression score into a [0.0, 1.0] fractional similarity ratio before subtracting from 1.0 to obtain novelty.

8.10 Analysis of Extreme Values
-------------------------------
  - When $s_{max} = 0.0$:
    sim_norm = 0.0 / 4.0 = 0.0 -> Novelty = (1.0 - 0.0) * 100% = 100.0% (Maximum possible novelty).
  - When $s_{max} = 1.0$:
    sim_norm = 1.0 / 4.0 = 0.25 -> Novelty = (1.0 - 0.25) * 100% = 75.0%.
  - When $s_{max} = 2.0$:
    sim_norm = 2.0 / 4.0 = 0.50 -> Novelty = (1.0 - 0.50) * 100% = 50.0%.
  - When $s_{max} = 3.0$:
    sim_norm = 3.0 / 4.0 = 0.75 -> Novelty = (1.0 - 0.75) * 100% = 25.0%.
  - When $s_{max} = 4.0$:
    sim_norm = 4.0 / 4.0 = 1.0 -> Novelty = (1.0 - 1.0) * 100% = 0.0% (Zero novelty / complete duplicate).
  - When $s_{max} < 0.0$ (out-of-range negative prediction):
    Clipping clamps to 0.0 -> Novelty = 100.0%.
  - When $s_{max} > 4.0$ (out-of-range positive prediction):
    Clipping clamps to 4.0 -> Novelty = 0.0%.

8.11 Novelty Assessment Tier Thresholds
---------------------------------------
In `src/utils/metrics.py` (lines 97–107), the novelty percentage is mapped to four standardized qualitative tiers:

+----------------------------+--------------------------------------------------------------+
| Novelty Percentage Range   | Exact Assessment Tier String                                 |
+----------------------------+--------------------------------------------------------------+
| 75.0% <= Novelty <= 100.0% | "High estimated novelty relative to the current database"   |
| 50.0% <= Novelty < 75.0%   | "Moderate Novelty (Partial Overlap)"                         |
| 25.0% <= Novelty < 50.0%   | "Low Novelty (Significant Prior Work)"                       |
| 0.0% <= Novelty < 25.0%    | "Very Low Novelty / Potential Duplicate"                     |
+----------------------------+--------------------------------------------------------------+

Rounding and Precision Standards:
  - Similarity scores: Rounded to 4 decimal places in API responses (`round(score, 4)`).
  - Novelty percentages: Rounded to 2 decimal places in API responses (`round(pct, 2)`) and formatted to 1 decimal place in the UI (`pct.toFixed(1) + "%"`).
"""

print("Writing Chapters 5 to 8 to text...")
