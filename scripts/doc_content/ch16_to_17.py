"""
Chapters 16 and 17 of the Complete Technical Machine Learning Documentation.
"""

CHAPTER_16_TITLE = "Chapter 16: Comprehensive Viva & Project Evaluation Q&A"
CHAPTER_16_TEXT = """================================================================================
CHAPTER 16: COMPREHENSIVE VIVA & PROJECT EVALUATION Q&A
================================================================================

This chapter contains 36 thorough, technically rigorous questions and answers prepared specifically for final-year project evaluations, external examiners, and viva voce defense.

Q1: What is the primary objective of your Academic Project Novelty Detection System?
A1: The objective is to assist academic project review panels and students by evaluating technical proposal novelty against an indexed institutional corpus of prior student work. It breaks proposals into 8 structured sections, computes section-by-section lexical similarities, feeds them to a regularized Ridge regression model, and generates transparent Explainable AI (XAI) evidence.

Q2: What are the eight proposal sections and why are they separated?
A2: The eight sections are Title, Abstract, Problem Statement, Objectives, Proposed Methodology, Dataset, Algorithm/Model, and Technology Tools. They are separated because capstone projects have distinct architectural components. Concatenating all text into one blob conceals where overlap occurs; separating them allows the model to evaluate methodology or dataset overlap independently from generic programming tools.

Q3: What machine learning model is used for similarity scoring?
A3: The system uses an L2-regularized Ridge Regression model (`alpha=1.0`) trained on eight section-level TF-IDF cosine similarities to predict an overall similarity score on a continuous [0.0, 4.0] scale.

Q4: What is TF-IDF and how does it work?
A4: TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a numerical statistic that reflects how important a word is to a specific document within a collection. Term Frequency (TF) measures how often a word occurs in the text, while Inverse Document Frequency (IDF) penalizes words that appear across all projects (e.g., "system", "model") and boosts rare, informative terms (e.g., "plantvillage", "fedavg").

Q5: What are n-grams and why did you choose unigrams and bigrams?
A5: An n-gram is a contiguous sequence of n tokens. A unigram is 1 word ("neural"), while a bigram is 2 consecutive words ("neural network"). Unigrams capture standalone keywords, while bigrams capture technical context and phrases that lose meaning when split.

Q6: Why did you use eight separate TF-IDF vectorizers instead of one shared vectorizer?
A6: Each section has a completely different vocabulary distribution, typical document length, and technical purpose. The `dataset` section contains benchmark proper nouns (83 n-grams), while the `abstract` contains high-level research summaries (454 n-grams). Fitting separate vectorizers ensures document frequencies and IDF weights accurately reflect section-specific importance.

Q7: What is Ridge Regression and why did you choose it over deep neural networks?
A7: Ridge regression is a linear model that minimizes the sum of squared residuals while adding an L2 penalty on coefficient magnitudes ($||w||_2^2$). It was chosen because:
  1. Complete Interpretability: Every prediction can be decomposed into additive section contributions ($w_k \times s_k$), satisfying academic transparency.
  2. Multicollinearity Stability: The 8 section similarities are correlated. Ridge regression stabilizes weights where standard OLS regression becomes erratic.
  3. High Sample Efficiency: Deep neural networks require tens of thousands of real samples and act as opaque black boxes.

Q8: What does the alpha parameter do in Ridge Regression?
A8: Alpha ($\alpha$) is the regularization hyperparameter. It controls the penalty on large coefficients. When $\alpha = 0$, Ridge simplifies to ordinary least squares (OLS). As $\alpha$ increases, coefficients are shrunk closer to zero, reducing model variance and preventing overfitting. Through 5-fold cross-validation, $\alpha=1.0$ and $\alpha=10.0$ were found to provide optimal generalization.

Q9: What is the target variable in your regression model?
A9: The target variable is `expert_similarity_label`, an integer rating from 0 to 4:
  - 0: Not Similar
  - 1: Slightly Similar
  - 2: Moderately Similar
  - 3: Highly Similar
  - 4: Nearly Duplicate (Identical)

Q10: How many projects and pairs are in your dataset?
A10: The dataset contains 600 unique projects (200 UG Mini, 200 UG Mega, 200 PG Dissertations) across 12 domains. There were originally 3,000 similarity pairs, which were deduplicated to 2,996 clean pairs.

Q11: Are the similarity labels real human faculty annotations or synthetic?
A11: They are 100% synthetic heuristic benchmark labels. The raw dataset explicitly notes: "SYNTHETIC BENCHMARK LABEL — must be replaced/validated by faculty annotation before publication." Across all 13 sheets of `Faculty_Verification_DomainWise.xlsx`, zero faculty evaluations are filled. This is a critical audit disclosure.

Q12: How did you clean the dataset? What duplicate pairs were removed?
A12: In `scripts/clean_dataset.py`, canonical undirected tuples `(min(A, B), max(A, B))` were audited. Exactly 4 redundant rows were removed: 1 exact duplicate (`PAIR-1819`) and 3 reversed duplicate pairs (`PAIR-2023`, `PAIR-2195`, `PAIR-2957`). Both counterparts had 100% matching labels (all label 0).

Q13: What is data leakage and how was it prevented during train/test splitting?
A13: Data leakage occurs when test set information inadvertently influences training. If pairwise rows are randomly split, the same project could appear in both training and test pairs, allowing the model to memorize project vocabulary. We implemented `project_disjoint_split`, which partitions the 600 unique project IDs (480 train, 120 test). Cross-boundary pairs (960 pairs) were discarded.

Q14: How did 5-fold cross-validation work in this project?
A14: The 600 project IDs were split into 5 disjoint folds of 120 projects each (`optimize_models.py`). In each fold, 480 projects formed the training pairs (~1,915 pairs) and 120 projects formed the validation pairs (~117 pairs). TF-IDF vectorizers were fitted strictly on the 480 training projects per fold.

Q15: Why is the holdout RMSE (0.7734) higher than the 5-fold CV mean (0.6511)?
A15: The holdout test set corresponds to Fold 5 of the cross-validation. Individual fold RMSEs ranged from 0.5707 (Fold 3) to 0.7734 (Fold 5). Fold 5 represents the most challenging partition in the data. The 5-fold mean (0.6511) averages across easy and hard folds, whereas the holdout evaluated on Fold 5.

Q16: How are the section weights decided? Did you choose them manually?
A16: The section weights were NOT chosen manually. They were mathematically learned by scikit-learn's `Ridge(alpha=1.0)` to minimize squared prediction errors on the training dataset.

Q17: What are the actual learned weights in your production model?
A17: From `models/ridge_fusion_model.joblib`:
  - Intercept: -0.0706
  - `dataset`: +2.2054
  - `abstract`: +0.3980
  - `objectives`: +0.2367
  - `title`: +0.1340
  - `methodology`: +0.0099
  - `algorithm`: -0.0651
  - `problem`: -0.1259
  - `technology`: -0.1456

Q18: Why does the dataset section have such an enormous weight (+2.2054)?
A18: This is a synthetic dataset artifact. The rule-based generator that created the labels almost always assigned identical datasets to High-Similarity pairs and different datasets to Low-Similarity pairs. The Ridge algorithm discovered that dataset overlap was the strongest statistical predictor of the synthetic label and assigned it over 55% of the total predictive weight.

Q19: Why are some weights negative (e.g., technology = -0.1456)?
A19: This is a suppressor variable effect caused by multicollinearity. Generic tools ("Python", "TensorFlow") and standard algorithms ("CNN") appear ubiquitously across hundreds of completely unrelated projects. If the model had positive weights on tools, unrelated projects would receive false overlap scores. The negative weight suppresses this generic noise.

Q20: What is the mathematical formula for the novelty score?
A20: `Novelty (%) = 100.0 * (1.0 - (max_predicted_similarity / 4.0))`, where `max_predicted_similarity` is clipped to [0.0, 4.0].

Q21: Why is 4.0 used as the denominator in the novelty formula?
A21: Because the regression model was trained on a 0 to 4 ground-truth scale (where 4 represents an identical/duplicate project). Dividing by 4.0 normalizes the score to a [0.0, 1.0] fractional similarity ratio before subtracting from 1.0 to obtain novelty.

Q22: What are the four novelty assessment tiers and their thresholds?
A22:
  - 75.0% - 100.0%: "High estimated novelty relative to the current database"
  - 50.0% - 74.9%: "Moderate Novelty (Partial Overlap)"
  - 25.0% - 49.9%: "Low Novelty (Significant Prior Work)"
  - 0.0% - 24.9%: "Very Low Novelty / Potential Duplicate"

Q23: What is the theoretical maximum prediction ceiling of your model?
A23: With an intercept of -0.0706 and coefficient sum of +2.6476, the maximum raw prediction for an exact duplicate is $-0.0706 + 2.6476 = 2.5768$ (out of 4.0). This maps to a minimum novelty of $35.58\%$. The model can never output a score below 25%, meaning the "Very Low Novelty / Potential Duplicate" tier is mathematically unreachable under current weights!

Q24: Why did an exact copy of a project rank #6 behind near-peers during the audit?
A24: Because of negative weights on `technology_tools` (-0.1456) and `problem_statement` (-0.1259). An exact copy suffers a -0.3366 penalty for sharing 100% of the tools and problem statement, whereas near-peers with differing tools avoid this penalty and score up to 2.6551. This is a non-monotonicity limitation.

Q25: What is candidate self-exclusion and how does it work?
A25: When evaluating a proposal, the system must not match a project against itself. `NoveltyExplainer` checks:
  1. Case-insensitive project ID match.
  2. Case-insensitive exact title match.
  3. Identical feature vector match (all 8 similarities >= 0.9999).
Matching records are masked out, ensuring the candidate is only evaluated against prior peer projects.

Q26: What is Explainable AI (XAI) in your system and how is it implemented?
A26: XAI provides human-understandable evidence explaining why a prediction was made. Our system implements four layers:
  1. Ranked Sections Table (ordering sections by overlap degree).
  2. Additive Model Attributions (displaying exact $w_k \times s_k$ contributions).
  3. Shared Informative Terms (displaying common technical n-grams filtered for stop words).
  4. Source-Verified Snippets (displaying verbatim text windows centered around shared keywords with clean word boundaries).

Q27: How do you extract and verify keyword snippets?
A27: `extract_snippet` extracts a 45-character window around the keyword and snaps outward to the nearest whitespace to prevent mid-word truncations. `verify_snippet` strictly confirms that the snippet body is an exact substring of the original source record.

Q28: How does the system handle user proposal submissions without contaminating the database?
A28: Scoring a proposal via `/api/score` is strictly read-only and in-memory. If a user explicitly clicks "Save Proposal", `/api/save` writes the proposal and its report to a separate SQLite database (`data/proposals.db`). The 600 baseline reference projects in Joblib are never mutated.

Q29: How does the system handle duplicate submissions in the database?
A29: In `src/storage/db.py`, before inserting a new proposal, the database checks for an existing record with the same normalized alphanumeric title. If found, it raises `DuplicateProposalError` and FastAPI returns HTTP 409 Conflict.

Q30: Why does your model fail when a proposal is paraphrased?
A30: Because TF-IDF is a lexical bag-of-words model that measures exact word matching. When concepts are paraphrased using different synonyms, TF-IDF n-gram overlap drops to near zero, causing the model to output a false 99.6% novelty score.

Q31: What is the single most important limitation of your system?
A31: 100% of the ground-truth similarity labels are synthetic heuristic rules, and zero real faculty evaluations exist in the dataset. High evaluation metrics simply reflect the model learning the synthetic labeling rule.

Q32: How would you improve and scientifically validate this project in the future?
A32:
  1. Collect real faculty annotations for 300 project pairs in `Faculty_Verification_DomainWise.xlsx`.
  2. Retrain using Non-Negative Least Squares (NNLS) to constrain $w_i \ge 0$, fixing non-monotonicity.
  3. Calibrate the output scale so identical projects map to 4.0 (0% novelty).
  4. Integrate dense Sentence-Transformers (`all-MiniLM-L6-v2` or `Specter2`) to detect paraphrasing.
  5. Evaluate on a true, sealed holdout of real student projects.

Q33: Can your system be used to prove plagiarism or accuse a student of copying?
A33: Absolutely not. The system estimates lexical similarity relative to a local reference database. It cannot establish intent, uncredited copying, or academic dishonesty. Shared terminology often occurs naturally because students use standard algorithms or datasets.

Q34: How fast does the model evaluate a proposal at runtime?
A34: Because TF-IDF vectorization and Ridge regression are simple matrix operations, transforming 8 sections and computing cosine similarity against 600 projects takes less than 35 milliseconds.

Q35: What libraries are used in the backend?
A35: Python 3.12, FastAPI, Pydantic v2, Uvicorn, Scikit-Learn (TF-IDF & Ridge), Pandas, NumPy, SciPy, Joblib, and SQLite3.

Q36: What libraries are used in the frontend?
A36: React 19, Vite 6, Tailwind CSS v4, Recharts (for similarity and Ridge contribution bar charts), and Lucide React (for icons).
"""

CHAPTER_17_TITLE = "Chapter 17: Glossary, Summary Cheat Sheet & Viva Advice"
CHAPTER_17_TEXT = """================================================================================
CHAPTER 17: GLOSSARY, SUMMARY CHEAT SHEET & VIVA ADVICE
================================================================================

17.1 Technical Glossary (25 Key Terms)
--------------------------------------
1. Token: An individual word or character sequence extracted during text processing.
2. N-Gram: A contiguous sequence of n tokens. Unigram = 1 word; Bigram = 2 words.
3. TF (Term Frequency): The frequency of a word within a specific section text.
4. IDF (Inverse Document Frequency): A logarithmic metric penalizing common words across the corpus.
5. TF-IDF: The product of TF and IDF, representing word importance in sparse vector format.
6. L2 Normalization: Scaling a vector so its Euclidean length equals 1.0.
7. Cosine Similarity: The cosine of the angle between two normalized vectors (dot product), ranging [0, 1].
8. Ridge Regression: Linear regression augmented with an L2 penalty on coefficient magnitudes.
9. Regularization (L2): A penalty term (alpha * sum(w^2)) added to the loss function to prevent overfitting.
10. Alpha: The hyperparameter controlling the strength of L2 regularization in Ridge regression.
11. Intercept (beta_0): The baseline predicted value when all input features are zero (-0.0706).
12. Coefficient (w_i): The learned multiplier for each section similarity in the linear model.
13. Suppressor Variable: A feature with negative weight that cancels out irrelevant noise from collinear features.
14. Multicollinearity: High linear correlation among independent feature variables.
15. Project-Disjoint Split: Partitioning data by unique project IDs to guarantee zero train/test leakage.
16. Cross-Boundary Pairs: Pairs where one project is in Train and the other is in Test (must be discarded).
17. MAE (Mean Absolute Error): Average absolute difference between predicted and true labels.
18. RMSE (Root Mean Squared Error): Square root of mean squared error, penalizing large outliers.
19. R2 Score: Proportion of variance in target labels explained by the model.
20. Spearman Rho: Non-parametric rank correlation measuring monotonic ordering.
21. Novelty Percentage: 100 * (1 - predicted_similarity / 4.0).
22. Explainable AI (XAI): Techniques that make machine learning decision logic transparent and inspectable.
23. Additive Reconstruction: Verifying that Intercept + sum(Contributions) == Raw Prediction.
24. Candidate Self-Exclusion: Filtering out the query project itself to prevent trivial 100% self-similarity.
25. Synthetic Ground Truth: Labels generated by heuristic formulas rather than human expert adjudications.

17.2 The Entire ML Pipeline on One Page
---------------------------------------
1. Input: 8 structured text sections of an academic proposal.
2. Vectorization: 8 pre-fitted TfidfVectorizers (ngram_range=(1,2), stop_words='english') transform texts into 8 sparse vectors.
3. Comparison: Dot product against 600 baseline project vectors yields 8 cosine similarity scores per historical project.
4. Linear Fusion: Ridge regression combines similarities:
      y_raw = -0.0706 + (0.134*s_title) + (0.398*s_abstract) - (0.126*s_prob) + (0.237*s_obj)
              + (0.010*s_meth) + (2.205*s_data) - (0.065*s_algo) - (0.146*s_tech)
5. Filtering: Candidate project is self-excluded by ID, title, or vector identity.
6. Ranking: Historical projects are sorted by predicted similarity descending; top 5 are selected.
7. Novelty Formulation: s_max is the highest similarity among remaining projects.
      Novelty (%) = 100 * (1.0 - (s_max / 4.0))
8. XAI Generation: Sections are ranked; shared n-grams and source-verified text snippets are extracted.
9. Database Segregation: Scored proposal is displayed in React; stored in SQLite ONLY if user clicks "Save".

17.3 Most Important Formulas to Memorize
----------------------------------------
1. Novelty Formula:
      Novelty (%) = 100.0 * ( 1.0 - (s_max / 4.0) )

2. Ridge Prediction Formula:
      y_raw = beta_0 + sum_{i=1}^8 ( w_i * s_i )

3. Additive Contribution Formula:
      contribution_i = w_i * s_i
      y_raw == beta_0 + sum_{i=1}^8 contribution_i

4. Cosine Similarity Formula:
      s_i = ( v_cand . v_hist ) / ( ||v_cand|| * ||v_hist|| )

17.4 Most Important Numbers to Memorize
---------------------------------------
- 600: Total unique historical projects in the reference database (200 Mini, 200 Mega, 200 PG).
- 2,996: Total clean pairwise comparisons used in training (4 duplicate pairs removed from 3,000).
- 480 / 120: The 80/20 project-disjoint split (480 training projects = 1,916 pairs; 120 test projects = 120 pairs).
- 1,736: Total learned vocabulary size across all eight TF-IDF vectorizers.
- -0.0706: The model intercept ($\beta_0$).
- +2.2054: The dataset coefficient ($w_{dataset}$), representing over 55% of total predictive weight.
- +2.6476: The sum of all eight learned coefficients.
- 2.5768: The maximum possible raw prediction for a 100% identical duplicate project.
- 35.58%: The minimum possible novelty score for an exact duplicate (meaning < 25% tier is unreachable).
- 0.6511: 5-Fold Cross-Validation Mean RMSE ($\pm 0.0910$).
- 0.7734: Final Holdout RMSE (matches Fold 5 of cross-validation).
- 0: Total number of real faculty annotations in the raw verification workbook.

17.5 Final Viva Advice
----------------------
- Be Proud of What You Built: You built a working, end-to-end, full-stack application with React, Vite, Tailwind, FastAPI, SQLite, and an Explainable AI pipeline with sub-35ms response times.
- Be Scientifically Honest: External examiners love students who understand the limitations of their own models. If asked, "Is this ready for real-world deployment?", answer:
  "No, sir/ma'am. The software architecture is complete and mathematically verified, but the model was trained on synthetic benchmark labels. Before real-world deployment, faculty members must annotate real project pairs in our verification package, and we should integrate dense semantic embeddings to defend against paraphrasing."
  Answering with scientific honesty demonstrates mature engineering and academic integrity!
"""

print("Writing Chapters 16 and 17 to text...")
