"""
Chapters 9 to 12 of the Complete Technical Machine Learning Documentation.
"""

CHAPTER_9_TITLE = "Chapter 9: How the Model Decides Its Weights"
CHAPTER_9_TEXT = """================================================================================
CHAPTER 9: HOW THE MODEL DECIDES ITS WEIGHTS
================================================================================

9.1 Where Do the Weights Come From?
-----------------------------------
A common misconception among students preparing for viva is assuming that the developer chose the section weights by hand (e.g., "I decided that the dataset is worth 50% and the title is worth 10%").

In this system:
  - THE DEVELOPER DID NOT PICK THE SECTION WEIGHTS.
  - The weights were mathematically derived by scikit-learn's Ridge regression algorithm to minimize the squared difference between the model's predictions and the 3,000 ground-truth similarity labels in the training dataset.

9.2 The Mathematical Optimization Process
-----------------------------------------
Ridge regression fits weights $w_1, \dots, w_8$ and intercept $b$ by solving:

           min_{w, b} sum_{j=1}^{1916} ( y_j - ( sum_{k=1}^8 w_k * s_{jk} + b ) )^2 + alpha * sum_{k=1}^8 w_k^2

The optimization considers two competing forces:
  1. Fit to Data: Minimize prediction error against the training labels $y_j \in [0, 4]$.
  2. L2 Shrinkage Penalty: Penalize large weights ($\alpha \sum w_k^2$) to prevent any single section from taking on an infinite or unstable weight due to feature collinearity.

At $\alpha = 1.0$, the closed-form matrix calculus yields the exact weights stored in `models/ridge_fusion_model.joblib`:

+----------------------+--------------------+---------------------------------------------------------------+
| Feature Name         | Learned Weight     | Qualitative Role in Prediction                                |
+----------------------+--------------------+---------------------------------------------------------------+
| Intercept (beta_0)   | -0.07063172        | Baseline bias constant when all section similarities are 0.   |
| dataset_similarity   | +2.20544639        | Dominant positive predictor (>55% of total scale).            |
| abstract_similarity  | +0.39804354        | Strong secondary positive predictor.                          |
| objectives_sim       | +0.23673393        | Moderate positive predictor.                                  |
| title_similarity     | +0.13404783        | Modest positive predictor.                                    |
| methodology_sim      | +0.00989785        | Negligible positive predictor (~0.01).                        |
| algorithm_similarity | -0.06507632        | Negative suppressor variable (-0.065).                        |
| problem_similarity   | -0.12586323        | Negative suppressor variable (-0.126).                        |
| technology_similarity| -0.14564642        | Strongest negative suppressor variable (-0.146).              |
+----------------------+--------------------+---------------------------------------------------------------+
| Sum of Coefficients  | +2.64758357        | Theoretical maximum total feature contribution when s_i = 1.0 |
+----------------------+--------------------+---------------------------------------------------------------+

9.3 Why Is the Dataset Weight So Huge (+2.2054)?
------------------------------------------------
Notice that `dataset_similarity` alone contributes up to $+2.2054$ points out of a 4.0 scale!
Why did the machine learning algorithm assign over 55% of its total predictive capacity to the dataset?
  - Root Cause: Synthetic Dataset Generation Artifact.
  - In `Academic_Project_Similarity_Synthetic_Dataset.xlsx`, whenever pairs were synthesized to have High Similarity (Label 3) or Duplicate Similarity (Label 4), the generator almost always assigned them identical dataset names (e.g., both used "PlantVillage", or both used "BraTS MRI").
  - Conversely, whenever pairs had Label 0 ("Not Similar"), their datasets were completely different.
  - The Ridge regression algorithm discovered that `dataset_similarity` was the single cleanest, highest-signal predictor of the synthetic label in the training data, so it assigned it a dominant weight of $+2.2054$.

9.4 Why Are Some Weights Negative? The Suppressor Effect
--------------------------------------------------------
It seems counter-intuitive that sharing the same algorithm or technology stack would REDUCE predicted project similarity ($w_{tech} = -0.1456$, $w_{prob} = -0.1259$, $w_{algo} = -0.0651$).

Why did the model learn negative weights?
  - In statistics, this is known as a Suppressor Variable Effect caused by Multicollinearity.
  - Standard software tools ("Python", "TensorFlow", "Scikit-Learn", "CUDA") and generic algorithm names ("CNN", "SVM", "Random Forest") appear ubiquitously across hundreds of completely dissimilar projects in computer science.
  - A project on "Stock Price Prediction" and a project on "Skin Cancer Detection" both list "Python, Scikit-Learn, Pandas" in their technology stack and "Machine Learning / CNN" in their algorithm section.
  - If the model had positive weights on technology and algorithms, it would predict an artificially high similarity for completely unrelated projects.
  - The Ridge regression model learned that high technology similarity often acts as "noise" or false overlap. To correct for this and prevent false positives across different domains, the optimization assigned small negative weights to algorithms and technologies!

9.5 The Theoretical Ceiling Vulnerability (The 2.5768 Maximum)
--------------------------------------------------------------
The combination of an intercept of $-0.0706$ and a coefficient sum of $+2.6476$ leads to a severe mathematical limitation:
  - Suppose a candidate proposal is an exact 100% duplicate of a historical project across all 8 sections ($s_1 = s_2 = \dots = s_8 = 1.0$).
  - What does the Ridge model predict?
        y_raw = -0.07063172 + (1.0 * 2.64758357) = 2.57695185 (~ 2.5768)
  - What novelty percentage does this produce?
        Novelty (%) = 100 * (1.0 - (2.5768 / 4.0)) = 100 * (1.0 - 0.6442) = 35.58%
  - What tier does 35.58% map to?
        35.58% in [25.0%, 50.0%) -> "Low Novelty (Significant Prior Work)"
  - CRITICAL SCIENTIFIC AUDIT FINDING:
    Even for an exact, 100% identical twin of an existing project, the model prediction can never exceed 2.5768. Therefore, the estimated novelty can never drop below 35.58%. The system can NEVER reach the "Very Low Novelty / Potential Duplicate" tier (< 25.0%) under its current weights!

9.6 Step-by-Step Numerical Example of Feature Contributions
-----------------------------------------------------------
Suppose a candidate proposal is evaluated against historical project SYN-0142:
  - `title_similarity` = 0.5000       -> Contrib: 0.5000 * +0.1340 = +0.0670
  - `abstract_similarity` = 0.6000    -> Contrib: 0.6000 * +0.3980 = +0.2388
  - `problem_similarity` = 0.4000     -> Contrib: 0.4000 * -0.1259 = -0.0504
  - `objectives_sim` = 0.7000         -> Contrib: 0.7000 * +0.2367 = +0.1657
  - `methodology_sim` = 0.3000        -> Contrib: 0.3000 * +0.0099 = +0.0030
  - `dataset_similarity` = 1.0000     -> Contrib: 1.0000 * +2.2054 = +2.2054
  - `algorithm_similarity` = 0.8000   -> Contrib: 0.8000 * -0.0651 = -0.0521
  - `technology_similarity` = 0.9000  -> Contrib: 0.9000 * -0.1456 = -0.1310

Additive Reconstruction:
  Sum of Contributions = +0.0670 + 0.2388 - 0.0504 + 0.1657 + 0.0030 + 2.2054 - 0.0521 - 0.1310
                       = +2.4464
  Intercept            = -0.0706
  Raw Predicted Score  = -0.0706 + 2.4464 = 2.3758
  Clipped Score        = 2.3758
  Novelty Score (%)    = 100 * (1.0 - (2.3758 / 4.0)) = 100 * (1.0 - 0.59395) = 40.61% ("Low Novelty")
"""

CHAPTER_10_TITLE = "Chapter 10: What Happens When a User Submits a Proposal"
CHAPTER_10_TEXT = """================================================================================
CHAPTER 10: WHAT HAPPENS WHEN A USER SUBMITS A PROPOSAL
================================================================================

10.1 End-to-End Runtime Execution Trace
---------------------------------------
When a student or faculty member uses the React dashboard, the entire request-response cycle executes deterministically through 12 verifiable steps:

Step 1 (Form Input):
  The user enters the 8 sections in `frontend/src/components/proposal/ProposalForm.jsx`.
  Field validation ensures non-empty strings and minimum character lengths.

Step 2 (API POST Dispatch):
  The user clicks "Evaluate Novelty". `frontend/src/services/api.js` dispatches an HTTP POST request to:
  `http://127.0.0.1:8000/api/score`
  Payload contains JSON: `{ "project_title": "...", "abstract": "...", ... }`.

Step 3 (FastAPI Route Handling):
  FastAPI route handler `@app.post("/api/score")` in `src/api/main.py` receives the request.
  Pydantic model `ProposalInput` validates field non-emptiness and strip whitespace.

Step 4 (Service Delegation):
  The route calls `NoveltyService.score_proposal(candidate_dict, top_k=5)` in `src/api/service.py`.

Step 5 (Vectorization of Candidate Proposal):
  The candidate proposal is passed to `NoveltyExplainer.explain_proposal()` in `src/explainability/explainer.py`.
  The 8 section texts are transformed by the 8 pre-loaded `TfidfVectorizer` instances:
  `v_cand_k = vectorizers[k].transform([candidate_text_k])`

Step 6 (Multi-Project Cosine Matrix Multiplication):
  The candidate vector for each section is matrix-multiplied against the precomputed sparse TF-IDF matrix of all 600 reference projects:
  `sim_k = cosine_similarity(v_cand_k, tfidf_matrices[k])` -> 600 similarity scores for each section.

Step 7 (Batch Ridge Model Inference):
  The resulting $600 \times 8$ feature matrix $X$ is multiplied by the Ridge coefficient vector:
  `raw_preds = -0.07063 + (X @ coef_)`
  `clipped_preds = np.clip(raw_preds, 0.0, 4.0)`

Step 8 (Candidate Self-Exclusion Filtering):
  The candidate proposal is checked against the 600 projects to prevent self-matching:
  1. Project ID match: If candidate project_id matches historical project_id.
  2. Exact title match: If candidate title matches historical title (case-insensitive).
  3. Identical vector match: If all 8 section similarities are >= 0.9999.
  Matching records are strictly masked out of the eligible comparison set.

Step 9 (Deterministic Ranking & Top Match Selection):
  The remaining eligible projects are sorted:
  `sort_values(by=["predicted_similarity", "raw_prediction", "project_id"], ascending=[False, False, True])`
  The top 5 closest historical projects are selected.

Step 10 (Novelty Formulation):
  The maximum predicted similarity across the corpus is selected:
  `max_sim = ranked_df.iloc[0]["predicted_similarity"]`
  `novelty_pct = 100.0 * (1.0 - (max_sim / 4.0))`
  `novelty_tier = get_novelty_level(novelty_pct)`

Step 11 (XAI Evidence Generation):
  For each of the top 5 matches, `explain_pair()` extracts:
  - Section similarities and Ridge signed contributions.
  - Shared informative keywords using the fitted vocabulary.
  - Word-boundary-snapped source text snippets from candidate and historical records.
  - Section difference summary (high overlap >= 0.30 vs. differentiating < 0.15).

Step 12 (JSON Response & UI Presentation):
  The API returns `ProposalEvaluationResponse`.
  React renders the Novelty Gauge, Recharts bar charts, and expandable match cards.
  DATABASE NOTE: At this stage, NO record is written to SQLite.

10.2 Sequence Diagram of Runtime Execution
------------------------------------------
+----------+          +---------------+          +----------------+          +------------------+          +------------+
| Frontend |          | FastAPI /api  |          | NoveltyService |          | NoveltyExplainer |          | SQLite DB  |
+----+-----+          +-------+-------+          +-------+--------+          +--------+---------+          +-----+------+
     |                        |                          |                            |                          |
     | 1. POST /api/score     |                          |                            |                          |
     |----------------------->|                          |                            |                          |
     |                        | 2. Validate Proposal     |                            |                          |
     |                        |------------------------->|                            |                          |
     |                        |                          | 3. explain_proposal()      |                          |
     |                        |                          |--------------------------->|                          |
     |                        |                          |                            | 4. TF-IDF vectorization  |
     |                        |                          |                            |    & cosine sim (600 prj)|
     |                        |                          |                            | 5. Ridge inference       |
     |                        |                          |                            | 6. Self-exclusion filter |
     |                        |                          |                            | 7. Top-5 XAI evidence    |
     |                        |                          | 8. Return Report Payload   |<-------------------------|
     |                        |                          |<---------------------------|                          |
     |                        | 9. HTTP 200 JSON         |                            |                          |
     |<-----------------------|--------------------------|                            |                          |
     |                        |                          |                            |                          |
     | [User clicks "Save"]   |                          |                            |                          |
     | 10. POST /api/save     |                          |                            |                          |
     |----------------------->|                          |                            |                          |
     |                        | 11. Store in SQLite      |                            |                          |
     |                        |--------------------------------------------------------------------------------->|
     |                        | 12. HTTP 201 Saved       |                            |                          |
     |<-----------------------|----------------------------------------------------------------------------------|
"""

CHAPTER_11_TITLE = "Chapter 11: Explainable AI (XAI) Architecture"
CHAPTER_11_TEXT = """================================================================================
CHAPTER 11: EXPLAINABLE AI (XAI) ARCHITECTURE
================================================================================

11.1 The Four Layers of Explainability
--------------------------------------
The explainability engine (`src/explainability/explainer.py`) implements four complementary layers of transparent evidence:
  - Layer 1 (Section-Level Rankings): Ranks the 8 sections by raw cosine similarity to show exactly where the overlap is concentrated.
  - Layer 2 (Additive Model Attributions): Exposes the exact mathematical Ridge signed contribution for each section, reconstructing the prediction.
  - Layer 3 (Shared Informative Keywords): Identifies the exact technical terms and n-grams shared between both proposals, filtered for generic stop words.
  - Layer 4 (Source-Verified Text Snippets): Extracts verbatim text windows centered around shared keywords with clean word boundaries.

11.2 Layer 1: Ranked Sections Table
------------------------------------
For each match, sections are sorted in descending order of cosine similarity:
  `ranked_sections = sorted(sections, key=lambda x: x["raw_similarity"], reverse=True)`
Each entry displays:
  - Section Name (e.g., "Dataset", "Abstract", "Algorithm")
  - Raw Cosine Similarity ($s_k$)
  - Model Weight ($w_k$)
  - Signed Contribution ($w_k \times s_k$)

11.3 Layer 2: Additive Ridge Feature Attributions
-------------------------------------------------
Because Ridge regression is an additive linear model:
              Predicted Score = Intercept + sum( Signed Contributions )
In the dashboard and API report, every prediction is fully decomposable. A faculty evaluator can verify with a pocket calculator that:
              -0.0706 + sum(contributions) == Predicted Similarity

11.4 Layer 3: Informative Keyword Extraction (`extract_shared_terms`)
---------------------------------------------------------------------
How does the system identify shared keywords between two sections?
  1. Both texts are transformed by the fitted section vectorizer:
     `v_cand = vec.transform([cand_text])`
     `v_hist = vec.transform([hist_text])`
  2. The element-wise Hadamard product of the two normalized vectors is computed:
     `overlap = v_cand.multiply(v_hist)`
     In cosine similarity, the total score is the sum of these element-wise products: $\sum_k v_1[k] \cdot v_2[k]$.
     Each non-zero entry in `overlap` represents an exact n-gram that contributes to the cosine similarity!
  3. Stop-Word and Low-Information Filtering (`is_informative_term`):
     Common generic academic boilerplate terms are filtered out:
     `GENERIC_LOW_INFO_TERMS = {"using", "system", "project", "approach", "method", "results", ...}`
  4. Physical Verification Check:
     To ensure integrity, the code executes a strict physical regex check:
     `rf"(?<!\w){re.escape(term)}(?!\w)"`
     The term MUST physically appear as an exact whole word in BOTH source texts!

11.5 Layer 4: Source-Verified Snippet Extraction (`extract_snippet`)
--------------------------------------------------------------------
To display context around shared keywords without ugly mid-word chops:
  1. Word-Boundary Snapping: When cutting a window of 45 characters before and after the keyword, the algorithm snaps outward to the nearest whitespace.
  2. Ellipsis Integrity: Ellipses (`...`) are added ONLY if characters were actually omitted before the start or after the end.
  3. Verification Check (`verify_snippet`):
     `return (body in source_text) and (term.lower() in body.lower())`
     If the extracted snippet does not exist as an exact substring of the original project text, it is rejected.

11.6 Candidate Self-Exclusion Algorithm
---------------------------------------
If a student re-submits a proposal that is already indexed in the database (or evaluates an existing project), the system must not compare the project against itself (which would trivially produce similarity 1.0 and 0% novelty).

In `NoveltyExplainer.explain_proposal()`, self-exclusion executes three checks:
  1. Project ID Match: If `candidate_id` matches `eligible_df["project_id"]` (case-insensitive).
  2. Exact Title Match: If candidate title matches `historical_titles` (case-insensitive).
  3. Identical Vector Match: If all 8 section similarities are >= 0.9999.
Matching rows are removed from comparison, ensuring the candidate is only evaluated against *prior peer work*.
"""

CHAPTER_12_TITLE = "Chapter 12: Comprehensive Analysis of Every Notebook & Script"
CHAPTER_12_TEXT = """================================================================================
CHAPTER 12: COMPREHENSIVE ANALYSIS OF EVERY NOTEBOOK & SCRIPT
================================================================================

12.1 Detailed Audit of Jupyter Notebooks (`notebooks/`)
-------------------------------------------------------

NOTEBOOK 01: `notebooks/01_data_exploration.ipynb`
  - Purpose: Exploratory data analysis (EDA) of the raw Excel dataset.
  - Input Files: `data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx`, `data/raw/Project_Similarity_Pairs_0_to_4.xlsx`.
  - Output Files: In-memory summary tables and console distributions.
  - Main Libraries: `pandas`, `pathlib`.
  - Code Summary:
      * Inspects sheets: `Projects`, `SimilarityPairs`, `DataDictionary`, `README`.
      * Verifies shapes: 600 projects, 3,000 pairs.
      * Audits label frequencies: Label 0 (44.5%), Label 1 (11.1%), Label 2 (17.1%), Label 3 (19.4%), Label 4 (7.8%).
      * Confirms balanced program levels: exactly 200 UG Mini, 200 UG Mega, 200 PG Dissertations.
  - Beginner Takeaway: Understands the raw dataset schema, column types, and the synthetic nature of the labels.

NOTEBOOK 02: `notebooks/02_tfidf_baseline.ipynb`
  - Purpose: Evaluates a global concatenated text TF-IDF baseline model.
  - Input Files: `Academic_Project_Similarity_Synthetic_Dataset.xlsx`.
  - Output Files: `results/tfidf_baseline_correlations.csv`.
  - Main Libraries: `pandas`, `sklearn.feature_extraction.text.TfidfVectorizer`, `scipy.stats.spearmanr`.
  - Code Summary:
      * Concatenates all 8 text fields into a single `combined_text` string per project.
      * Fits a global `TfidfVectorizer(stop_words='english', ngram_range=(1,2))`.
      * Computes pairwise cosine similarity on `combined_text`.
      * Measures correlation against `expert_similarity_label`: Pearson r ~ 0.62, Spearman rho ~ 0.58.
  - Beginner Takeaway: Concatenating all text into one blob loses section structure and yields suboptimal correlation.

NOTEBOOK 03: `notebooks/03_section_wise_tfidf.ipynb`
  - Purpose: Implements section-disaggregated TF-IDF feature extraction across the 8 sections.
  - Input Files: `Academic_Project_Similarity_Synthetic_Dataset.xlsx`.
  - Output Files: `results/section_wise_tfidf_features.csv`.
  - Main Libraries: `pandas`, `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.metrics.pairwise.cosine_similarity`.
  - Code Summary:
      * Defines section mapping dictionary for 8 fields.
      * Fits 8 separate `TfidfVectorizer` instances on individual section columns.
      * Computes 8 separate pairwise cosine similarities for all 3,000 pairs.
  - Beginner Takeaway: Disaggregating text into 8 distinct section similarities allows individual section importance weighting.

NOTEBOOK 04: `notebooks/04_feature_label_analysis.ipynb`
  - Purpose: Analyzes individual section correlations and multicollinearity with ground-truth similarity labels.
  - Input Files: `section_wise_tfidf_features.csv`, `Project_Similarity_Pairs_0_to_4.xlsx`.
  - Output Files: Correlation heatmaps, feature pair scatter plots.
  - Main Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`.
  - Code Summary:
      * Computes correlation matrix between the 8 section similarities and the target label.
      * Discovers that `dataset_similarity` has the highest correlation with similarity labels (r ~ 0.81).
      * Discovers that `problem` and `technology` have high collinearity with `methodology`.
  - Beginner Takeaway: Highlights that dataset overlap dominates synthetic labels and identifies collinear features.

NOTEBOOK 05: `notebooks/05_weighted_section_fusion.ipynb`
  - Purpose: Experiments with linear section fusion: unweighted average, constrained non-negative optimization, and Ridge regression.
  - Input Files: `section_wise_tfidf_features.csv`.
  - Output Files: Learned section weight vectors and residual plots.
  - Main Libraries: `scipy.optimize.minimize`, `sklearn.linear_model.Ridge`, `sklearn.metrics`.
  - Code Summary:
      * Evaluates unweighted average: RMSE ~ 0.88.
      * Optimizes constrained non-negative weights: RMSE ~ 0.72.
      * Fits unconstrained Ridge regression: RMSE ~ 0.65.
  - Beginner Takeaway: Shows that Ridge regression achieves the lowest error on the synthetic benchmark.

NOTEBOOK 06: `notebooks/06_dense_semantic_embeddings.ipynb`
  - Purpose: Explores dense transformer semantic embeddings (`all-MiniLM-L6-v2`) and hybrid TF-IDF + Dense fusion.
  - Input Files: `Academic_Project_Similarity_Synthetic_Dataset.xlsx`.
  - Output Files: `models/section_dense_extractor.joblib`, `results/hybrid_model_comparison.csv`.
  - Main Libraries: `sentence_transformers`, `torch`, `sklearn.linear_model.Ridge`.
  - Code Summary:
      * Encodes each of the 8 sections into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
      * Computes 8 dense cosine similarities, forming a 16-feature hybrid matrix (8 TF-IDF + 8 Dense).
      * Trains Ridge regression on 16 features: RMSE ~ 0.61.
  - Beginner Takeaway: Dense embeddings capture conceptual similarity but are computationally heavier and less transparent than pure TF-IDF.

12.2 Detailed Audit of Training & Evaluation Scripts (`scripts/`)
----------------------------------------------------------------

SCRIPT 1: `scripts/clean_dataset.py`
  - Purpose: Audits raw pairs, identifies duplicates, and outputs `data/processed/similarity_pairs_clean.csv`.
  - Inputs: `data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx`.
  - Outputs: `data/processed/similarity_pairs_clean.csv`, `dataset_cleaning_manifest.json`, `dataset_cleaning_report.md`.
  - Key Logic: Detects 1 exact duplicate and 3 reversed duplicate pairs using canonical min/max ID tuples.

SCRIPT 2: `scripts/optimize_models.py`
  - Purpose: Executes 5-fold project-disjoint cross-validation and hyperparameter tuning for Ridge regression and Gradient Boosting.
  - Inputs: `Academic_Project_Similarity_Synthetic_Dataset.xlsx`, `similarity_pairs_clean.csv`.
  - Outputs: `results/model_optimization/fold_results.csv`, `summary.csv`, `best_model_config.json`.
  - Key Logic: Disjoint KFold splitting on project IDs; fits TF-IDF vectorizers strictly on training projects per fold.

SCRIPT 3: `scripts/evaluate_holdout.py`
  - Purpose: Evaluates the selected Ridge model on the test partition.
  - Inputs: `Academic_Project_Similarity_Synthetic_Dataset.xlsx`, `similarity_pairs_clean.csv`.
  - Outputs: `results/final_evaluation/holdout_metrics.json`, `holdout_predictions.csv`, `final_evaluation_report.md`.
  - Key Logic: Evaluates on 120 test pairs using 480 training projects. Note: Test set matches Fold 5 of cross-validation.

SCRIPT 4: `scripts/train.py`
  - Purpose: Production pipeline training script that exports fitted checkpoints to `models/`.
  - Inputs: `Academic_Project_Similarity_Synthetic_Dataset.xlsx`, `similarity_pairs_clean.csv`.
  - Outputs: `models/section_tfidf_extractor.joblib`, `models/ridge_fusion_model.joblib`.
  - Key Logic: Fits 8 vectorizers on all 600 projects and trains Ridge regression model on training pairs.

SCRIPT 5: `scripts/score_novelty.py`
  - Purpose: Standalone CLI tool to evaluate candidate project proposals from terminal or JSON file.
  - Inputs: JSON file or CLI flags (`--title`, `--abstract`, etc.), `models/*.joblib`.
  - Outputs: Formatted terminal novelty report or structured JSON.
"""

print("Writing Chapters 9 to 12 to text...")
