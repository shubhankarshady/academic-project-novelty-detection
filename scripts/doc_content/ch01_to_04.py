"""
Chapters 1 to 4 of the Complete Technical Machine Learning Documentation.
"""

CHAPTER_1_TITLE = "Chapter 1: Project Overview & Core Fundamentals"
CHAPTER_1_TEXT = """================================================================================
CHAPTER 1: PROJECT OVERVIEW & CORE FUNDAMENTALS
================================================================================

1.1 What the Academic Project Novelty Detection System Does
------------------------------------------------------------
The Academic Project Novelty Detection System is an end-to-end, explainable machine learning software application designed to assist academic faculty committees, project coordinators, and student researchers in evaluating the novelty of proposed undergraduate (UG) and postgraduate (PG) technical capstone proposals.

When a student submits a new project proposal, the system decomposes the proposal into eight structured academic text sections, compares these sections against a baseline reference corpus of historical student projects, calculates section-by-section lexical similarity scores, feeds these scores through a calibrated linear model, and outputs:
  1. An estimated overall novelty percentage (0.0% to 100.0%).
  2. A categorized risk/novelty assessment tier (e.g., "High", "Moderate", "Low", "Very Low").
  3. The closest historical projects in the database, ranked by multi-section similarity.
  4. Transparent Explainable AI (XAI) evidence, including section-by-section similarity scores, exact source-verified text snippets, shared technical keywords, and linear model feature contributions.

1.2 What Problem It Attempts to Solve
--------------------------------------
In engineering and computer science colleges, a pervasive academic challenge is project duplication. Every academic cycle, hundreds of students submit capstone project synopses that repeat well-worn formulas with minor cosmetic variations—for example:
  - "Student Attendance System using Face Recognition and OpenCV"
  - "Heart Disease Prediction using Machine Learning and Random Forest"
  - "Fake News Detection using Natural Language Processing and Naive Bayes"
  - "Crop Disease Detection using Convolutional Neural Networks and PlantVillage"

College project review panels typically face three severe operational constraints:
  - Memory Decay & Information Siloing: Faculty members cannot remember the details of hundreds of projects submitted across past academic cohorts, different degree programs (B.Tech, MCA, M.Tech), or sister departments.
  - Manual Review Fatigue: Evaluating hundreds of multi-page technical documents by hand leads to subjective, inconsistent grading and oversight of duplicate prior work.
  - The "Black-Box" AI Dilemma: Commercial plagiarism tools (such as Turnitin) only perform verbatim string-matching across global web pages and reports. They do not understand the structured architecture of an engineering project proposal (Problem vs. Methodology vs. Dataset vs. Algorithm), nor do they provide actionable diagnostic guidance explaining *why* an idea lacks technical novelty.

1.3 Why Comparing an Academic Proposal Against Prior Work Is Useful
--------------------------------------------------------------------
Comparing a candidate proposal against an institutional project database yields distinct pedagogical and academic advantages:
  - Diagnostic Guidance for Students: Rather than receiving a blunt rejection, students receive an itemized breakdown showing exactly which sections overlap with prior projects (e.g., "Your proposed CNN model and PlantVillage dataset overlap heavily with Project SYN-0157, but your Problem Statement introduces a novel field constraint").
  - Encouraging Real Technical Innovation: Students are incentivized to differentiate their methodology, formulate novel problem constraints, or evaluate alternative datasets before committing months to development.
  - Archival Consistency: Academic institutions maintain an indexed, searchable historical repository of institutional project outputs.

1.4 What the System Receives as Input
-------------------------------------
The system strictly accepts eight structured text sections representing an academic proposal:
  1. `project_title`: The formal title of the capstone project.
  2. `abstract`: A concise high-level overview of the work and main contribution.
  3. `problem_statement`: The technical challenge, research gap, or domain limitation addressed.
  4. `objectives`: Itemized, measurable technical aims and deliverables.
  5. `proposed_methodology`: The end-to-end technical approach, pipeline architecture, and implementation workflow.
  6. `dataset`: The specific data sources, public benchmarks, hardware sensors, or collection protocols.
  7. `algorithm_model`: The specific machine learning architectures, statistical formulas, or mathematical models used.
  8. `technology_tools`: The programming languages, software libraries, accelerators, and hardware environments.
  (Optional metadata: `domain` and `project_id`).

1.5 What the System Produces as Output
--------------------------------------
Upon submission, the scoring pipeline returns a structured JSON payload rendered into an interactive dashboard:
  - `overall_novelty_score_pct`: A single floating-point percentage in [0.0, 100.0]%.
  - `novelty_assessment_tier`: One of four standardized qualitative categories.
  - `max_similarity_score`: The maximum multi-section similarity score (on a 0.0 to 4.0 regression scale) found across all reference corpus projects.
  - `corpus_size_evaluated`: The exact number of historical reference projects evaluated (600 baseline projects).
  - `top_matches`: An array of the top 5 closest historical projects, containing:
      * Historical project ID, title, and domain.
      * Predicted similarity and raw similarity prediction.
      * Section-level cosine similarity scores across all eight sections.
      * Ridge model signed feature contributions reconstructing the exact predicted score.
      * Ranked sections table ordering sections from highest overlap to lowest.
      * High-overlap vs. differentiating section summary.
      * Source-verified evidence snippets with shared informative keywords.
  - `framework_disclaimers`: Prominent scientific notices regarding lexical limitations and corpus relativity.

1.6 What "Novelty" Means in This Implementation
------------------------------------------------
In this system, novelty is NOT an absolute metaphysical or philosophical judgment of human creativity. It is defined strictly as a mathematical inversion of maximum predicted lexical and feature similarity relative to a closed reference corpus:

              Novelty (%) = 100.0 * (1.0 - (max_predicted_similarity / 4.0))

In simple terms:
  - If a candidate proposal is completely lexically and structurally disjoint from every project in the database (similarity = 0.0), its estimated novelty is 100.0%.
  - If a candidate proposal shares identical terminology, dataset names, and methodology with a past project such that predicted similarity is high (e.g., 2.65 out of 4.0), its estimated novelty drops (e.g., to ~33.8%).

1.7 What the System Can and Cannot Conclude
-------------------------------------------
WHAT THE SYSTEM CAN CONCLUDE:
  - "The submitted proposal shares extensive lexical n-grams, technical terms, and identical dataset/algorithm references with Historical Project SYN-0157."
  - "The Methodology and Dataset sections exhibit high similarity (>= 0.30 cosine similarity) against past departmental projects."
  - "The candidate proposal's Problem Statement uses terminology unseen in the 600 baseline projects."

WHAT THE SYSTEM CANNOT CONCLUDE:
  - Plagiarism: The system cannot prove academic dishonesty or illicit copying. Shared terminology often occurs naturally because two projects use the same standard textbook algorithms or public datasets.
  - Absolute Scientific Originality: The system cannot declare an idea globally novel. It only searches the indexed 600 projects in its local SQLite/Joblib reference corpus, not the entire global corpus of IEEE, ACM, or arXiv literature.
  - Feasibility or Technical Merit: A proposal can receive a 100% novelty score simply by using nonsensical or exotic words that have never appeared in the database. The system does not verify whether the proposed math works or whether the student can actually build it.

1.8 Distinction: Similarity vs. Novelty vs. Plagiarism vs. Proof of Originality
-------------------------------------------------------------------------------
Understanding these four concepts is essential for viva defense and project reports:

+----------------------+------------------------------------------------------------------------+
| Concept              | Definition in this System                                              |
+----------------------+------------------------------------------------------------------------+
| Project Similarity   | The degree of structural and lexical overlap between two technical     |
|                      | project records across 8 standardized sections, predicted on a [0, 4] |
|                      | scale using TF-IDF cosine similarities and a Ridge regression model.   |
+----------------------+------------------------------------------------------------------------+
| Novelty Estimation   | The mathematical complement of maximum similarity relative to the      |
|                      | reference database: Novelty % = 100 * (1 - Sim / 4.0). Higher novelty  |
|                      | means lower observed overlap against known prior projects.             |
+----------------------+------------------------------------------------------------------------+
| Plagiarism Detection | An investigative finding of intellectual property theft or dishonest   |
|                      | uncredited copying. Requires human intent adjudication and full text   |
|                      | alignment. THIS SYSTEM DOES NOT PROVE PLAGIARISM.                      |
+----------------------+------------------------------------------------------------------------+
| Proof of Originality | A global legal or scientific validation (such as a patent grant or peer|
|                      | reviewed journal acceptance) verifying that no human on earth has ever |
|                      | developed the idea. THIS SYSTEM CANNOT PROVE ORIGINALITY.              |
+----------------------+------------------------------------------------------------------------+

1.9 A Simple Explanatory Example
--------------------------------
Consider a student proposal titled: "Deep Learning for Skin Lesion Classification".
  - Section `dataset`: "ISIC 2024 Challenge Archive"
  - Section `algorithm_model`: "ResNet-50 Convolutional Neural Network"
  - Section `technology_tools`: "PyTorch, CUDA, Python"

If the reference corpus already contains Project SYN-0089: "Automated Melanoma Screening Using ResNet-50 on ISIC Data":
  1. The system computes a 1.0 cosine similarity on the `dataset` section ("ISIC").
  2. It computes a 0.85 cosine similarity on the `algorithm_model` section ("ResNet-50 CNN").
  3. The Ridge model combines these similarities into a predicted similarity of ~2.45 (out of 4.0).
  4. The estimated novelty is calculated as: 100 * (1 - 2.45 / 4.0) = 38.75% ("Low Novelty").
  5. The XAI engine generates an explanation showing the student that their dataset and algorithm match SYN-0089, but highlights their unique Problem Statement regarding low-resource smartphone lenses as their differentiating factor.
"""

CHAPTER_2_TITLE = "Chapter 2: Complete Repository & File Structure Map"
CHAPTER_2_TEXT = """================================================================================
CHAPTER 2: COMPLETE REPOSITORY & FILE STRUCTURE MAP
================================================================================

2.1 Complete Repository Tree
-----------------------------
The repository is organized into distinct, modular subsystems: Data Storage, Machine Learning Core, Explainable AI Engine, FastAPI Backend, React Frontend, Automation Scripts, and Evaluation Results:

academic-project-novelty-detection/
|-- data/
|   |-- raw/
|   |   |-- Academic_Project_Similarity_Synthetic_Dataset.xlsx  # Raw master Excel dataset
|   |   |-- Project_Similarity_Pairs_0_to_4.xlsx               # Raw 3,000 similarity pair records
|   |   `-- Faculty_Verification_DomainWise.xlsx               # 13-sheet faculty verification template
|   |-- processed/
|   |   |-- similarity_pairs_clean.csv                         # Cleaned 2,996 deduplicated pair rows
|   |   |-- dataset_cleaning_manifest.json                     # Metadata and audit trail of cleaning
|   |   `-- dataset_cleaning_report.md                         # Detailed markdown report of deduplication
|   |-- faculty_batches/                                       # Domain-split batches for faculty annotation
|   `-- proposals.db                                           # SQLite database for submitted user proposals
|-- models/
|   |-- section_tfidf_extractor.joblib                         # Fitted SectionTFIDFExtractor (600 projects)
|   |-- ridge_fusion_model.joblib                              # Fitted Ridge regression model (alpha=1.0)
|   `-- section_dense_extractor.joblib                         # Precomputed dense transformer embeddings
|-- notebooks/                                                 # Research and prototyping notebooks
|   |-- 01_data_exploration.ipynb                              # Exploratory data analysis (EDA)
|   |-- 02_tfidf_baseline.ipynb                                # Global concatenated text TF-IDF baseline
|   |-- 03_section_wise_tfidf.ipynb                            # Section-by-section TF-IDF vectorization
|   |-- 04_feature_label_analysis.ipynb                        # Section correlation & feature-label analysis
|   |-- 05_weighted_section_fusion.ipynb                       # Section weight optimization & Ridge training
|   `-- 06_dense_semantic_embeddings.ipynb                     # Sentence-Transformers & hybrid fusion
|-- scripts/                                                   # Command-line tools and pipeline scripts
|   |-- clean_dataset.py                                       # Audit and deduplicate raw pair records
|   |-- train.py                                               # End-to-end model training & artifact export
|   |-- optimize_models.py                                     # 5-fold cross-validation & hyperparameter tuning
|   |-- evaluate_holdout.py                                    # Final holdout partition evaluation
|   |-- score_novelty.py                                       # Standalone CLI proposal scoring tool
|   |-- compare_three_models.py                                # Benchmark TF-IDF vs Dense vs Hybrid
|   |-- analyze_errors.py                                      # Detailed residual and error analysis
|   |-- generate_faculty_verification.py                       # Build domain-wise Excel verification files
|   |-- split_faculty_batches.py                               # Partition pairs into faculty review batches
|   `-- make_docs.py                                           # Automated technical documentation builder
|-- src/                                                       # Production Python package
|   |-- data/
|   |   |-- __init__.py
|   |   `-- loader.py                                          # Clean data loading functions
|   |-- features/
|   |   |-- __init__.py
|   |   |-- section_tfidf.py                                   # SectionTFIDFExtractor class (8 vectorizers)
|   |   `-- dense_embeddings.py                                # Dense Sentence-Transformers extractor
|   |-- models/
|   |   |-- __init__.py
|   |   `-- hybrid_fusion.py                                   # HybridSectionFusionModel class
|   |-- explainability/
|   |   |-- __init__.py
|   |   `-- explainer.py                                       # NoveltyExplainer & XAI evidence generator
|   |-- storage/
|   |   |-- __init__.py
|   |   `-- db.py                                              # DatabaseManager SQLite CRUD and deduplication
|   |-- api/
|   |   |-- __init__.py
|   |   |-- schemas.py                                         # Pydantic v2 schemas for API contracts
|   |   |-- service.py                                         # Singleton NoveltyService loading models
|   |   `-- main.py                                            # FastAPI application and route endpoints
|   `-- utils/
|       |-- __init__.py
|       `-- metrics.py                                         # Evaluation metrics, disjoint splits & formulas
|-- tests/                                                     # Comprehensive unit and integration test suite
|   |-- test_api.py                                            # FastAPI endpoint and schema tests
|   |-- test_storage.py                                        # SQLite CRUD and duplicate conflict tests
|   |-- test_xai.py                                            # Explainability and evidence extraction tests
|   `-- test_scientific_reliability.py                         # Scientific audit & controlled test case suite
|-- results/                                                   # Evaluation reports, manifests, and audits
|   |-- model_optimization/                                    # 5-fold CV summaries and fold metrics
|   |-- final_evaluation/                                      # Final holdout metrics and split manifest
|   `-- validation/
|       `-- scientific_reliability_audit.md                    # Formal scientific reliability audit report
|-- frontend/                                                  # React 19 + Vite 6 + Tailwind CSS application
|   |-- src/
|   |   |-- components/                                        # Modular UI, layout, and report components
|   |   |-- pages/                                             # DashboardPage, EvaluatePage, HistoryPage
|   |   |-- services/api.js                                    # Centralized API fetch client
|   |   `-- App.jsx                                            # Root application coordinator
|   |-- package.json                                           # Frontend npm dependencies
|   `-- vite.config.js                                         # Vite dev server and API proxy configuration
`-- docs/                                                      # Formal documentation
    |-- ML_MODEL_COMPLETE_GUIDE.txt                            # Complete technical guide (Plain Text)
    `-- ML_MODEL_COMPLETE_GUIDE.pdf                            # Complete technical guide (PDF)

2.2 Purpose of Every Machine Learning File
-------------------------------------------
1. `src/features/section_tfidf.py`:
   - Contains `SectionTFIDFExtractor`.
   - Manages eight separate `TfidfVectorizer` instances (one per section).
   - Extracts section-level cosine similarity matrices for projects and candidate proposals.
2. `src/models/hybrid_fusion.py`:
   - Contains `HybridSectionFusionModel`.
   - Combines 8 lexical TF-IDF section features and 8 dense semantic embedding features into a 16-dimensional regression vector.
3. `src/explainability/explainer.py`:
   - Contains `NoveltyExplainer`.
   - Coordinates multi-section comparison, ranking, Ridge additive contribution reconstruction, keyword extraction, and source snippet verification.
4. `src/utils/metrics.py`:
   - Defines regression metrics (RMSE, MAE, R2, Pearson r, Spearman rho).
   - Implements `project_disjoint_split` and `project_disjoint_cv` to enforce zero data leakage.
   - Defines `calculate_novelty_score` and `get_novelty_level`.
5. `src/api/service.py`:
   - Implements `NoveltyService`, a memory-efficient singleton that loads the fitted model artifacts once at server startup and exposes the `score_proposal` method.
6. `scripts/clean_dataset.py`:
   - Audits the raw pairwise dataset, detects exact and reversed duplicate pairs, checks label consistency, and produces `similarity_pairs_clean.csv`.
7. `scripts/optimize_models.py`:
   - Executes 5-fold project-disjoint cross-validation across Ridge regression alphas and gradient boosting to determine the optimal model configuration.
8. `scripts/evaluate_holdout.py`:
   - Evaluates the selected Ridge model on the test partition and records predictions and metrics.
9. `scripts/score_novelty.py`:
   - Command-line interface allowing users and automated testing scripts to evaluate a proposal from a JSON file or terminal flags.

2.3 Exact System Execution Lifecycle
------------------------------------
The system operates across three distinct lifecycles:

PHASE 1: OFFLINE DATA PREPARATION & TRAINING LIFECYCLE
  Step 1: Raw Excel dataset (`Academic_Project_Similarity_Synthetic_Dataset.xlsx`) is loaded by `clean_dataset.py`.
  Step 2: Four redundant pairs are removed; 2,996 clean pairs are saved to `similarity_pairs_clean.csv`.
  Step 3: `optimize_models.py` executes 5-fold project-disjoint cross-validation.
  Step 4: The 600 projects are fitted into eight `TfidfVectorizer` instances using `SectionTFIDFExtractor`.
  Step 5: The Ridge regression model is fitted on training pair similarities and serialized to `models/ridge_fusion_model.joblib`.
  Step 6: The fitted `SectionTFIDFExtractor` is serialized to `models/section_tfidf_extractor.joblib`.

PHASE 2: SERVER STARTUP & MODEL LOADING LIFECYCLE
  Step 1: FastAPI server starts (`uvicorn src.api.main:app`).
  Step 2: `get_novelty_service()` in `src/api/service.py` is invoked.
  Step 3: `models/section_tfidf_extractor.joblib` and `models/ridge_fusion_model.joblib` are loaded into memory once.
  Step 4: `NoveltyExplainer` is initialized and cached in memory.
  Step 5: SQLite database `data/proposals.db` initializes its schema (`submitted_proposals` table).

PHASE 3: RUNTIME PROPOSAL EVALUATION LIFECYCLE
  Step 1: User fills the 8 proposal fields in the React frontend and clicks "Evaluate Novelty".
  Step 2: Frontend sends an HTTP POST request to `/api/score`.
  Step 3: FastAPI validates the payload using Pydantic schema `ProposalInput` in `schemas.py`.
  Step 4: `NoveltyService.score_proposal()` executes:
          - Transforms the 8 candidate text sections using the 8 pre-fitted TF-IDF vectorizers.
          - Computes cosine similarity against all 600 reference projects.
          - Multiplies similarities by Ridge coefficients and adds the intercept.
          - Applies self-exclusion rules (filtering out exact ID, title, or identical records).
          - Sorts historical projects by predicted similarity descending.
          - Generates XAI evidence (snippets and shared keywords) for the top 5 matches.
  Step 5: API returns the complete XAI JSON payload to the frontend.
  Step 6: React renders the novelty gauge, Recharts bar charts, and expandable match cards.
  Step 7: The candidate proposal is NOT saved to the database during scoring.
  Step 8: If the user explicitly clicks "Save Proposal", a separate HTTP POST to `/api/save` stores the proposal and its evaluation report in `data/proposals.db`.
"""

CHAPTER_3_TITLE = "Chapter 3: Dataset Explanation & Schema Inspection"
CHAPTER_3_TEXT = """================================================================================
CHAPTER 3: DATASET EXPLANATION & SCHEMA INSPECTION
================================================================================

3.1 Raw Dataset Overview: Academic_Project_Similarity_Synthetic_Dataset.xlsx
----------------------------------------------------------------------------
The system's training and baseline comparison data resides in `data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx`. This Microsoft Excel workbook consists of four distinct sheets:
  1. `Projects`: Contains 600 unique historical academic project records.
  2. `SimilarityPairs`: Contains 3,000 labeled pairwise comparisons between projects.
  3. `DataDictionary`: Documents field definitions, valid ranges, and source sheets.
  4. `README`: Describes dataset provenance, purpose, and synthetic limitations.

3.2 Inspection of the `Projects` Sheet (600 Rows, 16 Columns)
--------------------------------------------------------------
The `Projects` sheet defines the reference universe of historical capstone projects. It contains 600 rows and 16 columns:

+----------------------+--------------------+---------------------------------------------------------------+
| Field Name           | Data Type (Python) | Description / Role                                            |
+----------------------+--------------------+---------------------------------------------------------------+
| project_id           | string (object)    | Unique primary key (format: "SYN-0001" to "SYN-0600").        |
| academic_year        | string (object)    | Academic cohort year (e.g., "2022-23", "2023-24").            |
| department           | string (object)    | Academic unit (e.g., "Computer Science", "IT", "MCA").       |
| program_level        | string (object)    | Student tier: "UG Mini Project", "UG Mega Project", "PG Diss".|
| project_status       | string (object)    | Operational status (all marked "Completed").                  |
| domain               | string (object)    | Technical domain (12 distinct academic disciplines).          |
| project_title        | string (object)    | [SECTION 1] Project title.                                    |
| abstract             | string (object)    | [SECTION 2] Comprehensive summary abstract.                   |
| problem_statement    | string (object)    | [SECTION 3] Domain challenge and motivation.                  |
| objectives           | string (object)    | [SECTION 4] Numbered technical research goals.                |
| proposed_methodology | string (object)    | [SECTION 5] Architecture, algorithms, and dataflow.           |
| dataset              | string (object)    | [SECTION 6] Datasets and benchmarks used.                     |
| algorithm_model      | string (object)    | [SECTION 7] Primary machine learning/algorithmic models.     |
| technology_tools     | string (object)    | [SECTION 8] Software libraries, tools, and frameworks.        |
| keywords             | string (object)    | Comma-separated domain keywords.                              |
| data_source          | string (object)    | Provenance flag: 100% of rows contain "SYNTHETIC".            |
+----------------------+--------------------+---------------------------------------------------------------+

Program Level Distribution (Exactly balanced):
  - UG Mini Projects: 200 records (33.33%)
  - UG Mega Projects: 200 records (33.33%)
  - PG Dissertations: 200 records (33.33%)

Domain Coverage (12 Domains):
  - Artificial Intelligence & Machine Learning
  - Computer Vision
  - Natural Language Processing
  - Internet of Things (IoT)
  - Cybersecurity
  - Blockchain
  - Cloud Computing
  - FinTech
  - Healthcare & Medical Informatics
  - Smart Agriculture
  - Education Technology
  - Computer Networks & Software Defined Networking (SDN)

3.3 Inspection of the `SimilarityPairs` Sheet (3,000 Rows, 8 Columns)
---------------------------------------------------------------------
The `SimilarityPairs` sheet contains the training targets for the regression model:

+--------------------------------+--------------------+---------------------------------------------------------------+
| Field Name                     | Data Type (Python) | Description / Role                                            |
+--------------------------------+--------------------+---------------------------------------------------------------+
| pair_id                        | string (object)    | Unique identifier (format: "PAIR-0001" to "PAIR-3000").       |
| project_A_id                   | string (object)    | Foreign key referencing project_id of Project A.             |
| project_B_id                   | string (object)    | Foreign key referencing project_id of Project B.             |
| same_semantic_family           | integer (int64)    | Binary flag: 1 if projects share a domain/topic, 0 otherwise.|
| baseline_lexical_overlap_proxy | float (float64)    | Continuous synthetic proxy of word overlap [0.6364 to 1.0].   |
| expert_similarity_label        | integer (int64)    | REGRESSION TARGET: Integer similarity label in [0, 4].        |
| similarity_category            | string (object)    | Human-readable tier name (e.g., "Not Similar", "Highly Sim"). |
| annotation_note                | string (object)    | Metadata disclosure: "SYNTHETIC BENCHMARK LABEL...".         |
+--------------------------------+--------------------+---------------------------------------------------------------+

Target Similarity Label Distribution (Before vs. After Cleaning):
  - Label 0 ("Not Similar"): 1,336 original rows (44.53%) -> 1,332 clean rows (44.46%)
  - Label 1 ("Slightly Similar"): 334 rows (11.13%) -> 334 clean rows (11.15%)
  - Label 2 ("Moderately Similar"): 514 rows (17.13%) -> 514 clean rows (17.16%)
  - Label 3 ("Highly Similar"): 581 rows (19.37%) -> 581 clean rows (19.39%)
  - Label 4 ("Nearly Duplicate"): 235 rows (7.83%) -> 235 clean rows (7.84%)
  Total: 3,000 original pairs -> Exactly 2,996 clean pairs.

3.4 Audit of Dataset Deduplication (Removal of 4 Redundant Rows)
----------------------------------------------------------------
During data cleaning (`scripts/clean_dataset.py`), the 3,000 pairs were audited for duplicate relationships:
  1. PAIR-1819 (SYN-0524 vs SYN-0451, Label 0): Removed as an exact duplicate of PAIR-1324.
  2. PAIR-2023 (SYN-0130 vs SYN-0097, Label 0): Removed as a reversed duplicate of PAIR-0432 (SYN-0097 vs SYN-0130).
  3. PAIR-2195 (SYN-0165 vs SYN-0249, Label 0): Removed as a reversed duplicate of PAIR-1401 (SYN-0249 vs SYN-0165).
  4. PAIR-2957 (SYN-0538 vs SYN-0174, Label 0): Removed as a reversed duplicate of PAIR-2588 (SYN-0174 vs SYN-0538).

Concordance Check: In all four duplicate instances, both counterparts carried identical label 0 annotations and matching proxy scores. No contradictory annotations existed. The cleaned dataset of 2,996 rows is saved in `data/processed/similarity_pairs_clean.csv`.

3.5 Ground Truth Disclosure: Synthetic Nature & Faculty Status
--------------------------------------------------------------
CRITICAL VIVA / REPORT DISCLOSURE:
  - 100% of the similarity labels are SYNTHETIC. The README explicitly states:
    "Synthetic labels are NOT ground truth. For publication, a representative subset must be independently annotated by faculty/domain experts..."
  - All 3,000 pairs have the note: "SYNTHETIC BENCHMARK LABEL — must be replaced/validated by faculty annotation before publication."
  - An inspection of `data/raw/Faculty_Verification_DomainWise.xlsx` confirms that the columns `Faculty Verdict (0-4)`, `Faculty Comments`, and `Verified By` contain ZERO (0) non-null values across all 13 sheets.
  - CONCLUSION: The machine learning model was trained to emulate a synthetic labeling rule, NOT real human academic expert judgment.

3.6 Training Dataset vs. Runtime Reference Corpus
-------------------------------------------------
It is critical to distinguish how data is used during training versus runtime:
  - Training Dataset: The 2,996 pairwise comparisons in `data/processed/similarity_pairs_clean.csv` (using 480 training projects) were used to train the Ridge regression model weights.
  - Runtime Reference Corpus: The 600 project records from the `Projects` sheet are serialized inside `models/section_tfidf_extractor.joblib`. When a user submits a new proposal at runtime, the candidate proposal is compared against these 600 fixed baseline projects.
  - Separation of User Submissions: Newly evaluated user proposals are saved into SQLite (`data/proposals.db`). They DO NOT enter or contaminate the 600 baseline projects in the reference corpus.

3.7 The Eight Proposal Sections & Their Individual Contributions
-----------------------------------------------------------------
1. `project_title`: Captures high-level focus (e.g., "Crop Disease Detection using CNN"). Short text (average 8-12 words).
2. `abstract`: Comprehensive summary containing technical context, primary dataset, algorithms, and expected performance.
3. `problem_statement`: Highlights the domain deficiency or literature gap motivating the research.
4. `objectives`: Outlines the technical milestones and deliverables.
5. `proposed_methodology`: Detailed operational workflow, mathematical pipelines, and architecture diagrams described in text.
6. `dataset`: The named data resources. IN THIS MODEL, THIS IS THE MOST SENSITIVE SECTION.
7. `algorithm_model`: Specific ML models, neural architectures, or algorithmic techniques.
8. `technology_tools`: Libraries, programming frameworks, and hardware environments.
"""

CHAPTER_4_TITLE = "Chapter 4: How Training Data Is Prepared"
CHAPTER_4_TEXT = """================================================================================
CHAPTER 4: HOW TRAINING DATA IS PREPARED
================================================================================

4.1 The Complete Data Preparation Pipeline
------------------------------------------
The transformation from raw Excel workbooks into machine-readable feature matrices proceeds through five verifiable steps:

Raw Excel Files -> Data Loading & Validation -> Deduplication -> Project-Disjoint Splitting -> Pairwise Feature Extraction

4.2 Step 1: Loading Raw Data (`src/data/loader.py`)
---------------------------------------------------
Data loading is managed by three functions in `src/data/loader.py`:
  - `load_projects(filepath)`: Loads the `Projects` sheet using `pandas.read_excel`. Validates that all 16 required columns exist and that `project_id` values are unique non-empty strings. Missing text fields are imputed with empty strings (`""`).
  - `load_similarity_pairs(filepath)`: Loads the `SimilarityPairs` sheet (or clean CSV). Verifies required keys (`pair_id`, `project_A_id`, `project_B_id`, `expert_similarity_label`), strips whitespace from identifiers, and casts labels to integers.
  - `load_feature_csv(filepath)`: Loads pre-extracted feature tables, validating that feature columns are numeric floats.

4.3 Step 2: Cleaning and Deduplication (`scripts/clean_dataset.py`)
-------------------------------------------------------------------
The raw 3,000 pairs contain undirected comparisons where project order does not change semantic similarity. `clean_dataset.py` constructs a canonical undirected key for every pair:

              canonical_key = (min(proj_A, proj_B), max(proj_A, proj_B))

  - Rows sharing the same `canonical_key` are flagged as duplicates.
  - Exactly one exact duplicate (`PAIR-1819`) and three reversed duplicates (`PAIR-2023`, `PAIR-2195`, `PAIR-2957`) were dropped.
  - The remaining 2,996 clean rows were exported to `data/processed/similarity_pairs_clean.csv`.

4.4 Step 3: Project-Disjoint Partitioning (`src/utils/metrics.py`)
------------------------------------------------------------------
Standard random splitting of pair rows causes severe data leakage: if Project SYN-0010 appears in a training pair (SYN-0010 vs SYN-0025) and in a test pair (SYN-0010 vs SYN-0080), the model has already memorized the vocabulary and features of SYN-0010 during training!

To prevent data leakage, `project_disjoint_split` partitions the dataset by UNIQUE PROJECTS, not pair rows:
  1. Extract all 600 unique project IDs.
  2. Shuffle project IDs using a fixed seed (`seed=42`).
  3. Partition project IDs into 80% training projects (480 projects) and 20% test projects (120 projects).
  4. Train Pairs: Kept ONLY if BOTH `project_A_id` AND `project_B_id` belong to the 480 training projects (yielding 1,916 pairs).
  5. Test Pairs: Kept ONLY if BOTH `project_A_id` AND `project_B_id` belong to the 120 test projects (yielding 120 pairs).
  6. Cross-Boundary Pairs: Any pair where one project is in Train and the other is in Test (960 pairs) is DISCARDED entirely from evaluation.

Verification:
  - Train Pairs: 1,916
  - Test Pairs: 120
  - Cross-Boundary Pairs Discarded: 960
  - Total: 1,916 + 120 + 960 = 2,996 clean pairs.
  - Leakage status: ZERO project overlap across train and test partitions.

4.5 Step 4: Constructing Pairwise Features (`src/features/section_tfidf.py`)
----------------------------------------------------------------------------
For each pair $(A, B)$ in the training set, the model does NOT simply concatenate raw text. Instead, it extracts an eight-dimensional feature vector $\\vec{x} = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]^T$, where each $s_i$ represents the cosine similarity between the TF-IDF vectors of Section $i$:

               s_i = cosine_similarity( v_A^{(i)}, v_B^{(i)} )

Where:
  - $v_A^{(i)}$ is the TF-IDF representation of Project A in Section $i$.
  - $v_B^{(i)}$ is the TF-IDF representation of Project B in Section $i$.
  - The 8 section features are:
      1. `title_similarity`
      2. `abstract_similarity`
      3. `problem_similarity`
      4. `objectives_similarity`
      5. `methodology_similarity`
      6. `dataset_similarity`
      7. `algorithm_similarity`
      8. `technology_similarity`

4.6 Step 5: Understanding the Synthetic Label Generation Logic
--------------------------------------------------------------
Analysis of `data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx` reveals the deterministic generation rules:
  - When `same_semantic_family == 0` (projects from different domains):
    The synthetic generator assigned `expert_similarity_label = 0` ("Not Similar").
  - When `same_semantic_family == 1` (projects within the same domain/family):
    The synthetic generator assigned labels 1, 2, 3, or 4 based on token overlap templates:
      * Label 1: Partial overlap in generic tools.
      * Label 2: Overlap in problem domain and methodology.
      * Label 3: Overlap in problem, methodology, and dataset.
      * Label 4: Near-identical dataset, algorithm, and abstract.

4.7 Concrete Before-and-After Example
-------------------------------------
BEFORE PREPARATION (Raw Text Pair):
  - Project A (SYN-0001):
      * `project_title`: "Prototype Crop Disease Detection Using CNN"
      * `dataset`: "PlantVillage"
      * `algorithm_model`: "CNN"
  - Project B (SYN-0157):
      * `project_title`: "AI-Assisted Crop Disease Detection Using CNN"
      * `dataset`: "PlantVillage"
      * `algorithm_model`: "CNN"
  - Ground Truth Label: 3 ("Highly Similar")

AFTER PREPARATION (Feature Vector Row):
  - `title_similarity`: 0.6131
  - `abstract_similarity`: 1.0000
  - `problem_similarity`: 1.0000
  - `objectives_similarity`: 1.0000
  - `methodology_similarity`: 0.0613
  - `dataset_similarity`: 1.0000
  - `algorithm_similarity`: 1.0000
  - `technology_similarity`: 0.0425
  - Target `y`: 3.0
"""

print("Writing Chapters 1 to 4 to text...")
