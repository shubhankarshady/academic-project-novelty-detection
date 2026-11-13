# Academic Project Novelty Detection & Explainability Platform

An end-to-end Machine Learning and Explainable AI (XAI) application designed to evaluate candidate academic computer science project proposals against a reference corpus of prior academic works.

The system decomposes proposals into eight standard academic text sections, calculates section-wise lexical similarities, predicts an aggregate similarity score using Ridge Regression, derives a percentage-based Novelty Score, and generates transparent, additive feature attributions with verified source-text keyword evidence.

---

## 1. Project Objective & Critical Limitations

### Primary Objective
To assist academic project review committees, faculty mentors, and student researchers by providing an automated, transparent baseline assessment of how closely a new project proposal matches previously completed or indexed academic works.

### Critical System Limitations (Current Prototype Status)
> [!WARNING]
> **Synthetic Benchmark Ground Truth:**
> The current system was trained and evaluated against **3,000 synthetic benchmark pairs** generated via heuristic similarity rules. Across the entire benchmark and faculty verification sheets, there are currently **zero (0) human faculty annotations**. High evaluation metrics ($R^2 \approx 0.77$, $\text{RMSE} \approx 0.65$) reflect mathematical rule fitting, not validated real-world academic novelty detection.

> [!CAUTION]
> **Not a Plagiarism or Originality Certifier:**
> The output novelty percentage is an estimated heuristic relative solely to the indexed 600-project reference corpus. The system **cannot** establish absolute academic originality, prove plagiarism, or detect copyright violations.

---

## 2. Application Architecture

The project is structured into three cleanly decoupled layers:

```
┌─────────────────────────────────────────────────────────────┐
│                 React 19 + Vite 8 Frontend                  │
│   (ProposalForm, NoveltyGauge, SectionCharts, SimilarCards) │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP REST (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend Service                  │
│       (/api/health, /api/score, /api/save, /api/proposals)   │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│   ML & Explainability (XAI)  │ │   SQLite Persistence Layer │
│ - SectionTFIDFExtractor      │ │ - submitted_proposals table│
│ - Ridge Regression Model     │ │ - Content hash deduplicat. │
│ - NoveltyExplainer Engine    │ │ - Strictly segregated from │
│ - Reference Corpus (600 projs│ │   reference scoring matrix │
└──────────────────────────────┘ └────────────────────────────┘
```

---

## 3. The Eight Academic Proposal Sections

Proposals are analyzed across eight distinct functional sections rather than as single concatenated text blobs:

| # | Section Key | Canonical Field | Description & Role |
|:---:|---|---|---|
| 1 | `title` | `project_title` | Project title; captures primary topic and domain keywords. |
| 2 | `abstract` | `abstract` | High-level summary of problem, methodology, and outcome. |
| 3 | `problem` | `problem_statement` | Specific real-world or theoretical challenge being addressed. |
| 4 | `objectives` | `objectives` | Core milestones, goals, and intended deliverable targets. |
| 5 | `methodology` | `proposed_methodology` | Technical workflows, data pipelines, and validation strategies. |
| 6 | `dataset` | `dataset` | Benchmark datasets, sensor feeds, or synthetic data sources. |
| 7 | `algorithm` | `algorithm_model` | Core algorithms, neural architectures, or mathematical models. |
| 8 | `technology` | `technology_tools` | Languages, frameworks, hardware, and deployment software stack. |

---

## 4. Dataset Structure & Quality Validation

### Reference Corpus & Benchmark Pairs
- **Reference Projects:** [`data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx`](file:///data/raw/Academic_Project_Similarity_Synthetic_Dataset.xlsx) (Sheet: `Projects`): Exactly **600 academic project records** spanning 12 CS/IT domains.
- **Raw Similarity Pairs:** Same workbook (Sheet: `SimilarityPairs`): **3,000 pairwise records**.
- **Cleaned Similarity Pairs:** [`data/processed/similarity_pairs_clean.csv`](file:///data/processed/similarity_pairs_clean.csv): **2,996 clean pairs**. Exactly 4 redundant rows (1 exact duplicate and 3 reversed pairs) were audited and removed via [`scripts/clean_dataset.py`](file:///scripts/clean_dataset.py).
- **Target Variable (`expert_similarity_label`):** Continuous/Integer similarity scale from `0.0` to `4.0`:
  - `0`: Not Similar ($44.5\%$)
  - `1`: Slightly Similar ($11.2\%$)
  - `2`: Moderately Similar ($17.2\%$)
  - `3`: Highly Similar ($19.4\%$)
  - `4`: Nearly Duplicate ($7.8\%$)
- **Faculty Verification Package:** [`data/raw/Faculty_Verification_DomainWise.xlsx`](file:///data/raw/Faculty_Verification_DomainWise.xlsx) contains 12 domain-specific review sheets structured for faculty panels to collect genuine human ground truth.

---

## 5. TF-IDF Feature Extraction & Cosine Similarity

- **Implementation:** [`SectionTFIDFExtractor`](file:///src/features/section_tfidf.py)
- **Configuration:** 8 independent `TfidfVectorizer` instances with `ngram_range=(1, 2)` and `stop_words="english"`.
- **Vocabulary Sizes:**
  - `project_title`: 450 terms
  - `abstract`: 454 terms
  - `problem_statement`: 199 terms
  - `objectives`: 188 terms
  - `proposed_methodology`: 227 terms
  - `dataset`: 83 terms
  - `algorithm_model`: 19 terms
  - `technology_tools`: 89 terms
- **Similarity Calculation:** Candidate proposal sections are vectorized against the fitted vocabularies and matched via cosine similarity:
  $$\text{sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} \in [0.0, 1.0]$$
- **Saved Checkpoint:** [`models/section_tfidf_extractor.joblib`](file:///models/section_tfidf_extractor.joblib) (2.0 MB).

---

## 6. Ridge Regression & Active Model (8 Features)

The active production serving model predicts pairwise similarity on the $[0.0, 4.0]$ scale using an 8-dimensional lexical feature vector:

$$\hat{y} = b + \sum_{i=1}^8 w_i \cdot x_i$$

### Exact Learned Parameters ([`models/ridge_fusion_model.joblib`](file:///models/ridge_fusion_model.joblib))
- **Estimator:** `sklearn.linear_model.Ridge(alpha=1.0)`
- **Intercept ($b$):** `-0.0706`
- **Coefficients ($w$):**
  - `title_similarity`: `+0.1340`
  - `abstract_similarity`: `+0.3980`
  - `problem_similarity`: `-0.1259` *(Negative)*
  - `objectives_similarity`: `+0.2367`
  - `methodology_similarity`: `+0.0099`
  - `dataset_similarity`: `+2.2054` *(Dominates > 55% of the total scale)*
  - `algorithm_similarity`: `-0.0651` *(Negative)*
  - `technology_similarity`: `-0.1456` *(Negative)*

---

## 7. Dense Semantic Embedding Pipeline (Existing but Currently Inactive)

A dense transformer embedding pipeline was developed and trained, but is currently **disconnected from the live API serving pipeline**:
- **Implementation:** [`SectionDenseEmbeddingExtractor`](file:///src/features/dense_embeddings.py) using `SentenceTransformer("all-MiniLM-L6-v2")`.
- **Precomputed Artifact:** [`models/section_dense_extractor.joblib`](file:///models/section_dense_extractor.joblib) (99.4 MB) contains 384-dimensional normalized embeddings for all 600 reference projects across all 8 sections.
- **Hybrid Fusion Model:** [`src/models/hybrid_fusion.py`](file:///src/models/hybrid_fusion.py) and [`models/hybrid_fusion_model.joblib`](file:///models/hybrid_fusion_model.joblib) train a 16-feature Ridge model (8 TF-IDF + 8 Dense).
- **Reason Inactive in Production:** In benchmark experiments on synthetic labels, dense features exhibited severe collinearity with lexical features, leading to negative weights without improving RMSE ($0.6544$ vs $0.6537$). To maintain simple linear explainability, runtime serving was wired strictly to TF-IDF. Reconnecting semantic search is planned for Phase 4.

---

## 8. Novelty Score Formula & Known Model Defects

### Novelty Score Formula
In [`src/utils/metrics.py`](file:///src/utils/metrics.py) and [`src/explainability/explainer.py`](file:///src/explainability/explainer.py):
$$\text{sim\_norm} = \text{clip}\left(\frac{\hat{y}}{4.0}, 0.0, 1.0\right)$$
$$\text{Novelty Score (\%)} = (1.0 - \text{sim\_norm}) \times 100.0$$

### Novelty Assessment Tiers
- $\ge 75.0\%$: High estimated novelty relative to the current database
- $50.0\% - 74.9\%$: Moderate Novelty (Partial Overlap)
- $25.0\% - 49.9\%$: Low Novelty (Significant Prior Work)
- $< 25.0\%$: Very Low Novelty / Potential Duplicate

### Known Model Defects
1. **Theoretical Ceiling Defect:**
   $$\hat{y}_{\max} = -0.0706 + \sum_{i=1}^8 w_i = 2.5768$$
   $$\text{Novelty}_{\min} = 100 \times \left(1 - \frac{2.5768}{4.0}\right) = 35.58\%$$
   An exact, 100% identical copy across all 8 sections can never score higher than $2.58 / 4.0$, receiving a novelty score of $35.58\%$. The "Very Low Novelty / Potential Duplicate" tier ($< 25\%$) is mathematically unreachable.
2. **Negative Weight Non-Monotonicity Defect:**
   Sharing identical `technology_tools` ($-0.1456$), `problem_statement` ($-0.1259$), or `algorithm_model` ($-0.0651$) penalizes similarity. Consequently, an exact copy ranks #6 in the corpus while near-peers with differing technology rank higher.
3. **Disproportionate Dataset Weight Defect:**
   The `dataset` section coefficient ($+2.2054$) accounts for $>55\%$ of the total 4.0 scale. Sharing a standard dataset name (e.g. `PlantVillage`) between completely unrelated domains drops novelty below $50\%$.
4. **Lexical Paraphrase Blindness:**
   Because runtime serving uses lexical TF-IDF only, a paraphrased proposal expressing identical concepts in alternative vocabulary receives a similarity of $0.0175$, yielding an unwarranted $99.56\%$ novelty score.

---

## 9. Explainable AI (XAI) Implementation

Implemented in [`NoveltyExplainer`](file:///src/explainability/explainer.py):
1. **Additive Ridge Reconstruction:** Every predicted similarity is decomposed into signed contributions verifying:
   $$\text{Intercept} + \sum_{i=1}^8 \text{Signed Contribution}_i = \hat{y} \quad (\text{accurate to } < 10^{-5})$$
2. **Whole-Word Keyword Verification:** Shared n-grams are extracted via fitted TF-IDF vocabularies, filtered for generic academic boilerplate (`GENERIC_LOW_INFO_TERMS`), and strictly verified to appear with whole-word boundaries in both source texts.
3. **Source-Text Snippets:** Snippets are extracted with word-boundary snapping and ellipsis markers only when context was truncated.
4. **Self-Exclusion:** Candidate proposals are prevented from matching themselves by project ID, normalized title match, or feature vector identity.
5. **Transparency Disclaimers:** Every report includes disclaimers stating that attributions represent statistical associations relative to the indexed database, not causal proof of copying.

---

## 10. FastAPI Backend & SQLite Persistence

- **Backend Application:** [`src/api/main.py`](file:///src/api/main.py)
- **Singleton Service:** [`src/api/service.py`](file:///src/api/service.py) loads models once into memory on startup.
- **REST Endpoints:**
  - `GET /api/health`: Service health, corpus size (600), and submitted proposal count.
  - `POST /api/score`: Evaluates a candidate proposal against reference corpus with self-exclusion; does **not** persist proposal.
  - `POST /api/save`: Persists evaluated proposal and XAI report to SQLite; rejects duplicates with HTTP 409 Conflict.
  - `GET /api/proposals`: Paginated list of previously saved proposals (newest first).
  - `GET /api/proposals/{project_id}`: Full proposal record and saved evaluation JSON.
- **Database Architecture:** [`src/storage/db.py`](file:///src/storage/db.py) manages `data/proposals.db`.
  - Enforces deterministic SHA-256 content hashing across all 8 normalized sections.
  - Generates sequential submission IDs (`SUB-0001`, `SUB-0002`).
  - **Strict Independence Guarantee:** Saved user proposals are stored strictly for user auditing and history. They are **never** injected into the reference TF-IDF matrix and **never** trigger automatic model retraining.

---

## 11. Frontend Architecture

Built with **React 19**, **Vite 8**, **Tailwind CSS v4**, and **Recharts**:
- **Pages:**
  - `DashboardPage`: Corpus statistics, system health, and recent submissions table.
  - `EvaluatePage`: Full 8-section proposal submission form, live validation, interactive novelty score gauge, section similarity bar charts, and top similar project cards.
  - `HistoryPage`: Searchable, filterable audit log of past evaluations with detailed inspection modal.
- **Components:**
  - `SimilarProjectCard`: Renders predicted similarity ($[0.0, 4.0]$ scale), section-by-section breakdown, signed Ridge contributions, shared keywords, and side-by-side source snippets.
  - `NoveltyGauge`: Visual SVG semi-circle gauge displaying calibrated novelty percentage.
  - `SectionCharts`: Recharts dual-bar chart comparing candidate proposal sections against retrieved matches.

---

## 12. Installation & Environment Setup

### Prerequisites
- Python 3.12 (tested on 3.12.9)
- Node.js 18+ and npm
- Windows PowerShell or POSIX-compatible terminal

### Backend Setup
```bash
# Clone repository
git clone https://github.com/shubhankarshady/academic-project-novelty-detection.git
cd academic-project-novelty-detection

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## 13. Commands to Start Applications

### Start the FastAPI Backend Server
From the project root:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at: `http://127.0.0.1:8000/docs`

### Start the React Frontend Development Server
In a separate terminal:
```bash
cd frontend
npm run dev
```
Frontend web application will be available at: `http://localhost:5173`

---

## 14. Commands to Run Tests & Evaluation

### Run Automated Unit and Integration Test Suite
```bash
python -m pytest tests/
```
Runs 49 automated test cases across API endpoints, SQLite storage, XAI engine, and scientific reliability.

### Run Novelty Scoring via Standalone CLI Tool
```bash
python scripts/score_novelty.py \
  --title "Deep Learning for Autonomous Drone Navigation" \
  --abstract "We present a deep reinforcement learning framework for quadrotor indoor path planning." \
  --dataset "Gazebo point cloud simulated telemetry" \
  --algorithm "Proximal Policy Optimization (PPO)" \
  --technology "ROS, Gazebo, PyTorch, Python" \
  --top_k 3
```

### Build Production Frontend Bundle
```bash
cd frontend
npm run build
```

---

## 15. Known Issues & Roadmap Phases

### Current Known Issues
1. **Ridge Theoretical Ceiling:** Maximum predicted similarity is $2.58 / 4.0$, capping minimum novelty at $35.58\%$.
2. **Negative Section Weights:** Negative weights on technology, problem statement, and algorithm cause non-monotonic rankings.
3. **Paraphrase Sensitivity:** Lexical TF-IDF misses synonym-substituted conceptual duplicates.
4. **Scikit-Learn Version Warning:** Serialized models (created with 1.9.1) emit unpickle warnings in 1.8.0 environments.
5. **Synthetic Evaluation Ground Truth:** Requires faculty annotations for true academic validation.

### Next Implementation Phases
- **Phase 1 (Current):** Establish accurate baseline, resolve documentation conflicts, and verify test reproducibility.
- **Phase 2 (Proposed):** Constrained optimization (Non-Negative Least Squares) to enforce non-negative weights ($w_i \ge 0$) and calibrate the prediction range ($X=\vec{1} \implies \hat{y}=4.0$).
- **Phase 3 (Proposed):** Faculty annotation collection and inter-annotator agreement validation.
- **Phase 4 (Proposed):** Integrate dense semantic embeddings into the live API for paraphrase defense.
- **Phase 5 (Proposed):** Update frontend cards to reflect hybrid retrieval channels and calibrated scores.
- **Phase 6 (Proposed):** Final regression testing and continuous integration (CI).