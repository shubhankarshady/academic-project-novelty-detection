# Academic Project Novelty Detection System
## Final Project Presentation Outline & Slide Deck Guide

**Degree / Course**: Bachelor of Technology in Computer Science & Engineering  
**Project Category**: Natural Language Processing / Explainable AI / Web Application  
**Target Duration**: 15–20 Minutes (followed by Demonstration & Viva)  

---

### Slide 1: Title & Project Identification

- **Slide Title**: Academic Project Novelty Detection Using Multi-Section Lexical-Semantic Fusion and Explainable AI
- **Header Details**: Final Year Project Evaluation / Viva Examination
- **Presenter(s)**: [Student Name(s) & Roll Numbers]
- **Supervisor**: [Faculty Guide Name & Department]
- **Institution**: Department of Computer Science & Engineering
- **Suggested Visuals**: 
  - Dual-pane icon representation: Structured academic document on the left $\rightarrow$ Explainable novelty gauge on the right.
  - Institution logo and date.
- **Speaker Notes**:
  > "Good morning, respected members of the evaluation committee and external examiner. Today, we present our final year capstone project: 'Academic Project Novelty Detection Using Multi-Section Lexical-Semantic Fusion and Explainable AI'. This system addresses the growing problem of project duplication, academic redundancy, and lack of transparency during proposal evaluations."

---

### Slide 2: Problem Statement & Academic Motivation

- **Key Points**:
  - **The Academic Screening Challenge**: University departments review hundreds of capstone proposals annually. Manual verification across years of past archives is prone to human oversight.
  - **Superficial Wording vs. Real Duplication**: Students often rephrase existing projects under new titles, which slips past traditional keyword search while replicating datasets and methodologies.
  - **Inadequacy of General Plagiarism Checkers**: Tools like Turnitin match verbatim character strings across the public internet; they do not understand academic proposal structures (Title, Dataset, Methodology) or technical concept equivalence.
  - **The Black-Box Dilemma**: Opaque neural networks that give a single similarity score cannot justify rejection or guidance to students.
- **Suggested Visuals**:
  - Diagram contrasting *"Verbatim Web Plagiarism"* vs. *"Structural Academic Redundancy"*.
- **Speaker Notes**:
  > "Every academic semester, faculty panels face duplicate proposals where titles are changed, but the dataset, algorithm, and problem are completely identical to projects submitted two years prior. Traditional plagiarism checkers only look for copied text strings on the internet. They do not parse structured academic sections, nor can they tell faculty why two projects are conceptually redundant. Our goal was to build a domain-aware, transparent, section-based novelty detection platform."

---

### Slide 3: Project Objectives

- **Key Points**:
  - **1. Section-Aware Representation**: Parse proposals into 8 functional sections (Title, Abstract, Problem, Objectives, Methodology, Dataset, Algorithm, Tools).
  - **2. Dual Lexical-Semantic Fusion**: Combine sublinear lexical n-grams (exact terminology) with dense contextual embeddings (paraphrase comprehension).
  - **3. Bounded Monotonic Scoring**: Develop a constrained linear fusion head ensuring non-negative weights ($w_k \ge 0$), bounded similarity $[0.0, 4.0]$, and an exact Novelty Percentage $[0.0\%, 100.0\%]$.
  - **4. Transparent Explainability (XAI)**: Provide exact additive feature attribution, section overlap classifications, and word-boundary keyword context snippets.
  - **5. Controlled Empirical Calibration**: Tune hybrid parameter $\alpha$ via cross-validation strictly on calibration data without leaking held-out faculty evaluations.
  - **6. Full-Stack Interactive Platform**: Deliver a production-ready FastAPI backend and React 19 SaaS dashboard with persistent SQLite audit storage.
- **Suggested Visuals**:
  - Six numbered circular milestone badges illustrating each objective.
- **Speaker Notes**:
  > "Our objectives were driven by engineering rigor and academic utility. We aimed not just to generate a similarity percentage, but to evaluate proposals along 8 distinct academic dimensions, provide transparent additive explanations, and build a complete web application with a FastAPI backend and a React dashboard."

---

### Slide 4: Limitations of Existing Systems

- **Key Points (Comparative Table)**:

| Approach | Working Principle | Major Limitation |
|---|---|---|
| **Manual Faculty Screening** | Memory and manual keyword searches | Subjective, unscalable across multi-year institutional archives |
| **Commercial Plagiarism Tools** | Monolithic text string matching across web | Blind to paraphrasing; ignores structural academic sections |
| **Pure TF-IDF Matching** | Lexical bag-of-words / n-gram overlap | Completely fails when students use synonymous vocabulary |
| **Pure Transformer Embeddings** | Cosine similarity of whole-document vector | Suffers from high background boilerplate noise; unexplainable |

- **Suggested Visuals**:
  - Table showing comparison checkmarks and crossmarks across approaches.
- **Speaker Notes**:
  > "When we reviewed existing solutions, we observed a clear trade-off. Manual reviews don't scale. Generic plagiarism checkers treat proposals as flat text. Pure lexical TF-IDF misses paraphrased duplicates, while pure deep transformer embeddings get confused by common academic filler phrases like 'the proposed system aims to'. This necessitated a hybrid, section-weighted architecture."

---

### Slide 5: Proposed System Overview

- **Key Points**:
  - **Dual-Channel Processing**: Evaluates each proposal through two parallel pipelines:
    - *Lexical Branch*: 8 Sublinear TF-IDF vectorizers (unigrams + bigrams).
    - *Semantic Branch*: `all-MiniLM-L6-v2` generating 384-dimensional normalized dense vectors.
  - **Calibrated Convex Fusion**: $s_k^{\text{hybrid}} = \alpha \cdot s_k^{\text{lex}} + (1 - \alpha) \cdot s_k^{\text{sem}}$ with empirically tuned $\alpha = 0.60$.
  - **Constrained Linear Combination**: $\hat{y} = \sum_{k=1}^8 w_k \cdot s_k^{\text{hybrid}}$ with non-negative candidate weights summing to $4.0$.
  - **Three Operational Modes**:
    1. *TF-IDF Baseline* (Production default)
    2. *Hybrid Baseline* ($\alpha=0.50$)
    3. *Tuned Hybrid* ($\alpha=0.60$, Recommended)
- **Suggested Visuals**:
  - High-level pipeline flowchart: Proposal Input $\rightarrow$ Dual Section Extractors $\rightarrow$ Hybrid Fusion $\rightarrow$ Constrained Head $\rightarrow$ XAI Report.
- **Speaker Notes**:
  > "The proposed architecture processes candidate proposals through two complementary branches. The lexical branch catches identical libraries, dataset names, and exact keywords. The semantic branch catches conceptual paraphrasing using sentence-transformers. These are merged using a convex weighting factor alpha, then passed to a constrained linear model."

---

### Slide 6: Dataset & Evaluation Cohort

- **Key Points**:
  - **Baseline Historical Reference Corpus**:
    - **600 structured academic projects** across 7 domains (AI/ML, Data Science, Cyber Security, Healthcare Informatics, Cloud, IoT, Web).
    - Contains all 8 text fields representing realistic undergraduate and master's projects.
  - **Faculty Evaluation Dataset ($N=120$ Pairs)**:
    - Evaluated on a discrete ordinal scale from $0$ (Unrelated) to $4$ (Identical/Near-Duplicate).
    - **Strict Split Protocol**:
      - `DEV_CALIBRATION` ($N = 30$ pairs): Used exclusively for $\alpha$ tuning via 5-fold cross-validation.
      - `HELD_OUT_EVALUATION` ($N = 90$ pairs): Kept strictly untouched and unblinded until final verification.
  - **Scientific Governance Note**:
    - Ratings represent preliminary human evaluation from a single faculty reviewer (`Reviewer 1`). Reviewer 2 and panel adjudication remain pending.
- **Suggested Visuals**:
  - Pie chart showing 600-project domain distribution; Split diagram showing $30$ dev vs. $90$ held-out pairs.
- **Speaker Notes**:
  > "For our baseline reference corpus, we indexed 600 structured projects across 7 major computing domains. For scientific evaluation, we prepared 120 unique project pairs annotated by our faculty reviewer on a 0 to 4 scale. Crucially, to prevent data leakage, we partitioned this into 30 development pairs for calibration, and kept 90 pairs completely untouched for final held-out testing."

---

### Slide 7: Structured 8-Section Proposal Schema & Preprocessing

- **Key Points**:
  - **The 8 Required Proposal Sections**:
    1. `Title`
    2. `Abstract`
    3. `Problem Statement`
    4. `Objectives`
    5. `Proposed Methodology`
    6. `Dataset`
    7. `Algorithm / Model`
    8. `Technology & Tools`
  - **Text Preprocessing Pipeline**:
    - Unicode normalization and lowercase conversion.
    - Preserves technical hyphens (e.g., *ResNet-50*, *TF-IDF*) while stripping noisy control characters.
    - English stop-word filtering during TF-IDF vectorization.
    - Content hashing: SHA-256 fingerprinting prevents redundant database submissions.
- **Suggested Visuals**:
  - Graphic of an academic proposal showing the 8 highlighted bounding boxes.
- **Speaker Notes**:
  > "Rather than treating the proposal as a single monolithic block of text, we parse it into 8 standard functional sections. Preprocessing cleans the text, removes stop-words, and validates section length. Each section is analyzed independently, allowing the system to distinguish between a shared tool and an identical dataset."

---

### Slide 8: Lexical TF-IDF vs. Dense Semantic Embeddings

- **Key Points**:
  - **Lexical Branch (Sublinear TF-IDF)**:
    - Formula: $\text{tf}_{\text{sublinear}} = 1 + \log(\text{tf})$
    - Captures exact vocabulary, n-grams (1, 2), tool libraries, and benchmark identifiers.
    - Fast and deterministic, but blind to synonyms.
  - **Semantic Branch (Dense Embeddings)**:
    - Pre-trained `all-MiniLM-L6-v2` generating 384-dimensional unit normalized vectors ($\|\mathbf{e}\|_2 = 1.0$).
    - Negative dot products clamped to $0.0$ to ensure monotonic bounds: $s^{\text{sem}} = \max(0.0, \min(1.0, \mathbf{e}_C \cdot \mathbf{e}_H))$.
    - Captures conceptual phrasing (e.g., *"foliar pathology"* matches *"crop disease"*).
- **Suggested Visuals**:
  - Side-by-side comparison diagram showing vocabulary vector space vs. contextual vector space.
- **Speaker Notes**:
  > "Here we compare the two feature channels. TF-IDF uses sublinear term scaling to dampen repetitive words and extracts unigrams and bigrams. It is great for exact library names like PyTorch or dataset names like PlantVillage. MiniLM generates dense 384-dimensional embeddings that capture conceptual semantics. Clamping negative cosine values guarantees that similarity never drops below zero."

---

### Slide 9: Convex Hybrid Fusion & Alpha Tuning

- **Key Points**:
  - **Hybrid Convex Combination**:
    $$s_k^{\text{hybrid}} = \alpha \cdot s_k^{\text{lex}} + (1 - \alpha) \cdot s_k^{\text{sem}}$$
  - **5-Fold Cross-Validation on Calibration Split ($N=30$)**:
    - Grid tested: $\alpha \in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]$
    - $\alpha = 0.60$ achieved minimum CV RMSE ($0.3921$), near-zero bias ($\text{MBE} = -0.0356$), and maximum exact agreement ($80.0\%$).
  - **Balancing Trade-offs**:
    - Lower $\alpha \le 0.40$: Semantic channel overpredicts similarity due to generic academic jargon.
    - Higher $\alpha \ge 0.70$: Pure lexical bias misses paraphrased projects.
    - **$\alpha = 0.60$ represents the optimal 60% Lexical + 40% Semantic balance**.
- **Suggested Visuals**:
  - Line chart plotting CV RMSE and Mean Bias Error against tested values of $\alpha$, showing the minimum at $\alpha = 0.60$.
- **Speaker Notes**:
  > "To combine lexical and semantic similarities, we use a convex combination parameterized by alpha. On our 30 calibration pairs, we ran 5-fold cross-validation across alpha values from 0.2 to 0.8. At alpha 0.60—meaning 60% TF-IDF and 40% semantic—we achieved the lowest RMSE of 0.3921 and near-zero bias. This configuration was frozen for held-out testing."

---

### Slide 10: Section Weighting & Novelty Percentage

- **Key Points**:
  - **Constrained Linear Head (SLSQP Optimization)**:
    $$\hat{y} = \sum_{k=1}^8 w_k \cdot s_k^{\text{hybrid}}, \quad w_k \ge 0, \quad \sum w_k = 4.0, \quad b = 0.0$$
  - **Active Empirical Weights**:
    - **Dataset**: $w = 1.9529$ ($48.82\%$ influence) — *Primary novelty factor*.
    - **Title**: $w = 0.9535$ ($23.84\%$ influence) — *Core domain indicator*.
    - **Technology & Tools**: $w = 0.7160$ ($17.90\%$ influence) — *Technical ecosystem*.
    - **Proposed Methodology**: $w = 0.3776$ ($9.44\%$ influence) — *Experimental procedure*.
    - **Abstract, Problem, Objectives, Algorithm**: $w = 0.0$ ($0.0\%$) — *Eliminates generic boilerplate variance*.
  - **Novelty Percentage Formula**:
    $$\text{Novelty \%} = \left(1 - \frac{\hat{y}_{\max}}{4.0}\right) \times 100\%$$
- **Suggested Visuals**:
  - Horizontal bar chart showing the relative percentage weights of the 8 sections; Bounded scale diagram mapping similarity $[0, 4] \rightarrow$ Novelty $[100\%, 0\%]$.
- **Speaker Notes**:
  > "Instead of standard Ridge regression which produced negative weights and negative intercepts, we designed a constrained model where all weights are non-negative, the intercept is zero, and the weights sum exactly to 4.0. Dataset similarity carries 48.8% of the weight because reusing an identical benchmark is the primary driver of redundancy in computer science capstone projects."

---

### Slide 11: End-to-End System Architecture

- **Key Points**:
  - **Frontend Layer**: React 19, Tailwind CSS, Lucide icons, Vite build system.
  - **Backend API Layer**: FastAPI REST service with automated OpenAPI documentation.
  - **In-Memory Lifespan Preloading**: Heavy transformer models and TF-IDF matrixes load once into RAM at startup for sub-second inference.
  - **Persistence Layer**: SQLite (`data/proposals.db`) with relational integrity between proposals and XAI reports.
  - **Stateless Scoring**: Evaluating a proposal (`POST /api/score`) is strictly non-mutating; saving is deferred until explicitly committed.
- **Suggested Visuals**:
  - System architecture diagram (`docs/diagrams/system_architecture.mmd`) showing communication across Client, API, Service, and SQLite layers.
- **Speaker Notes**:
  > "Our software architecture is built for performance and maintainability. The FastAPI backend loads the machine learning artifacts once at startup. When a student evaluates a proposal, the scoring is performed entirely in memory without writing to the database. Only when the evaluator clicks 'Save Proposal' is it committed to SQLite with duplicate checking."

---

### Slide 12: Explainable AI (XAI) Methodology

- **Key Points**:
  - **1. Additive Linear Feature Attribution**:
    - Because $b = 0.0$, the similarity score is the exact sum of signed contributions: $\hat{y} = \sum w_i s_i$.
  - **2. Section Categorization**:
    - *High-Overlap Sections* ($\ge 0.30$): Flags areas of critical redundancy.
    - *Differentiating Sections* ($< 0.15$): Highlights original contributions.
  - **3. Keyword Evidence Extraction**:
    - Extracts shared informative technical n-grams while filtering generic academic stop-words.
    - Returns word-boundary text snippets quoting source context from both candidate and historical projects.
- **Suggested Visuals**:
  - Screenshot of the XAI section breakdown and keyword evidence card from the application UI.
- **Speaker Notes**:
  > "Explainability is a core feature of our platform. Because our model has a zero intercept, every predicted score can be broken down additively: faculty can see exactly how many points came from the dataset versus the title. Furthermore, the system extracts verbatim keyword snippets from the proposals, proving why an overlap was detected."

---

### Slide 13: Experimental Results (Held-Out Evaluation, $N=90$)

- **Key Points**:
  - Evaluated on **90 untouched held-out pairs**:

| Metric | TF-IDF Ridge Baseline | Semantic Model | Baseline Hybrid ($\alpha=0.50$) | **Tuned Hybrid ($\alpha=0.60$)** |
|---|:---:|:---:|:---:|:---:|
| **Exact Ordinal Agreement** | 83.33% | 65.56% | 74.44% | **84.44% (+10.0%)** |
| **Macro $F_1$ Score** | 50.62% | 40.00% | 57.35% | **69.72% (+12.4%)** |
| **Within-One Rating Agreement** | 97.78% | 98.89% | 98.89% | **98.89%** |
| **Mean Absolute Error (MAE)** | 0.4713 | 0.4563 | 0.3155 | **0.3270** |
| **Continuous RMSE** | 0.6379 | 0.5869 | 0.4421 | **0.4410** |
| **Spearman Rank Correlation** | 0.7716 | 0.8299 | 0.8502 | **0.8514** |

- **Zero Major Classification Errors**: 89 of 90 predictions were within one rating point of human faculty judgment.
- **Suggested Visuals**:
  - Grouped bar chart comparing Exact Agreement and Macro F1 across all four models; Confusion matrix heatmap for the Tuned Hybrid model.
- **Speaker Notes**:
  > "On our held-out test set of 90 pairs, the tuned hybrid model demonstrated clear superiority. Exact ordinal agreement reached 84.44%, outperforming both individual TF-IDF and semantic baselines. Most notably, Macro F1 surged from 57.35% to 69.72%, showing strong balanced performance across both duplicate and original proposals, with 98.89% of predictions within one rating point."

---

### Slide 14: Limitations & Governance Safeguards

- **Key Points**:
  - **1. Preliminary Human Annotation**:
    - Data derives from single-reviewer faculty annotations (`Reviewer 1`). Independent double-review and adjudication consensus remain in progress.
  - **2. Corpus-Relative Metric**:
    - Novelty is estimated strictly relative to the indexed 600-project reference repository. It does not establish absolute worldwide originality or prove legal plagiarism.
  - **3. Production Default Safeguard**:
    - In accordance with academic governance, `tfidf` baseline remains the default operational mode, while `tuned_hybrid` is selectable.
  - **4. Language Scope**:
    - Preprocessing and stop-word dictionaries currently target English academic proposals.
- **Suggested Visuals**:
  - Disclaimer banner showing the governance boundary icons and safety checks.
- **Speaker Notes**:
  > "We maintain complete scientific honesty regarding our limitations. Our evaluation is preliminary because it is based on a single faculty reviewer; multi-reviewer inter-rater reliability is part of future work. We also emphasize that novelty is relative to our institutional corpus, and the system is designed as a decision-support advisory tool, not an autonomous disciplinary judge."

---

### Slide 15: Conclusion & Viva Defense

- **Key Points**:
  - **Summary of Contributions**:
    - Formulated an 8-section structured representation for academic proposals.
    - Implemented and calibrated a hybrid lexical-semantic fusion model ($\alpha = 0.60$).
    - Designed an additive, non-negative explainability head ($[0.0, 4.0] \rightarrow [100\%, 0\%]$).
    - Built a robust, production-ready web application backed by 114 automated tests.
  - **Readiness for Live Demonstration**:
    - Backend running on `http://127.0.0.1:8000`.
    - Dashboard active on `http://localhost:5173`.
- **Suggested Visuals**:
  - Concluding slide with repository GitHub link, documentation links, and *"Thank You — Questions & Viva Defense"*.
- **Speaker Notes**:
  > "In conclusion, we have built an explainable, empirically calibrated, and fully functional system for academic project novelty detection. All 114 automated tests pass, the frontend builds cleanly, and the system is ready for live demonstration. We now invite your questions and feedback. Thank you."

