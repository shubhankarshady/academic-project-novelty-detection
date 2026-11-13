# Academic Project Novelty Detection System
## 5–7 Minute Final Demonstration Script

**Target Duration**: 5 to 7 Minutes  
**Audience**: Final Year Viva Examination Committee, External Examiners, and Project Guides  
**Setup**: Two open terminal windows (FastAPI Backend + Vite Frontend) + One Google Chrome window  

---

## Pre-Demonstration Checklist (T-Minus 2 Minutes)

Before calling the examiners or starting your screen share, verify both services are live:
1. **Backend Terminal**: Running `python -m uvicorn src.api.main:app --port 8000 --reload`
2. **Frontend Terminal**: Running `npm --prefix frontend run dev`
3. **Browser**: Open `http://localhost:5173` in full-screen mode.
4. **Pre-Check Health**: Verify the green *"Healthy"* pill is visible in the top navigation bar.

---

## Demonstration Script & Chronological Timeline

```mermaid
timeline
    title 5-7 Minute Demonstration Flow
    0:00 - 1:00 : Dashboard Overview & System Health
    1:00 - 2:30 : Near-Duplicate Detection & Novelty Gauge
    2:30 - 4:00 : Model Comparison (TF-IDF vs Tuned Hybrid on Paraphrase)
    4:00 - 5:15 : Explainable AI & Keyword Snippet Inspection
    5:15 - 6:15 : Proposal Persistence & Audit History
    6:15 - 7:00 : Input Validation & Edge Case Handling
```

---

### Minute 0:00 – 1:00: Introduction & Dashboard Overview

**Action**: Display the **Dashboard Overview** tab (`http://localhost:5173`).

**What to Say**:
> *"Respected examiners, welcome to our demonstration of the Academic Project Novelty Detection System. 
> We begin on the System Dashboard. As you can see from our top metric cards, the system currently indexes a baseline reference corpus of **600 historical projects** across 7 computer science domains. 
> The system status indicates that our machine learning models and dense transformer embeddings are loaded directly in memory, and the active baseline scoring engine is set to our production TF-IDF Ridge model.
> Below, our submission audit table shows previous student proposals stored in our local SQLite database."*

**Expected Observations**:
- Top KPI stat cards render cleanly: `Reference Corpus: 600`, `Service Status: Healthy`, `Submitted Proposals: Active count`.
- Corpus Status card shows explainable AI badge.

---

### Minute 1:00 – 2:30: Scenario 1 — Evaluating a Near-Duplicate Proposal

**Action**:
1. Click the **"Evaluate New Proposal"** button in the top banner.
2. In the top-right toolbar dropdown, select **"Preset: Demo Sample"** (or Preset 0: Crop Disease Detection).
3. The 8 required proposal fields auto-populate.
4. Click the purple **"Evaluate Novelty"** button.

**What to Say**:
> *"Now we evaluate a student proposal. I will load our first test scenario: a proposal titled 'Automated Crop Disease Identification Using Deep Convolutional Networks'. 
> Notice that the proposal is structured into 8 distinct sections—Title, Abstract, Problem Statement, Objectives, Methodology, Dataset, Algorithm, and Tools.
> I will click 'Evaluate Novelty'. The FastAPI backend evaluates this candidate across all 600 projects in memory within 300 milliseconds.
> The system immediately redirects to the Novelty & Explainability Report.
> Notice our circular radial gauge: the candidate received a **low novelty score of 0% to 5%**, triggering the red **'Low Novelty (High Similarity / Near-Duplicate)'** tier. 
> The top matching project is identified immediately as `SYN-0001: Prototype Crop Disease Detection Using CNN` with an extreme similarity score of nearly 4.0 out of 4.0."*

**Expected Observations**:
- Evaluation finishes smoothly in sub-second time.
- Radial gauge displays Low Novelty tier in amber/red.
- Top match is identified as `SYN-0001` with high predicted similarity.

---

### Minute 2:30 – 4:00: Scenario 2 — Model Comparison on Paraphrased Proposals

**Action**:
1. Switch back to the **"1. Proposal Input"** subtab.
2. In the Preset dropdown, select **"Preset: Strong Paraphrase"** (`CASE_C`).
3. Point out the rephrased vocabulary: *"Deep Vision Architecture for Automated Agricultural Pathology Recognition"* over *"foliar leaf collections"*.
4. On the **Scoring Model Architecture** card:
   - First select **"TF-IDF Baseline"** and click **"Evaluate Novelty"**.
   - Note the result.
   - Then select **"Tuned Hybrid (α=0.60)"** and click **"Evaluate Novelty"**.

**What to Say**:
> *"Now let us demonstrate the core strength of our Phase 5 research: handling sophisticated paraphrasing.
> Here we have a proposal that expresses the exact same task as SYN-0001, but the student has deliberately rephrased the vocabulary: calling it 'agricultural pathology' instead of 'crop disease', and 'foliar leaf collection' instead of 'PlantVillage'.
> When we run this through the pure **TF-IDF Baseline**, lexical matching fails to identify the high semantic overlap because none of the n-grams match verbatim. The similarity is low.
> Now, we switch to our **Tuned Hybrid Model (alpha = 0.60)**. In this mode, the system blends 60% TF-IDF with 40% dense semantic embeddings from all-MiniLM-L6-v2.
> When we re-evaluate, the dense embeddings immediately capture the conceptual equivalence between 'agricultural pathology' and 'crop disease', elevating the similarity score and correctly alerting faculty to investigate."*

**Expected Observations**:
- Under TF-IDF, pure lexical score is low.
- Under Tuned Hybrid, semantic embeddings surface topical overlap, and the active badge confirms: `Tuned Hybrid (α=0.60) — 60% Lexical + 40% Semantic`.

---

### Minute 4:00 – 5:15: Scenario 3 — Explainable AI (XAI) Deep-Dive

**Action**:
1. Scroll down to the **Section Cosine Similarity & Model Contributions** section of the report.
2. Point to the section similarity comparison chart.
3. Scroll down and expand the **Top Match Card (#1)**.
4. Point to the **Verified Keyword Evidence** and surrounding context snippets.

**What to Say**:
> *"A critical requirement of our project was that our system cannot be an opaque black box. Faculty must be able to justify their decisions.
> Here, our system provides three levels of explainability:
> First, our **Section Breakdown Chart** shows the similarity for each section. Faculty can immediately see whether the overlap is in the methodology or in the dataset.
> Second, our **Additive Linear Contributions**: because our candidate model has a zero intercept, the total predicted score is the exact sum of section contributions. You can see that the Dataset section contributed 1.95 points—accounting for nearly half the score.
> Third, our **Keyword Evidence**: expanding the match card reveals verbatim n-grams shared between both proposals, complete with word-boundary context snippets quoting the exact sentences from both texts."*

**Expected Observations**:
- Section bar charts show individual section similarities.
- Signed model contributions mathematically sum to the predicted score.
- Informative keyword cards display bidirectional snippets with ellipses.

---

### Minute 5:15 – 6:15: Scenario 4 — Proposal Persistence & Audit History

**Action**:
1. Scroll up to the **"Save Evaluated Proposal to Database"** card.
2. Click the green **"Save Proposal"** button.
3. Observe the confirmation badge: *"Saved as SUB-XXXX"*.
4. Click **"View in History"** (or click the **History** tab in the top navigation).
5. Locate the newly saved proposal in the table.
6. Click the eye icon to open the **Proposal Detail Inspection Modal**.

**What to Say**:
> *"Now we commit the proposal. Notice that evaluating the proposal was completely read-only—it did not pollute the database. 
> When faculty are satisfied with the analysis, they click 'Save Proposal'. 
> The system checks for duplicates using a SHA-256 content hash, saves the proposal and its full XAI report into SQLite, and assigns a unique ID, SUB-0002.
> Navigating to our **Submission History**, we can see a paginated audit log of all evaluated projects. Clicking on any project opens our complete Inspection Modal, allowing examiners to review past reports at any time."*

**Expected Observations**:
- Save proposal executes and returns HTTP 201 Created.
- Submission History table lists the record with novelty percentage badge and top match ID.
- Detail modal displays full proposal text and preserved XAI report.

---

### Minute 6:15 – 7:00: Scenario 5 — Input Validation & Edge Case Handling

**Action**:
1. Return to the **Evaluate** tab.
2. In any required section (e.g., `Problem Statement`), delete all text and type three spaces `"   "`.
3. Click **"Evaluate Novelty"**.
4. Observe the validation rejection.

**What to Say**:
> *"Finally, we demonstrate input robustness. If a student attempts to submit an incomplete proposal or bypass sections with whitespace, our Pydantic validation schema intercepts it immediately, returning an HTTP 422 Unprocessable Entity and highlighting the exact field requiring correction.
> In total, our platform is verified by **114 automated unit and integration tests** with 100% pass rate.
> This concludes our live demonstration. We are ready for questions from the evaluation panel."*

**Expected Observations**:
- Form validation triggers, showing a clean error alert: *"Problem Statement is required"* without crashing the backend.

---

## Emergency Fallback Steps (Troubleshooting Guide)

In the event of an unexpected hardware glitch or network interruption during viva presentation, follow these exact recovery protocols:

### Fallback 1: Backend Process Not Responding
- **Symptom**: Frontend shows *"Could not connect to FastAPI backend"*.
- **Quick Fix**:
  1. Open Terminal 1.
  2. Press `Ctrl + C` to terminate the process.
  3. Re-run:
     ```powershell
     python -m uvicorn src.api.main:app --port 8000
     ```
  4. Wait 3 seconds for lifespan startup, then refresh Chrome (`Ctrl + F5`).

### Fallback 2: Port 8000 or 5173 is In Use
- **Symptom**: `OSError: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`.
- **Quick Fix**:
  ```powershell
  # Free up port 8000
  Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
  # Free up port 5173
  Stop-Process -Id (Get-NetTCPConnection -LocalPort 5173).OwningProcess -Force
  ```
  Then restart both services.

### Fallback 3: SQLite Database Locked
- **Symptom**: `sqlite3.OperationalError: database is locked`.
- **Quick Fix**:
  Close any external SQLite viewer (such as DB Browser for SQLite or VS Code SQLite extension) that may hold an exclusive write lock on `data/proposals.db`.

### Fallback 4: Offline Viva Mode (No Internet Connection)
- **Symptom**: University examination hall has no Wi-Fi.
- **Why It Still Works**: The system is **100% self-contained and offline**. 
  - Sentence-transformers model checkpoint is cached locally.
  - TF-IDF vectorizers are serialized in `models/*.joblib`.
  - Reference corpus is stored locally in `data/`.
  - All communication happens on `localhost` (`127.0.0.1`).
  - No external cloud API or internet connection is required.

