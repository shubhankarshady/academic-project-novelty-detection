# Faculty Annotation Guidelines & Ordinal Rating Rubric

**Academic Project Novelty Detection — Human Verification Protocol (Phase 5.2)**  
**Document Version**: 2.0.0 (Updated for 120-Pair Dual-Batch Study)  
**Target Artifact**: `data/annotation/faculty_evaluation_sample_120.xlsx`  
**Programmatic Dataset**: `data/annotation/faculty_evaluation_sample_120.csv`  
**Research Master Dataset**: `data/annotation/faculty_evaluation_sample_120_master.csv`  
**Evaluation Scope**: Double-Blind Pairwise Academic Project Similarity Evaluation  

---

## 1. Objective and Context

The goal of this human verification study is to establish an empirical, expert ground truth for academic project similarity. Two independent faculty reviewers evaluate pairwise comparisons between a **Candidate Proposal** and a **Reference Project** across eight canonical project sections:

1. **Title**
2. **Abstract**
3. **Problem Statement**
4. **Objectives**
5. **Proposed Methodology**
6. **Dataset & Benchmark Resources**
7. **Algorithm & Architecture**
8. **Technology Tools & Frameworks**

Reviewers assign an ordinal score from **0 to 4** based on the standardized rubric below.

> [!IMPORTANT]
> **What This Study Evaluates**: This annotation evaluates **research content overlap and conceptual similarity**.  
> **What This Study Does NOT Evaluate**: This study is **not a plagiarism investigation** and does not make accusations of academic dishonesty, copyright infringement, or intellectual property violation. Ratings must reflect academic and technical resemblance alone.

---

## 2. Standardized Ordinal Rating Rubric (0 – 4 Scale)

Reviewers must select an integer score $\in \{0, 1, 2, 3, 4\}$ according to the following operational criteria:

| Rating | Category | Operational Definition | Primary Indicators | Actionable Faculty Guidance |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Not Similar** | Completely distinct research initiatives with no meaningful problem, methodology, or topical overlap. | • Unrelated application domains.<br>• Divergent research objectives.<br>• Disjoint methodologies and data sources. | Assign **0** if proposals only share generic academic filler (e.g., *"analysis"*, *"machine learning"*, *"evaluation"*, *"system"*). |
| **1** | **Slightly Similar** | Broad topical resemblance or common technical stack without direct research overlap. | • Same broad field (e.g., Healthcare AI), but entirely different problems.<br>• Shared standard toolkits (e.g., PyTorch, React, OpenCV) with distinct objectives. | Assign **1** when technical tools overlap but the actual problem, methodology, and datasets are independent. |
| **2** | **Moderately Similar** | Partial research overlap: shares a major core component (e.g., identical dataset or algorithmic architecture), but applies it to a distinct objective or problem formulation. | • Identical dataset (e.g., *PlantVillage*) used for generative inpainting vs. disease classification.<br>• Same core algorithm (e.g., YOLO) applied to a different domain (e.g., underwater vs. traffic). | Assign **2** when one major section matches closely but problem formulation and objectives diverge significantly. |
| **3** | **Highly Similar** | Substantial conceptual, problem, and methodological convergence. Distinct wording or minor technical variations, but essentially the same research contribution. | • Paraphrased formulation of the same problem.<br>• Similar architecture and evaluation protocol.<br>• Shared dataset and matching objective goals. | Assign **3** when another research group would reasonably consider the proposal a duplicate contribution despite different phrasing. |
| **4** | **Nearly Duplicate / Duplicate** | Verbatim or substantially verbatim reproduction of an existing project across all primary sections. | • Verbatim or near-identical text across abstract, problem, and methodology.<br>• Same dataset, algorithm, tools, and objectives.<br>• Negligible or cosmetic token edits. | Assign **4** if the candidate proposal is a direct replica of the reference project with cosmetic or superficial changes. |

---

## 3. Disentangling Lexical Overlap from Conceptual Similarity

A fundamental limitation of automated similarity systems is confusing **lexical (word-level) overlap** with **conceptual (semantic) similarity**. Reviewers must adhere to the following principles:

### A. High Lexical Overlap with Low Conceptual Similarity $\implies$ Rating 0 or 1
- **Scenario**: Two proposals share substantial vocabulary (e.g., *"convolutional neural network"*, *"accuracy"*, *"loss function"*, *"data preprocessing"*, *"evaluation metrics"*), but Proposal A detects brain lesions on MRI scans while Proposal B classifies crop diseases on drone images.
- **Verdict**: Mark **0** (or **1** if in related imaging contexts). Do not inflate the similarity rating because of standard technical jargon.

### B. Low Lexical Overlap with High Conceptual Similarity $\implies$ Rating 3 or 4
- **Scenario**: Proposal A proposes *"lightweight residual convolutional networks for detecting foliar pathogens on agricultural crop imagery using PlantVillage"*, while Proposal B proposes *"an efficient deep residual vision model for identifying plant leaf blights across botanical leaf benchmarks"*.
- **Verdict**: Mark **3** or **4**. Although different words and synonyms are used, the underlying research problem, methodology, benchmark data, and technical objectives are identical.

### C. Shared Benchmark Datasets with Divergent Objectives $\implies$ Rating 2
- **Scenario**: Both proposals use the *BraTS* MRI dataset. Proposal A performs 3D brain tumor boundary segmentation; Proposal B performs synthetic brain MRI image generation via Diffusion Models for privacy preservation.
- **Verdict**: Mark **2**. While the data resource is identical, the research problems and algorithmic frameworks are distinct.

---

## 4. Handling Edge Cases & Insufficient Information

1. **Short, Vague, or Under-Specified Sections**:
   - If a proposal provides very brief sections (e.g., *"Used CNN on custom dataset"*), evaluate similarity based on what is verifiable. Do not assume unstated similarities.
   - If insufficient information prevents determining whether the methodology is identical, default to **1** (slight) or **2** (moderate) depending on domain overlap, and document the ambiguity in your comments column.
2. **Generic Academic Boilerplate**:
   - Phrases such as *"The proposed system aims to provide an efficient and scalable solution with high accuracy"* contain zero research-specific information. Disregard generic filler text entirely during scoring.
3. **Cross-Domain Technology Transfer**:
   - When a standard model (e.g., Graph Neural Networks) is transferred from social network analysis to chemical molecular property prediction, the rating should typically be **1** or **2**, reflecting methodology reuse in a completely distinct domain.

---

## 5. Reviewer Workflow & Independent Assessment Protocol

### Step 1: Independent Double-Blind Review
- Two independent faculty reviewers (Reviewer 1 and Reviewer 2) receive the annotation workbook:  
  `data/annotation/faculty_evaluation_sample_120.xlsx`.
- The evaluation pairs are organized into two worksheets of 60 pairs each:
  - **`Evaluation_Batch_1`**: Pairs `EVAL-001` through `EVAL-060`.
  - **`Evaluation_Batch_2`**: Pairs `EVAL-061` through `EVAL-120`.
- Both reviewers must complete their evaluations **independently** without consulting each other or examining model prediction scores.
- Reviewers record their integer ratings (0, 1, 2, 3, or 4) in their designated column:
  - Reviewer 1 records in `reviewer_1_rating` and adds reasoning in `reviewer_1_comments`.
  - Reviewer 2 records in `reviewer_2_rating` and adds reasoning in `reviewer_2_comments`.
- Reviewers are completely blind to synthetic labels, model scores, or expected categories to prevent bias.

### Step 2: Conflict Resolution and Adjudication Procedure
Once both reviewers have completed all pairs in a batch, the adjudication process resolves disagreements:

1. **Exact Agreement ($R_1 = R_2$)**:
   - The agreed score is automatically adopted as the `adjudicated_rating`.
2. **Minor Disagreement ($|R_1 - R_2| = 1$)**:
   - *Non-boundary cases* (e.g., $R_1=0, R_2=1$ or $R_1=1, R_2=2$): The reviewers conduct a brief discussion to reach consensus. If consensus is not reached, the average $\frac{R_1 + R_2}{2}$ is recorded, or an appointed senior adjudicator decides.
   - *Threshold-boundary cases* (e.g., $R_1=2, R_2=3$ spanning Moderate vs. High similarity, or $R_1=3, R_2=4$ spanning Substantial vs. Duplicate): A formal discussion examines whether the proposal constitutes an actionable duplicate.
3. **Major Disagreement ($|R_1 - R_2| \ge 2$)**:
   - For example, $R_1=1$ (Slight) and $R_2=3$ (High).
   - Both reviewers re-read the pair side by side.
   - An independent third faculty member (Lead Adjudicator) reviews the pair, enters the consensus score into `adjudicated_rating`, and documents the resolution in `adjudication_comments`.
4. **Finalization**:
   - Upon completion of all rows in a batch, `annotation_status` is updated from `"PENDING_HUMAN_REVIEW"` to `"ADJUDICATED_FINAL"`.

---

## 6. Structure of the Annotation Workbook

The workbook `data/annotation/faculty_evaluation_sample_120.xlsx` contains four sheets:

1. **`Evaluation_Batch_1`** (60 Pairs, `EVAL-001` to `EVAL-060`):
   - **Identification Metadata**: `pair_id`, `project_candidate_id`, `project_reference_id`, `domain_candidate`, `domain_reference`.
   - **Side-by-Side Candidate & Reference Sections**:
     - `candidate_title` | `reference_title`
     - `candidate_abstract` | `reference_abstract`
     - `candidate_problem` | `reference_problem`
     - `candidate_objectives` | `reference_objectives`
     - `candidate_methodology` | `reference_methodology`
     - `candidate_dataset` | `reference_dataset`
     - `candidate_algorithm` | `reference_algorithm`
     - `candidate_technology` | `reference_technology`
   - **Reviewer 1 Section (Light Yellow)**: `reviewer_1_rating`, `reviewer_1_comments`.
   - **Reviewer 2 Section (Light Yellow)**: `reviewer_2_rating`, `reviewer_2_comments`.
   - **Adjudication Section (Light Green)**: `adjudicated_rating`, `adjudication_comments`.
   - **Status (Light Orange)**: `annotation_status` (`PENDING_HUMAN_REVIEW`).
2. **`Evaluation_Batch_2`** (60 Pairs, `EVAL-061` to `EVAL-120`):
   - Identical 28-column structure and formatting as Batch 1, containing the remaining 60 unique pairs.
3. **`Rating_Rubric`**:
   - Complete reference table embedded directly within the Excel file for convenient lookup during annotation.
4. **`Instructions`**:
   - Operational guide detailing the 6-step review procedure, visual color-coding legend, lexical vs. conceptual guidelines, and batch organization.

---

## 7. Submission Checklist for Reviewers

- [ ] All 60 rows in the assigned evaluation batch have an integer rating $\in \{0, 1, 2, 3, 4\}$.
- [ ] No rating cells are left blank in your assigned reviewer column.
- [ ] Any score of 2, 3, or 4 has at least a brief comment explaining the overlapping section(s).
- [ ] Ratings were assigned independently without consulting the other reviewer or external automated systems.
- [ ] The completed workbook is returned to the project repository coordinator for adjudication.
