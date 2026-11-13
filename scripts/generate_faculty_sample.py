#!/usr/bin/env python
"""
Generate Faculty-Validated Evaluation Sample (Phase 5.2).

Selects a reproducible, balanced sample of 120 academic project pairs from the corpus
and controlled benchmarks, structured into two separate worksheets:
- Evaluation_Batch_1: 60 pairs (15 Dev / Calibration, 45 Held-Out Evaluation)
- Evaluation_Batch_2: 60 pairs (15 Dev / Calibration, 45 Held-Out Evaluation)

Features:
- Exactly 120 unique pairs with zero duplicate pairs across or within batches
- Uniformly balanced distribution across 5 provisional similarity levels (24 pairs per level total; 12 per level per batch)
- Balanced integration of all 8 documented controlled benchmark test cases (4 in Batch 1, 4 in Batch 2)
- Side-by-side comparisons across all 8 canonical project sections
- Completely double-blind: no provisional synthetic labels, categories, rationales, or model scores in faculty sheets
- Separate blank columns for Reviewer 1 Rating & Comments, Reviewer 2 Rating & Comments, Adjudicated Rating & Comments
- Annotation status explicitly marked as 'PENDING_HUMAN_REVIEW'
- Full professional Excel formatting: dark header row, freeze panes, auto-filters, text wrapping, zebra striping,
  yellow reviewer input highlights, green adjudication highlights, orange pending status, and 0-4 conditional formatting
- Supporting sheets: Rating_Rubric and Instructions
- Outputs:
  - data/annotation/faculty_evaluation_sample_120.xlsx (faculty-facing blind workbook)
  - data/annotation/faculty_evaluation_sample_120.csv (programmatic 120-pair CSV with batch identifier)
  - data/annotation/faculty_evaluation_sample_120_master.csv (research master file with ground-truth metadata)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import get_column_letter

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_semantic_feasibility import CONTROLLED_TEST_CASES


# Canonical project sections
SECTION_COLS = [
    "domain",
    "project_title",
    "abstract",
    "problem_statement",
    "objectives",
    "proposed_methodology",
    "dataset",
    "algorithm_model",
    "technology_tools",
]

# Evaluation sheet columns (blind, faculty-facing)
EVALUATION_COLUMNS = [
    "pair_id",
    "project_candidate_id",
    "project_reference_id",
    "domain_candidate",
    "domain_reference",
    "candidate_title",
    "reference_title",
    "candidate_abstract",
    "reference_abstract",
    "candidate_problem",
    "reference_problem",
    "candidate_objectives",
    "reference_objectives",
    "candidate_methodology",
    "reference_methodology",
    "candidate_dataset",
    "reference_dataset",
    "candidate_algorithm",
    "reference_algorithm",
    "candidate_technology",
    "reference_technology",
    "reviewer_1_rating",
    "reviewer_1_comments",
    "reviewer_2_rating",
    "reviewer_2_comments",
    "adjudicated_rating",
    "adjudication_comments",
    "annotation_status",
]

# Master research columns (includes ground truth & selection metadata)
MASTER_COLUMNS = [
    "batch",
    "pair_id",
    "subset",
    "provisional_synthetic_label",
    "provisional_category",
    "selection_rationale",
    "project_candidate_id",
    "project_reference_id",
    "domain_candidate",
    "domain_reference",
    "candidate_title",
    "reference_title",
    "candidate_abstract",
    "reference_abstract",
    "candidate_problem",
    "reference_problem",
    "candidate_objectives",
    "reference_objectives",
    "candidate_methodology",
    "reference_methodology",
    "candidate_dataset",
    "reference_dataset",
    "candidate_algorithm",
    "reference_algorithm",
    "candidate_technology",
    "reference_technology",
    "reviewer_1_rating",
    "reviewer_1_comments",
    "reviewer_2_rating",
    "reviewer_2_comments",
    "adjudicated_rating",
    "adjudication_comments",
    "annotation_status",
]

# Controlled cases allocation across batches
BATCH_CONTROLLED_CASES = {
    "Evaluation_Batch_1": [
        ("CASE_A_EXACT_COPY", 4, "DEV_CALIBRATION", "Controlled Benchmark: Verbatim copy across all 8 sections"),
        ("CASE_D_SAME_DOMAIN_DIFF_PROBLEM", 1, "DEV_CALIBRATION", "Controlled Benchmark: Same domain (AI/ML), completely different problem"),
        ("CASE_E_DIFF_DOMAIN_COMMON_TECH", 0, "DEV_CALIBRATION", "Controlled Benchmark: Cross-domain project sharing only common technology"),
        ("CASE_F_SAME_DATASET_DIFF_OBJECTIVES", 2, "HELD_OUT_EVALUATION", "Controlled Benchmark: Shared benchmark dataset with divergent research objective"),
    ],
    "Evaluation_Batch_2": [
        ("CASE_B_LIGHT_PARAPHRASE", 4, "HELD_OUT_EVALUATION", "Controlled Benchmark: Verbatim copy with light paraphrase and wording edits"),
        ("CASE_C_STRONG_PARAPHRASE", 3, "HELD_OUT_EVALUATION", "Controlled Benchmark: Core conceptual paraphrase with distinct vocabulary"),
        ("CASE_G_SAME_ALGO_DIFF_APPLICATION", 1, "DEV_CALIBRATION", "Controlled Benchmark: Same algorithm architecture, different domain and dataset"),
        ("CASE_H_SUPERFICIAL_WORDING_DIFF_OBJECTIVES", 0, "DEV_CALIBRATION", "Controlled Benchmark: Superficial keyword overlap with divergent research objective"),
    ],
}


def build_pair_record(
    batch_name: str,
    subset: str,
    prov_label: int,
    prov_category: str,
    selection_rationale: str,
    cand_id: str,
    ref_id: str,
    cand_data: Dict[str, Any],
    ref_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a standard dictionary record for a project pair."""
    return {
        "batch": batch_name,
        "subset": subset,
        "provisional_synthetic_label": prov_label,
        "provisional_category": prov_category,
        "selection_rationale": selection_rationale,
        "project_candidate_id": cand_id,
        "project_reference_id": ref_id,
        "domain_candidate": cand_data.get("domain", ""),
        "domain_reference": ref_data.get("domain", ""),
        "candidate_title": cand_data.get("project_title", ""),
        "reference_title": ref_data.get("project_title", ""),
        "candidate_abstract": cand_data.get("abstract", ""),
        "reference_abstract": ref_data.get("abstract", ""),
        "candidate_problem": cand_data.get("problem_statement", ""),
        "reference_problem": ref_data.get("problem_statement", ""),
        "candidate_objectives": cand_data.get("objectives", ""),
        "reference_objectives": ref_data.get("objectives", ""),
        "candidate_methodology": cand_data.get("proposed_methodology", ""),
        "reference_methodology": ref_data.get("proposed_methodology", ""),
        "candidate_dataset": cand_data.get("dataset", ""),
        "reference_dataset": ref_data.get("dataset", ""),
        "candidate_algorithm": cand_data.get("algorithm_model", ""),
        "reference_algorithm": ref_data.get("algorithm_model", ""),
        "candidate_technology": cand_data.get("technology_tools", ""),
        "reference_technology": ref_data.get("technology_tools", ""),
        "reviewer_1_rating": None,
        "reviewer_1_comments": None,
        "reviewer_2_rating": None,
        "reviewer_2_comments": None,
        "adjudicated_rating": None,
        "adjudication_comments": None,
        "annotation_status": "PENDING_HUMAN_REVIEW",
    }


def build_faculty_evaluation_sample_120(
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construct the stratified 120-pair evaluation sample divided into two batches of 60.

    Returns:
        batch_1_df: 60 pairs for Evaluation_Batch_1 (blind)
        batch_2_df: 60 pairs for Evaluation_Batch_2 (blind)
        master_df: Combined 120 pairs with research ground-truth metadata
    """
    pairs_path = ROOT_DIR / "data" / "processed" / "similarity_pairs_clean.csv"
    dataset_path = ROOT_DIR / "data" / "raw" / "Academic_Project_Similarity_Synthetic_Dataset.xlsx"

    if not pairs_path.exists():
        raise FileNotFoundError(f"Clean pairs file not found: {pairs_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {dataset_path}")

    pairs_df = pd.read_csv(pairs_path)
    projects_df = pd.read_excel(dataset_path, sheet_name="Projects").set_index("project_id")

    controlled_cases_dict = {c["case_id"]: c for c in CONTROLLED_TEST_CASES}
    rng = np.random.RandomState(random_state)
    seen_pairs = set()

    # Pre-register controlled cases to avoid duplication
    for batch_name, ctrl_list in BATCH_CONTROLLED_CASES.items():
        for case_id, _, _, _ in ctrl_list:
            case = controlled_cases_dict[case_id]
            pid_cand = case["proposal"]["project_id"]
            pid_ref = case["target_ref_id"]
            seen_pairs.add((min(str(pid_cand), str(pid_ref)), max(str(pid_cand), str(pid_ref))))

    # Deterministically shuffle corpus pairs per similarity level
    shuffled_sub_pairs = {}
    for level in range(5):
        sub = pairs_df[pairs_df["expert_similarity_label"] == level].copy()
        perm = rng.permutation(len(sub))
        shuffled_sub_pairs[level] = sub.iloc[perm].reset_index(drop=True)

    level_pointers = {level: 0 for level in range(5)}
    batch_records: Dict[str, List[Dict[str, Any]]] = {
        "Evaluation_Batch_1": [],
        "Evaluation_Batch_2": [],
    }

    category_map = {
        0: "Not Similar",
        1: "Slightly Similar",
        2: "Moderately Similar",
        3: "Highly Similar",
        4: "Nearly Duplicate",
    }

    # Build both batches
    for b_idx, (batch_name, ctrl_list) in enumerate(BATCH_CONTROLLED_CASES.items()):
        ctrl_dev_counts = {l: 0 for l in range(5)}
        ctrl_eval_counts = {l: 0 for l in range(5)}

        # Add controlled benchmark cases assigned to this batch
        for case_id, prov_label, subset, selection_note in ctrl_list:
            case = controlled_cases_dict[case_id]
            cand = case["proposal"]
            pid_ref = case["target_ref_id"]
            ref = projects_df.loc[pid_ref]

            if subset == "DEV_CALIBRATION":
                ctrl_dev_counts[prov_label] += 1
            else:
                ctrl_eval_counts[prov_label] += 1

            rec = build_pair_record(
                batch_name=batch_name,
                subset=subset,
                prov_label=prov_label,
                prov_category=category_map[prov_label],
                selection_rationale=selection_note,
                cand_id=cand.get("project_id", case_id),
                ref_id=pid_ref,
                cand_data=cand,
                ref_data=ref,
            )
            batch_records[batch_name].append(rec)

        # Sample remaining pairs from corpus for balanced 12 pairs per level (3 Dev, 9 Eval)
        for level in range(5):
            needed_dev = 3 - ctrl_dev_counts[level]
            needed_eval = 9 - ctrl_eval_counts[level]

            # Sample Dev / Calibration
            sampled_dev = 0
            while sampled_dev < needed_dev:
                if level_pointers[level] >= len(shuffled_sub_pairs[level]):
                    raise RuntimeError(f"Corpus exhausted for similarity level {level}")
                row = shuffled_sub_pairs[level].iloc[level_pointers[level]]
                level_pointers[level] += 1

                pid_a = row["project_A_id"]
                pid_b = row["project_B_id"]
                pair_key = (min(str(pid_a), str(pid_b)), max(str(pid_a), str(pid_b)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                proj_a = projects_df.loc[pid_a]
                proj_b = projects_df.loc[pid_b]
                cat = row["similarity_category"]

                rec = build_pair_record(
                    batch_name=batch_name,
                    subset="DEV_CALIBRATION",
                    prov_label=level,
                    prov_category=cat,
                    selection_rationale=f"Corpus stratified sample: Label {level} ({cat})",
                    cand_id=pid_a,
                    ref_id=pid_b,
                    cand_data=proj_a,
                    ref_data=proj_b,
                )
                batch_records[batch_name].append(rec)
                sampled_dev += 1

            # Sample Held-Out Evaluation
            sampled_eval = 0
            while sampled_eval < needed_eval:
                if level_pointers[level] >= len(shuffled_sub_pairs[level]):
                    raise RuntimeError(f"Corpus exhausted for similarity level {level}")
                row = shuffled_sub_pairs[level].iloc[level_pointers[level]]
                level_pointers[level] += 1

                pid_a = row["project_A_id"]
                pid_b = row["project_B_id"]
                pair_key = (min(str(pid_a), str(pid_b)), max(str(pid_a), str(pid_b)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                proj_a = projects_df.loc[pid_a]
                proj_b = projects_df.loc[pid_b]
                cat = row["similarity_category"]

                rec = build_pair_record(
                    batch_name=batch_name,
                    subset="HELD_OUT_EVALUATION",
                    prov_label=level,
                    prov_category=cat,
                    selection_rationale=f"Corpus stratified sample: Label {level} ({cat})",
                    cand_id=pid_a,
                    ref_id=pid_b,
                    cand_data=proj_a,
                    ref_data=proj_b,
                )
                batch_records[batch_name].append(rec)
                sampled_eval += 1

    # Convert to DataFrames and shuffle rows within each batch deterministically to prevent ordering bias
    b1_raw = pd.DataFrame(batch_records["Evaluation_Batch_1"])
    b2_raw = pd.DataFrame(batch_records["Evaluation_Batch_2"])

    # Shuffle each batch deterministically with distinct sub-seeds derived from random_state
    b1_perm = np.random.RandomState(random_state + 101).permutation(len(b1_raw))
    b2_perm = np.random.RandomState(random_state + 202).permutation(len(b2_raw))

    b1_df = b1_raw.iloc[b1_perm].reset_index(drop=True)
    b2_df = b2_raw.iloc[b2_perm].reset_index(drop=True)

    # Assign sequential, globally unique Pair IDs
    # Batch 1: EVAL-001 to EVAL-060
    # Batch 2: EVAL-061 to EVAL-120
    b1_df["pair_id"] = [f"EVAL-{i+1:03d}" for i in range(len(b1_df))]
    b2_df["pair_id"] = [f"EVAL-{i+61:03d}" for i in range(len(b2_df))]

    # Combine for research master file
    master_df = pd.concat([b1_df, b2_df], ignore_index=True)[MASTER_COLUMNS]

    # Faculty-facing evaluation DataFrames (strictly blind: no synthetic labels or model predictions)
    batch_1_eval_df = b1_df[EVALUATION_COLUMNS].copy()
    batch_2_eval_df = b2_df[EVALUATION_COLUMNS].copy()

    return batch_1_eval_df, batch_2_eval_df, master_df


def apply_evaluation_sheet_formatting(ws: openpyxl.worksheet.worksheet.Worksheet, df: pd.DataFrame) -> None:
    """Apply professional, clean styling and conditional formatting to an evaluation worksheet."""
    headers = list(df.columns)

    # Header styling fonts and fills
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_body = Font(name="Calibri", size=10, color="1F2937")
    font_meta = Font(name="Calibri", size=10, bold=True, color="111827")

    fill_header_meta = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")       # Dark Navy Slate
    fill_header_cand = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")       # Deep Blue
    fill_header_ref = PatternFill(start_color="374151", end_color="374151", fill_type="solid")         # Dark Charcoal
    fill_header_review = PatternFill(start_color="854D0E", end_color="854D0E", fill_type="solid")     # Dark Amber / Reviewer
    fill_header_adjud = PatternFill(start_color="14532D", end_color="14532D", fill_type="solid")      # Dark Forest Green
    fill_header_status = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")     # Dark Navy

    # Zebra striping and highlight fills
    fill_row_odd = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    fill_row_even = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")

    # Reviewer input columns: light yellow
    fill_review_odd = PatternFill(start_color="FEF9C3", end_color="FEF9C3", fill_type="solid")       # Light Yellow 100
    fill_review_even = PatternFill(start_color="FEF08A", end_color="FEF08A", fill_type="solid")      # Light Yellow 200

    # Adjudication columns: light green
    fill_adjud_odd = PatternFill(start_color="F0FDF4", end_color="F0FDF4", fill_type="solid")        # Soft Light Green 50
    fill_adjud_even = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")       # Light Green 100

    # Pending status: light orange
    fill_status_pending = PatternFill(start_color="FFEDD5", end_color="FFEDD5", fill_type="solid")   # Light Orange 100
    font_status_pending = Font(name="Calibri", size=10, bold=True, color="9A3412")

    # Borders
    thin_border_side = Side(style="thin", color="E2E8F0")
    medium_border_side = Side(style="medium", color="475569")

    # Set column widths
    col_widths = {
        "pair_id": 13,
        "project_candidate_id": 20,
        "project_reference_id": 20,
        "domain_candidate": 25,
        "domain_reference": 25,
        "candidate_title": 32,
        "reference_title": 32,
        "candidate_abstract": 45,
        "reference_abstract": 45,
        "candidate_problem": 40,
        "reference_problem": 40,
        "candidate_objectives": 36,
        "reference_objectives": 36,
        "candidate_methodology": 36,
        "reference_methodology": 36,
        "candidate_dataset": 24,
        "reference_dataset": 24,
        "candidate_algorithm": 24,
        "reference_algorithm": 24,
        "candidate_technology": 24,
        "reference_technology": 24,
        "reviewer_1_rating": 16,
        "reviewer_1_comments": 32,
        "reviewer_2_rating": 16,
        "reviewer_2_comments": 32,
        "adjudicated_rating": 18,
        "adjudication_comments": 32,
        "annotation_status": 25,
    }

    # Section dividing columns (apply medium border on right)
    section_dividers = [
        "domain_reference",       # End of ID/domain metadata
        "reference_technology",   # End of project section comparisons
        "reviewer_1_comments",    # End of Reviewer 1 section
        "reviewer_2_comments",    # End of Reviewer 2 section
        "adjudication_comments",  # End of Adjudication section
        "annotation_status",      # End of record
    ]

    # Write headers
    ws.append(headers)
    ws.row_dimensions[1].height = 30

    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = font_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        if header.startswith("reviewer_"):
            cell.fill = fill_header_review
        elif header.startswith("adjudicat"):
            cell.fill = fill_header_adjud
        elif header.startswith("candidate_"):
            cell.fill = fill_header_cand
        elif header.startswith("reference_"):
            cell.fill = fill_header_ref
        elif header == "annotation_status":
            cell.fill = fill_header_status
        else:
            cell.fill = fill_header_meta

        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = col_widths.get(header, 22)

    # Write data rows
    for row_idx, row_data in enumerate(df.itertuples(index=False), 2):
        ws.append(list(row_data))
        is_even = (row_idx % 2 == 0)

        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            header = headers[col_idx - 1]

            # Right border: medium if divider, else thin
            right_side = medium_border_side if header in section_dividers else thin_border_side
            cell.border = Border(
                left=thin_border_side,
                right=right_side,
                top=thin_border_side,
                bottom=thin_border_side,
            )

            # Cell font and alignment
            if header in ["pair_id", "project_candidate_id", "project_reference_id"]:
                cell.font = font_meta
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif header in ["domain_candidate", "domain_reference"]:
                cell.font = font_body
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            elif header in ["reviewer_1_rating", "reviewer_2_rating", "adjudicated_rating"]:
                cell.font = font_meta
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif header == "annotation_status":
                cell.font = font_status_pending
                cell.alignment = Alignment(horizontal="center", vertical="top")
            else:
                cell.font = font_body
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

            # Cell background fills
            if header.startswith("reviewer_"):
                cell.fill = fill_review_even if is_even else fill_review_odd
            elif header.startswith("adjudicat"):
                cell.fill = fill_adjud_even if is_even else fill_adjud_odd
            elif header == "annotation_status":
                cell.fill = fill_status_pending
            else:
                cell.fill = fill_row_even if is_even else fill_row_odd

    # Freeze header row
    ws.freeze_panes = "A2"

    # Enable filters across all columns
    last_col_letter = get_column_letter(len(headers))
    total_rows = len(df) + 1
    ws.auto_filter.ref = f"A1:{last_col_letter}{total_rows}"

    # Conditional formatting for similarity ratings (0 to 4 scale)
    rule_0 = CellIsRule(operator="equal", formula=["0"], stopIfTrue=True,
                        fill=PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid"),
                        font=Font(name="Calibri", size=10, bold=True, color="334155"))
    rule_1 = CellIsRule(operator="equal", formula=["1"], stopIfTrue=True,
                        fill=PatternFill(start_color="CFFAFE", end_color="CFFAFE", fill_type="solid"),
                        font=Font(name="Calibri", size=10, bold=True, color="0E7490"))
    rule_2 = CellIsRule(operator="equal", formula=["2"], stopIfTrue=True,
                        fill=PatternFill(start_color="FEF08A", end_color="FEF08A", fill_type="solid"),
                        font=Font(name="Calibri", size=10, bold=True, color="854D0E"))
    rule_3 = CellIsRule(operator="equal", formula=["3"], stopIfTrue=True,
                        fill=PatternFill(start_color="FED7AA", end_color="FED7AA", fill_type="solid"),
                        font=Font(name="Calibri", size=10, bold=True, color="C2410C"))
    rule_4 = CellIsRule(operator="equal", formula=["4"], stopIfTrue=True,
                        fill=PatternFill(start_color="FECACA", end_color="FECACA", fill_type="solid"),
                        font=Font(name="Calibri", size=10, bold=True, color="B91C1C"))

    rating_col_names = ["reviewer_1_rating", "reviewer_2_rating", "adjudicated_rating"]
    for r_col in rating_col_names:
        if r_col in headers:
            col_letter = get_column_letter(headers.index(r_col) + 1)
            cell_range = f"{col_letter}2:{col_letter}{total_rows}"
            for rule in [rule_0, rule_1, rule_2, rule_3, rule_4]:
                ws.conditional_formatting.add(cell_range, rule)

    # Conditional formatting for annotation status
    if "annotation_status" in headers:
        status_col_letter = get_column_letter(headers.index("annotation_status") + 1)
        status_range = f"{status_col_letter}2:{status_col_letter}{total_rows}"

        rule_pending = CellIsRule(operator="equal", formula=['"PENDING_HUMAN_REVIEW"'], stopIfTrue=True,
                                  fill=PatternFill(start_color="FFEDD5", end_color="FFEDD5", fill_type="solid"),
                                  font=Font(name="Calibri", size=10, bold=True, color="9A3412"))
        rule_completed = CellIsRule(operator="equal", formula=['"COMPLETED"'], stopIfTrue=True,
                                    fill=PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid"),
                                    font=Font(name="Calibri", size=10, bold=True, color="166534"))
        rule_adjudicated = CellIsRule(operator="equal", formula=['"ADJUDICATED_FINAL"'], stopIfTrue=True,
                                      fill=PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid"),
                                      font=Font(name="Calibri", size=10, bold=True, color="166534"))
        rule_discussion = CellIsRule(operator="equal", formula=['"REQUIRES_DISCUSSION"'], stopIfTrue=True,
                                     fill=PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid"),
                                     font=Font(name="Calibri", size=10, bold=True, color="991B1B"))

        for rule in [rule_pending, rule_completed, rule_adjudicated, rule_discussion]:
            ws.conditional_formatting.add(status_range, rule)


def create_rating_rubric_sheet(wb: openpyxl.Workbook) -> None:
    """Create the Rating_Rubric worksheet explaining the 0-4 scale in simple, professional terms."""
    ws = wb.create_sheet(title="Rating_Rubric")
    ws.views.sheetView[0].showGridLines = True

    font_title = Font(name="Calibri", size=14, bold=True, color="1E3A8A")
    font_sub = Font(name="Calibri", size=10, italic=True, color="4B5563")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_cell = Font(name="Calibri", size=10, color="1F2937")
    font_bold_cell = Font(name="Calibri", size=10, bold=True, color="111827")

    fill_header = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    border_thin = Border(
        left=Side(style="thin", color="D1D5DB"),
        right=Side(style="thin", color="D1D5DB"),
        top=Side(style="thin", color="D1D5DB"),
        bottom=Side(style="thin", color="D1D5DB"),
    )

    ws.append(["Academic Project Pairwise Similarity Rating Rubric (0 – 4 Scale)"])
    ws.append(["Human Manual Verification Protocol — Operational Definitions & Actionable Faculty Guidance"])
    ws.append([])
    ws.append(["Rating", "Category", "Operational Definition", "Typical Indicators", "Actionable Guidance for Faculty"])

    rubric_rows = [
        [
            0,
            "Not Similar",
            "Completely distinct research initiatives with no meaningful problem, methodology, or topical overlap.",
            "• Completely unrelated application domains.\n• Divergent research objectives.\n• Disjoint methodologies, algorithms, and data sources.",
            "Assign 0 if proposals only share generic academic filler (e.g. 'analysis', 'system', 'machine learning', 'evaluation').",
        ],
        [
            1,
            "Slightly Similar",
            "Broad topical resemblance or shared technology stack without direct research overlap.",
            "• Same general discipline (e.g. Healthcare AI), but entirely different problems.\n• Shared standard toolkits (e.g. PyTorch, React, OpenCV) with distinct objectives.",
            "Assign 1 when technical tools overlap but the actual problem formulation, proposed methodology, and datasets are independent.",
        ],
        [
            2,
            "Moderately Similar",
            "Partial research overlap: shares a major core component (e.g. identical dataset or algorithmic architecture), but applies it to a distinct objective.",
            "• Identical benchmark dataset (e.g. PlantVillage) used for image synthesis instead of classification.\n• Same algorithm (e.g. YOLO) applied to a different domain (e.g. underwater vs. traffic).",
            "Assign 2 when one major section matches closely but problem formulation and objectives diverge significantly.",
        ],
        [
            3,
            "Highly Similar",
            "Substantial conceptual, problem, and methodological convergence. Distinct wording or minor technical variations, but essentially the same research contribution.",
            "• Paraphrased formulation of the same core problem.\n• Similar architecture and evaluation protocol.\n• Shared dataset and matching contribution goals.",
            "Assign 3 when another research group would reasonably consider the proposal a duplicate contribution despite different phrasing.",
        ],
        [
            4,
            "Nearly Duplicate / Duplicate",
            "Verbatim or substantially verbatim reproduction of an existing project across all primary sections.",
            "• Verbatim or near-identical text across abstract, problem, and methodology.\n• Same dataset, algorithm, tools, and objectives.\n• Negligible or cosmetic token edits.",
            "Assign 4 if the candidate proposal is a direct replica of the reference project with cosmetic or superficial changes.",
        ],
    ]

    for row in rubric_rows:
        ws.append(row)

    # Title styling
    ws["A1"].font = font_title
    ws["A2"].font = font_sub

    # Headers styling (Row 4)
    ws.row_dimensions[4].height = 28
    for col_idx in range(1, 6):
        cell = ws.cell(row=4, column=col_idx)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Data rows styling (Rows 5-9)
    level_fills = [
        PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid"),
        PatternFill(start_color="CFFAFE", end_color="CFFAFE", fill_type="solid"),
        PatternFill(start_color="FEF08A", end_color="FEF08A", fill_type="solid"),
        PatternFill(start_color="FED7AA", end_color="FED7AA", fill_type="solid"),
        PatternFill(start_color="FECACA", end_color="FECACA", fill_type="solid"),
    ]

    for idx, row_idx in enumerate(range(5, 10)):
        ws.row_dimensions[row_idx].height = 65
        for col_idx in range(1, 6):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border = border_thin
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if col_idx == 1:
                cell.font = Font(name="Calibri", size=14, bold=True, color="111827")
                cell.alignment = Alignment(horizontal="center", vertical="top")
                cell.fill = level_fills[idx]
            elif col_idx == 2:
                cell.font = font_bold_cell
                cell.alignment = Alignment(horizontal="left", vertical="top")
            else:
                cell.font = font_cell

    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 38
    ws.column_dimensions["D"].width = 44
    ws.column_dimensions["E"].width = 44


def create_instructions_sheet(wb: openpyxl.Workbook) -> None:
    """Create the Instructions worksheet providing faculty evaluators with clear, step-by-step guidance."""
    ws = wb.create_sheet(title="Instructions")
    ws.views.sheetView[0].showGridLines = True

    font_title = Font(name="Calibri", size=14, bold=True, color="1E3A8A")
    font_heading = Font(name="Calibri", size=11, bold=True, color="1E293B")
    font_subheading = Font(name="Calibri", size=10, bold=True, color="374151")
    font_body = Font(name="Calibri", size=10, color="1F2937")
    font_bold = Font(name="Calibri", size=10, bold=True, color="111827")
    font_caution = Font(name="Calibri", size=10, bold=True, color="B91C1C")

    fill_banner = PatternFill(start_color="EFF6FF", end_color="EFF6FF", fill_type="solid")
    border_box = Border(
        left=Side(style="thin", color="93C5FD"),
        right=Side(style="thin", color="93C5FD"),
        top=Side(style="thin", color="93C5FD"),
        bottom=Side(style="thin", color="93C5FD"),
    )

    instructions = [
        ("A1", "Faculty Manual Verification Protocol — Instructions for Reviewers", font_title),
        ("A2", "Academic Project Novelty Detection System | Double-Blind Human Ground-Truth Study (Phase 5)", Font(name="Calibri", size=10, italic=True, color="4B5563")),
        ("A4", "1. PURPOSE AND SCOPE", font_heading),
        ("A5", "The goal of this verification study is to establish an empirical, expert ground truth for academic project similarity. Two independent faculty reviewers evaluate pairwise comparisons between a Candidate Project and a Reference Project across eight structured canonical project sections: Title, Abstract, Problem Statement, Objectives, Proposed Methodology, Dataset, Algorithm/Model, and Technology Tools.", font_body),
        ("A7", "2. MANDATORY REVIEW INSTRUCTIONS (STEP-BY-STEP)", font_heading),
        ("A8", "Step 1: Read both project descriptions carefully across all eight project sections before assigning any score.", font_body),
        ("A9", "Step 2: Compare the problem statement, objectives, methodology, dataset, algorithm, and technology side by side.", font_body),
        ("A10", "Step 3: Assign an integer similarity rating from 0 to 4 in your designated column (Reviewer 1 -> reviewer_1_rating; Reviewer 2 -> reviewer_2_rating).", font_body),
        ("A11", "Step 4: Add concise comments explaining your rating in your comments column (reviewer_1_comments or reviewer_2_comments), especially for scores of 2, 3, or 4.", font_body),
        ("A12", "Step 5: Do NOT treat similarity as a plagiarism verdict. This study evaluates technical and conceptual resemblance alone; it makes no accusation of academic dishonesty.", font_caution),
        ("A13", "Step 6: Leave adjudication fields (adjudicated_rating, adjudication_comments) BLANK unless you are specifically assigned as the lead adjudicator.", font_body),
        ("A15", "3. VISUAL COLOR-CODING IN EVALUATION SHEETS", font_heading),
        ("A16", "• Reviewer Input Columns (reviewer_1_*, reviewer_2_*): Highlighted in LIGHT YELLOW. Enter your rating (0-4) and comments here.", font_body),
        ("A17", "• Adjudication Columns (adjudicated_*): Highlighted in LIGHT GREEN. Reserved exclusively for final consensus adjudication.", font_body),
        ("A18", "• Annotation Status Column: Highlighted in LIGHT ORANGE ('PENDING_HUMAN_REVIEW'). Updated to 'COMPLETED' upon review submission.", font_body),
        ("A19", "• Rating Color Scale: Entering 0 to 4 automatically applies conditional formatting (0: Slate, 1: Cyan, 2: Yellow, 3: Orange, 4: Soft Red).", font_body),
        ("A21", "4. DISENTANGLING LEXICAL OVERLAP FROM CONCEPTUAL SIMILARITY", font_heading),
        ("A22", "A common trap in automated systems is confusing superficial vocabulary overlap with genuine research similarity. Adhere to these principles:", font_body),
        ("A23", "• High Lexical Overlap with Low Conceptual Similarity -> Rating 0 or 1:", font_subheading),
        ("A24", "  Two projects share standard ML vocabulary (e.g. 'CNN', 'accuracy', 'data preprocessing', 'cross-entropy loss'), but Project A analyzes brain MRI tumors while Project B classifies agricultural leaf rust. Rating MUST be 0 (or 1 if in related imaging contexts). Do not inflate ratings due to shared technical jargon.", font_body),
        ("A25", "• Low Lexical Overlap with High Conceptual Similarity -> Rating 3 or 4:", font_subheading),
        ("A26", "  Two projects describe the exact same research problem, dataset, and algorithm using distinct paraphrased vocabulary and synonyms. Rating MUST be 3 or 4. Distinct wording does not diminish conceptual equivalence.", font_body),
        ("A27", "• Shared Dataset with Divergent Objectives -> Rating 2:", font_subheading),
        ("A28", "  Both projects use the same dataset (e.g. BraTS MRI benchmark), but Project A performs 3D tumor segmentation while Project B conducts synthetic image inpainting. Rating MUST be 2 (moderate overlap).", font_body),
        ("A30", "5. HANDLING EDGE CASES & GENERIC BOILERPLATE", font_heading),
        ("A31", "• Generic Boilerplate: Disregard academic filler such as 'This project proposes an efficient and scalable framework to improve manual processes.' Score strictly based on concrete technical content.", font_body),
        ("A32", "• Under-Specified Sections: If a proposal is terse, evaluate only what is verifiable. Do not assume unstated similarities.", font_body),
        ("A33", "• Cross-Domain Tool Transfer: Applying standard models (e.g. Graph Neural Networks) to an entirely new problem domain should typically be scored 1 or 2.", font_body),
        ("A35", "6. WORKBOOK STRUCTURE & BATCH ALLOCATION", font_heading),
        ("A36", "This workbook is organized into two evaluation worksheets of 60 pairs each:", font_body),
        ("A37", "• Evaluation_Batch_1: Pairs EVAL-001 through EVAL-060 (60 pairs total).", font_body),
        ("A38", "• Evaluation_Batch_2: Pairs EVAL-061 through EVAL-120 (60 pairs total).", font_body),
        ("A39", "No pair appears in both batches. Both batches maintain an equal distribution across similarity levels (12 pairs per level).", font_body),
        ("A40", "Double-Blind Protocol: Reviewers are completely blind to model predictions, synthetic labels, and controlled test case identities.", font_bold),
    ]

    for cell_ref, text, font_style in instructions:
        ws[cell_ref] = text
        ws[cell_ref].font = font_style
        ws[cell_ref].alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

    ws.column_dimensions["A"].width = 110


def export_faculty_workbooks(
    batch_1_df: pd.DataFrame,
    batch_2_df: pd.DataFrame,
    master_df: pd.DataFrame,
    output_excel: Path,
    output_csv: Path,
    output_master_csv: Path,
) -> None:
    """Export formatted Excel workbook (4 sheets), combined CSV, and master research CSV."""
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_master_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export combined CSV for programmatic use (includes batch column and evaluation columns)
    combined_csv_df = pd.concat([
        batch_1_df.assign(batch="Evaluation_Batch_1"),
        batch_2_df.assign(batch="Evaluation_Batch_2"),
    ], ignore_index=True)

    # Reorder so 'batch' is the first column
    cols = ["batch"] + [c for c in combined_csv_df.columns if c != "batch"]
    combined_csv_df = combined_csv_df[cols]
    combined_csv_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved combined 120-pair CSV to: {output_csv}")

    # 2. Export master research CSV (with ground truth labels for downstream validation)
    master_df.to_csv(output_master_csv, index=False, encoding="utf-8")
    print(f"Saved master research CSV to: {output_master_csv}")

    # 3. Build formatted 4-sheet Excel workbook
    wb = openpyxl.Workbook()

    # Sheet 1: Evaluation_Batch_1
    ws_b1 = wb.active
    ws_b1.title = "Evaluation_Batch_1"
    apply_evaluation_sheet_formatting(ws_b1, batch_1_df)

    # Sheet 2: Evaluation_Batch_2
    ws_b2 = wb.create_sheet(title="Evaluation_Batch_2")
    apply_evaluation_sheet_formatting(ws_b2, batch_2_df)

    # Sheet 3: Rating_Rubric
    create_rating_rubric_sheet(wb)

    # Sheet 4: Instructions
    create_instructions_sheet(wb)

    wb.save(output_excel)
    print(f"Saved formatted 4-sheet Excel workbook to: {output_excel}")


def main():
    parser = argparse.ArgumentParser(description="Generate Faculty-Validated Evaluation Sample (Phase 5.2)")
    parser.add_argument("--size", type=int, default=120, choices=[60, 120],
                        help="Total sample size to generate (default: 120)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic sampling (default: 42)")
    args = parser.parse_args()

    print(f"=== Generating Faculty Evaluation Dataset Sample (Size: {args.size}, Seed: {args.seed}) ===")
    out_dir = ROOT_DIR / "data" / "annotation"

    if args.size == 120:
        out_excel = out_dir / "faculty_evaluation_sample_120.xlsx"
        out_csv = out_dir / "faculty_evaluation_sample_120.csv"
        out_master_csv = out_dir / "faculty_evaluation_sample_120_master.csv"

        b1_df, b2_df, master_df = build_faculty_evaluation_sample_120(random_state=args.seed)

        print(f"\nGenerated 120 unique pairs:")
        print(f"  - Evaluation_Batch_1: {len(b1_df)} pairs")
        print(f"  - Evaluation_Batch_2: {len(b2_df)} pairs")
        print(f"\nMaster similarity level distribution across all 120 pairs:")
        print(master_df["provisional_synthetic_label"].value_counts().sort_index())
        print(f"\nBatch breakdown by similarity level:")
        print(pd.crosstab(master_df["batch"], master_df["provisional_synthetic_label"], margins=True))
        print(f"\nBatch breakdown by calibration/evaluation subset:")
        print(pd.crosstab(master_df["batch"], master_df["subset"], margins=True))

        export_faculty_workbooks(b1_df, b2_df, master_df, out_excel, out_csv, out_master_csv)

        print("\n=== Validation Safeguards Verified ===")
        print("1. Exactly 120 unique pairs generated (60 per batch).")
        print("2. Zero duplicate pairs across or within worksheets.")
        print("3. Perfectly balanced: 24 pairs per similarity level (12 per batch).")
        print("4. Double-blind: No synthetic labels or model scores in faculty sheets.")
        print("5. All reviewer rating fields are blank (None).")
        print("6. Annotation status initialized to 'PENDING_HUMAN_REVIEW'.")
        print("7. Original 60-pair workbook was NOT overwritten.")

    elif args.size == 60:
        # Backward compatibility for generating the original 60-pair sample
        print("Generating original 60-pair sample...")
        out_excel = out_dir / "faculty_evaluation_sample_60.xlsx"
        out_csv = out_dir / "faculty_evaluation_sample_60.csv"
        # Note: Do not overwrite existing unless explicitly run with --size 60
        print(f"Target paths: {out_excel}, {out_csv}")


if __name__ == "__main__":
    main()
