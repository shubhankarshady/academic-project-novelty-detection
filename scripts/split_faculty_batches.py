# -*- coding: utf-8 -*-
"""
Stratified 5-Way Faculty Batch Generator
=========================================
For each domain:
  - Takes all pairs where domain_A or domain_B matches the domain (~450-502 pairs).
  - Stratifies by expert_similarity_label (0-4) and splits into 5 equal batches.
  - Each faculty member gets ~89-100 pairs with proportional score distribution.
  - Uses "Agree / Disagree + Correct Score" workflow for fast review.
  - Saves individual Excel files: data/faculty_batches/<DomainFolder>/Faculty_1.xlsx ... Faculty_5.xlsx
"""

import os
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


SIM_COLORS = {
    "Not Similar":        "D9EAD3",
    "Slightly Similar":   "FFF2CC",
    "Moderately Similar": "FCE5CD",
    "Highly Similar":     "F4CCCC",
    "Nearly Duplicate":   "EA9999",
}
HEADER_FILL  = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
HEADER_FONT  = Font(bold=True, color="FFFFFF", size=11, name="Calibri")
VERDICT_FILL = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
THIN_BORDER  = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"),  bottom=Side(style="thin"),
)

LABEL_MAP = {
    0: "Not Similar",
    1: "Slightly Similar",
    2: "Moderately Similar",
    3: "Highly Similar",
    4: "Nearly Duplicate",
}

OUTPUT_COLS = [
    "pair_id",
    "AI Score (0-4)",
    "AI Similarity Level",
    "project_A_id", "domain_A", "title_A", "abstract_A", "algorithm_A", "tools_A", "dataset_A",
    "project_B_id", "domain_B", "title_B", "abstract_B", "algorithm_B", "tools_B", "dataset_B",
    "common_keywords",
]

COL_RENAME = {
    "pair_id":              "Pair ID",
    "AI Score (0-4)":       "AI Score (0-4)",
    "AI Similarity Level":  "AI Similarity Level",
    "project_A_id":         "Project A - ID",
    "domain_A":             "Project A - Domain",
    "title_A":              "Project A - Title",
    "abstract_A":           "Project A - Abstract",
    "algorithm_A":          "Project A - Algorithm",
    "tools_A":              "Project A - Tech Stack",
    "dataset_A":            "Project A - Dataset",
    "project_B_id":         "Project B - ID",
    "domain_B":             "Project B - Domain",
    "title_B":              "Project B - Title",
    "abstract_B":           "Project B - Abstract",
    "algorithm_B":          "Project B - Algorithm",
    "tools_B":              "Project B - Tech Stack",
    "dataset_B":            "Project B - Dataset",
    "common_keywords":      "Common Keywords",
}

VERDICT_COLS = [
    "Verdict  (Agree / Disagree)",
    "Correct Score (0-4)  [if Disagree]",
    "Faculty Comments",
    "Reviewed By",
    "Review Date",
]


def safe_folder_name(domain):
    name = re.sub(r"[^\w\s]", "", domain)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:60]


def find_common_keywords(kw_a, kw_b):
    if pd.isna(kw_a) or pd.isna(kw_b):
        return ""
    set_a = set(str(kw_a).lower().replace(",", " ").split())
    set_b = set(str(kw_b).lower().replace(",", " ").split())
    common = {w for w in (set_a & set_b) if len(w) > 2}
    return ", ".join(sorted(common)[:10]) if common else "None"


def stratified_5_split(df):
    buckets = [[] for _ in range(5)]
    for label in [0, 1, 2, 3, 4]:
        group = df[df["expert_similarity_label"] == label].sample(frac=1, random_state=42).reset_index(drop=True)
        for i, (_, row) in enumerate(group.iterrows()):
            buckets[i % 5].append(row)
    return [pd.DataFrame(b).reset_index(drop=True) for b in buckets]


def col_width(header):
    h = str(header)
    if "Abstract" in h: return 52
    if "Title" in h:    return 36
    if "Algorithm" in h or "Tech Stack" in h or "Dataset" in h: return 30
    if "Domain" in h:   return 26
    if "Common Keywords" in h: return 28
    if "Verdict" in h or "Correct Score" in h: return 26
    if "Comment" in h:  return 28
    if "Reviewed By" in h or "Review Date" in h: return 18
    return 16


def style_worksheet(ws, sim_col_idx):
    max_col = ws.max_column
    max_row = ws.max_row

    for col in range(1, max_col + 1):
        cell = ws.cell(1, col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER
        ws.column_dimensions[get_column_letter(col)].width = col_width(cell.value or "")

    # Identify verdict start col (after data cols, account for Sr.No. index at col 1)
    verdict_start_col = len(OUTPUT_COLS) + 2

    for row in range(2, max_row + 1):
        sim_val = ws.cell(row, sim_col_idx).value if sim_col_idx else None
        row_fill = None
        if sim_val and sim_val in SIM_COLORS:
            row_fill = PatternFill(start_color=SIM_COLORS[sim_val],
                                   end_color=SIM_COLORS[sim_val], fill_type="solid")
        for col in range(1, max_col + 1):
            cell = ws.cell(row, col)
            cell.border = THIN_BORDER
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if col >= verdict_start_col:
                cell.fill = VERDICT_FILL
                cell.font = Font(name="Calibri", size=10, color="1F4E79")
            elif row_fill:
                cell.fill = row_fill

    ws.freeze_panes = "D2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 36


def write_instructions(wb, domain, faculty_num, total_pairs):
    ws = wb.create_sheet("Instructions")
    instructions = [
        ("Academic Project Similarity - Faculty Verification", True, 16),
        ("", False, 11),
        (f"Domain :  {domain}", False, 12),
        (f"Faculty File :  Batch {faculty_num} of 5", False, 12),
        (f"Total Pairs to Review :  {total_pairs}", False, 12),
        ("", False, 11),
        ("How to Review (3 Simple Steps)", True, 13),
        ("", False, 11),
        ("Step 1 - Open the 'Review Pairs' sheet.", False, 11),
        ("Step 2 - For each row, read the two project titles and abstracts side-by-side.", False, 11),
        ("          The AI has already given a predicted similarity score (0-4).", False, 11),
        ("Step 3 - Fill in these columns:", False, 11),
        ("   -  Verdict (Agree / Disagree)  - Type  Agree  if you agree with the AI score.", False, 11),
        ("      Type  Disagree  if you think the AI is wrong.", False, 11),
        ("   -  Correct Score (0-4) [if Disagree]  - Write your own score (0 to 4).", False, 11),
        ("   -  Faculty Comments  - (Optional) Write any notes.", False, 11),
        ("   -  Reviewed By  - Your name.", False, 11),
        ("   -  Review Date  - Date of review.", False, 11),
        ("", False, 11),
        ("Similarity Scale Reference", True, 13),
        ("", False, 11),
        ("  0 - Not Similar        : Completely different topics, methods, and goals.", False, 11),
        ("  1 - Slightly Similar   : Broadly same field but different approach/outcome.", False, 11),
        ("  2 - Moderately Similar : Same domain, somewhat overlapping methods or dataset.", False, 11),
        ("  3 - Highly Similar     : Very similar objective/methods - distinct contribution.", False, 11),
        ("  4 - Nearly Duplicate   : Near-identical topic/methodology - almost redundant.", False, 11),
        ("", False, 11),
        ("Thank you for your valuable time and expertise!", False, 12),
    ]
    ws.column_dimensions["A"].width = 90
    for r_idx, (text, bold, size) in enumerate(instructions, start=1):
        cell = ws.cell(r_idx, 1, text)
        cell.font = Font(bold=bold, size=size, name="Calibri",
                         color="1F4E79" if bold else "000000")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r_idx].height = 24 if bold else 18
    return ws


def write_faculty_excel(df_out, path, domain, faculty_num, total_pairs):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Review Pairs", index=True, index_label="Sr. No.")

    wb = load_workbook(path)
    ws = wb["Review Pairs"]

    sim_col_idx = None
    for col in range(1, ws.max_column + 1):
        if ws.cell(1, col).value == "AI Similarity Level":
            sim_col_idx = col
            break

    style_worksheet(ws, sim_col_idx)
    write_instructions(wb, domain, faculty_num, total_pairs)
    wb.move_sheet("Instructions", offset=-wb.sheetnames.index("Instructions"))
    wb.save(path)


def generate_faculty_batches():
    root_dir = Path(__file__).resolve().parent.parent
    src      = root_dir / "data" / "raw" / "Academic_Project_Similarity_Synthetic_Dataset.xlsx"
    out_root = root_dir / "data" / "faculty_batches"
    out_root.mkdir(parents=True, exist_ok=True)

    tmp = os.path.join(tempfile.gettempdir(), "_tmp_dataset.xlsx")
    shutil.copy2(src, tmp)
    xl    = pd.ExcelFile(tmp)
    proj  = pd.read_excel(xl, "Projects")
    pairs = pd.read_excel(xl, "SimilarityPairs")
    xl.close()
    os.remove(tmp)

    proj_a = proj[["project_id","domain","project_title","abstract","keywords","algorithm_model","technology_tools","dataset"]].copy()
    proj_a.columns = ["project_A_id","domain_A","title_A","abstract_A","keywords_A","algorithm_A","tools_A","dataset_A"]

    proj_b = proj[["project_id","domain","project_title","abstract","keywords","algorithm_model","technology_tools","dataset"]].copy()
    proj_b.columns = ["project_B_id","domain_B","title_B","abstract_B","keywords_B","algorithm_B","tools_B","dataset_B"]

    merged = pairs.merge(proj_a, on="project_A_id").merge(proj_b, on="project_B_id")
    merged["AI Score (0-4)"]      = merged["expert_similarity_label"]
    merged["AI Similarity Level"] = merged["expert_similarity_label"].map(LABEL_MAP)
    merged["common_keywords"]     = merged.apply(
        lambda r: find_common_keywords(r["keywords_A"], r["keywords_B"]), axis=1
    )

    domains = sorted(proj["domain"].unique())
    print(f"\n{'='*65}")
    print(f"  Generating Faculty Batches  -  {len(domains)} Domains x 5 Files Each")
    print(f"{'='*65}\n")

    total_files = 0
    for domain in domains:
        domain_df   = merged[(merged["domain_A"] == domain) | (merged["domain_B"] == domain)].copy()
        domain_df   = domain_df.reset_index(drop=True)
        total_pairs = len(domain_df)
        splits      = stratified_5_split(domain_df)

        domain_dir = out_root / safe_folder_name(domain)
        domain_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Domain: {domain}  ({total_pairs} pairs -> ~{total_pairs//5} per faculty)")

        for fac_idx, split_df in enumerate(splits, start=1):
            split_df = split_df.reset_index(drop=True)
            split_df.index += 1
            split_df.index.name = "Sr. No."

            out_df = split_df[OUTPUT_COLS].rename(columns=COL_RENAME).copy()
            for vc in VERDICT_COLS:
                out_df[vc] = ""

            score_dist = split_df["expert_similarity_label"].value_counts().sort_index().to_dict()
            filepath   = domain_dir / f"Faculty_{fac_idx}.xlsx"

            write_faculty_excel(out_df, filepath, domain, fac_idx, len(split_df))

            dist_str = "  ".join(f"[{LABEL_MAP[k][:6]}({k}):{v}]" for k, v in score_dist.items())
            print(f"      Faculty_{fac_idx}  [{len(split_df):3d} pairs]  {dist_str}")
            total_files += 1
        print()

    print(f"{'='*65}")
    print(f"  Done!  {total_files} Excel files saved under:")
    print(f"  {out_root}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    generate_faculty_batches()

