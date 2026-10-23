"""
Complete Technical Documentation Generator for Academic Project Novelty Detection System.
Generates:
1. docs/ML_MODEL_COMPLETE_GUIDE.txt (Plain text with ASCII diagrams and complete formulas)
2. docs/ML_MODEL_COMPLETE_GUIDE.pdf (Professional publication-quality PDF via ReportLab)
"""

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

TXT_PATH = DOCS_DIR / "ML_MODEL_COMPLETE_GUIDE.txt"
PDF_PATH = DOCS_DIR / "ML_MODEL_COMPLETE_GUIDE.pdf"

print("Compiling complete documentation text...")
