from pathlib import Path
import pandas as pd

def load_projects(filepath: str | Path) -> pd.DataFrame:
    """Load projects dataset Excel file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Projects dataset not found at {filepath}")
    
    excel_file = pd.ExcelFile(filepath)
    sheet = "Projects" if "Projects" in excel_file.sheet_names else excel_file.sheet_names[0]
    df = pd.read_excel(excel_file, sheet_name=sheet)
    return df

def load_similarity_pairs(filepath: str | Path) -> pd.DataFrame:
    """Load project similarity pairs Excel file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Similarity pairs dataset not found at {filepath}")
    
    excel_file = pd.ExcelFile(filepath)
    sheet = "SimilarityPairs" if "SimilarityPairs" in excel_file.sheet_names else excel_file.sheet_names[0]
    df = pd.read_excel(excel_file, sheet_name=sheet)
    return df

def load_feature_csv(filepath: str | Path) -> pd.DataFrame:
    """Load pre-computed section similarity feature CSV."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Feature CSV not found at {filepath}")
    return pd.read_csv(filepath)
