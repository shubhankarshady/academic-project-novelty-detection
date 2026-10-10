import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

SECTION_COLUMNS = {
    "title": "project_title",
    "abstract": "abstract",
    "problem": "problem_statement",
    "objectives": "objectives",
    "methodology": "proposed_methodology",
    "dataset": "dataset",
    "algorithm": "algorithm_model",
    "technology": "technology_tools"
}

DENSE_FEATURE_NAMES = [f"{sec}_dense_similarity" for sec in SECTION_COLUMNS.keys()]

class SectionDenseEmbeddingExtractor:
    """Extract section-wise dense transformer embeddings and cosine similarities using SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        section_cols: Optional[Dict[str, str]] = None,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.section_cols = section_cols or SECTION_COLUMNS
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.project_ids: List[str] = []
        self.project_to_index: Dict[str, int] = {}
        self.projects_df: Optional[pd.DataFrame] = None

    def _lazy_load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def fit_transform_projects(self, projects_df: pd.DataFrame):
        """Encode all text sections for corpus projects into dense vectors."""
        self._lazy_load_model()
        self.projects_df = projects_df.copy()
        self.project_ids = list(projects_df["project_id"])
        self.project_to_index = {pid: idx for idx, pid in enumerate(self.project_ids)}

        for sec_name, col in self.section_cols.items():
            text_list = projects_df[col].fillna("").astype(str).tolist()
            # Encode with sentence transformers
            vecs = self.model.encode(
                text_list,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self.embeddings[sec_name] = vecs
        return self

    def extract_pair_features(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate section-wise dense cosine similarities for project pairs."""
        feature_df = pairs_df.copy()

        for sec_name in self.section_cols.keys():
            emb_matrix = self.embeddings[sec_name]
            scores = []
            for _, row in pairs_df.iterrows():
                idx_a = self.project_to_index.get(row["project_A_id"])
                idx_b = self.project_to_index.get(row["project_B_id"])
                if idx_a is not None and idx_b is not None:
                    # Vectors are normalized, so dot product = cosine similarity
                    sim = float(np.dot(emb_matrix[idx_a], emb_matrix[idx_b]))
                else:
                    sim = 0.0
                scores.append(sim)
            feature_df[f"{sec_name}_dense_similarity"] = scores

        return feature_df

    def compute_candidate_similarity(self, candidate_dict: Dict[str, Any]) -> pd.DataFrame:
        """Compute dense similarity between a candidate project proposal and all corpus projects."""
        self._lazy_load_model()
        similarities = {}

        for sec_name, col in self.section_cols.items():
            corpus_emb = self.embeddings[sec_name]
            text = str(candidate_dict.get(col, ""))
            cand_emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
            # Dot product with normalized embeddings
            sim_scores = np.dot(corpus_emb, cand_emb)
            similarities[f"{sec_name}_dense_similarity"] = sim_scores

        sim_df = pd.DataFrame(similarities)
        sim_df["project_id"] = self.project_ids
        return sim_df
