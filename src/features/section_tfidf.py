import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

FEATURE_NAMES = [f"{sec}_similarity" for sec in SECTION_COLUMNS.keys()]

class SectionTFIDFExtractor:
    """Extract section-wise TF-IDF features across 8 project text sections."""
    
    def __init__(self, section_cols=None, ngram_range=(1, 2), stop_words="english"):
        self.section_cols = section_cols or SECTION_COLUMNS
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.vectorizers = {}
        self.tfidf_matrices = {}
        self.project_ids = []
        self.project_to_index = {}
        self.projects_df = None

    def fit_transform_projects(self, projects_df: pd.DataFrame):
        """Fit TfidfVectorizers on all projects for each section."""
        self.projects_df = projects_df.copy()
        self.project_ids = list(projects_df["project_id"])
        self.project_to_index = {pid: idx for idx, pid in enumerate(self.project_ids)}

        for sec_name, col in self.section_cols.items():
            vec = TfidfVectorizer(stop_words=self.stop_words, ngram_range=self.ngram_range)
            text_data = projects_df[col].fillna("").astype(str)
            matrix = vec.fit_transform(text_data)
            self.vectorizers[sec_name] = vec
            self.tfidf_matrices[sec_name] = matrix
        return self

    def extract_pair_features(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Extract similarity features for a set of project pairs."""
        feature_df = pairs_df.copy()

        for sec_name in self.section_cols.keys():
            matrix = self.tfidf_matrices[sec_name]
            scores = []
            for _, row in pairs_df.iterrows():
                idx_a = self.project_to_index.get(row["project_A_id"])
                idx_b = self.project_to_index.get(row["project_B_id"])
                if idx_a is not None and idx_b is not None:
                    sim = cosine_similarity(matrix[idx_a], matrix[idx_b])[0][0]
                else:
                    sim = 0.0
                scores.append(sim)
            feature_df[f"{sec_name}_similarity"] = scores

        return feature_df

    def compute_candidate_similarity(self, new_project_dict: dict) -> pd.DataFrame:
        """
        Compute section similarity scores between a single candidate project 
        and all existing projects in the corpus.
        """
        similarities = {}
        for sec_name, col in self.section_cols.items():
            vec = self.vectorizers[sec_name]
            corpus_matrix = self.tfidf_matrices[sec_name]
            candidate_text = str(new_project_dict.get(col, ""))
            candidate_vec = vec.transform([candidate_text])
            sim_scores = cosine_similarity(candidate_vec, corpus_matrix)[0]
            similarities[f"{sec_name}_similarity"] = sim_scores

        sim_df = pd.DataFrame(similarities)
        sim_df["project_id"] = self.project_ids
        return sim_df
