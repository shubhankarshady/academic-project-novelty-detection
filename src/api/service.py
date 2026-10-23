"""
Novelty Scoring Service for the Academic Project Novelty Detection API.
Manages one-time model loading, candidate evaluation, and section difference analysis.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import joblib

from src.explainability.explainer import NoveltyExplainer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_TFIDF_PATH = ROOT_DIR / "models" / "section_tfidf_extractor.joblib"
DEFAULT_RIDGE_PATH = ROOT_DIR / "models" / "ridge_fusion_model.joblib"


class NoveltyService:
    """
    Singleton service that loads trained ML/XAI artifacts once at startup
    and exposes high-level evaluation methods.
    """

    def __init__(
        self,
        tfidf_path: Optional[Union[str, Path]] = None,
        ridge_path: Optional[Union[str, Path]] = None,
        explainer: Optional[NoveltyExplainer] = None
    ):
        if explainer is not None:
            self.explainer = explainer
            self.reference_corpus_size = len(explainer.projects_df)
            return

        self.tfidf_path = Path(tfidf_path) if tfidf_path is not None else DEFAULT_TFIDF_PATH
        self.ridge_path = Path(ridge_path) if ridge_path is not None else DEFAULT_RIDGE_PATH

        if not self.tfidf_path.exists():
            raise FileNotFoundError(f"TF-IDF extractor checkpoint not found at: {self.tfidf_path}")
        if not self.ridge_path.exists():
            raise FileNotFoundError(f"Ridge model checkpoint not found at: {self.ridge_path}")

        # Load models once
        self.tfidf_extractor = joblib.load(self.tfidf_path)
        self.ridge_model = joblib.load(self.ridge_path)

        # Initialize Explainability Engine
        self.explainer = NoveltyExplainer(
            tfidf_extractor=self.tfidf_extractor,
            model=self.ridge_model
        )
        self.reference_corpus_size = len(self.explainer.projects_df)

    def score_proposal(
        self,
        candidate_dict: Dict[str, Any],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate candidate proposal against the historical reference corpus.
        Does NOT save the proposal to the database or mutate the reference corpus.
        """
        candidate_copy = dict(candidate_dict)
        if not candidate_copy.get("project_id"):
            candidate_copy["project_id"] = "NEW_PROPOSAL"

        # Explain proposal against reference corpus (strictly self-excluded)
        report = self.explainer.explain_proposal(
            candidate_dict=candidate_copy,
            top_k=top_k,
            exclude_self=True
        )

        # Enrich each top match with section difference summary
        for match in report.get("top_matches", []):
            sec_scores = match.get("section_similarity_scores", {})
            high_overlap = [sec for sec, score in sec_scores.items() if score >= 0.30]
            differentiating = [sec for sec, score in sec_scores.items() if score < 0.15]
            match["difference_summary"] = {
                "high_overlap_sections": high_overlap,
                "differentiating_sections": differentiating
            }

        return report


# Global singleton instance
_novelty_service_instance: Optional[NoveltyService] = None

def get_novelty_service(
    tfidf_path: Optional[Union[str, Path]] = None,
    ridge_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False
) -> NoveltyService:
    """Retrieve or initialize the singleton NoveltyService instance."""
    global _novelty_service_instance
    if force_reload or _novelty_service_instance is None:
        _novelty_service_instance = NoveltyService(
            tfidf_path=tfidf_path,
            ridge_path=ridge_path
        )
    return _novelty_service_instance

def set_novelty_service(service: NoveltyService) -> None:
    """Explicitly set the service instance (useful for testing with mock models)."""
    global _novelty_service_instance
    _novelty_service_instance = service
