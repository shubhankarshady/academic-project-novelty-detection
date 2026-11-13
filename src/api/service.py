"""
Novelty Scoring Service for the Academic Project Novelty Detection API.
Manages one-time model loading, candidate evaluation, section difference analysis,
and configurable scoring modes (production TF-IDF vs experimental hybrid).
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import os
import joblib

from src.explainability.explainer import NoveltyExplainer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_TFIDF_PATH = ROOT_DIR / "models" / "section_tfidf_extractor.joblib"
DEFAULT_RIDGE_PATH = ROOT_DIR / "models" / "ridge_fusion_model.joblib"
DEFAULT_DENSE_PATH = ROOT_DIR / "models" / "section_dense_extractor.joblib"
DEFAULT_CANDIDATE_PATH = ROOT_DIR / "models" / "candidate_constrained_fusion_model.joblib"


class NoveltyService:
    """
    Singleton service that loads trained ML/XAI artifacts once at startup
    and exposes high-level evaluation methods.
    Supports two scoring modes:
      1. 'tfidf': Existing production TF-IDF Ridge fusion model (default).
      2. 'hybrid': Experimental lexical-semantic fusion model with configurable alpha.
    """

    def __init__(
        self,
        tfidf_path: Optional[Union[str, Path]] = None,
        ridge_path: Optional[Union[str, Path]] = None,
        dense_path: Optional[Union[str, Path]] = None,
        candidate_path: Optional[Union[str, Path]] = None,
        scoring_mode: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
        explainer: Optional[NoveltyExplainer] = None
    ):
        if explainer is not None:
            self.explainer = explainer
            self.scoring_mode = getattr(explainer, "scoring_mode", "tfidf")
            self.hybrid_alpha = getattr(explainer, "hybrid_alpha", 0.50)
            self.reference_corpus_size = len(explainer.projects_df)
            self._tfidf_explainer = explainer
            self._hybrid_explainer = None
            self.dense_extractor = getattr(explainer, "dense_extractor", None)
            return

        self.tfidf_path = Path(tfidf_path) if tfidf_path is not None else DEFAULT_TFIDF_PATH
        self.ridge_path = Path(ridge_path) if ridge_path is not None else DEFAULT_RIDGE_PATH
        self.dense_path = Path(dense_path) if dense_path is not None else DEFAULT_DENSE_PATH
        self.candidate_path = Path(candidate_path) if candidate_path is not None else DEFAULT_CANDIDATE_PATH

        # 1. Resolve scoring mode (default: 'tfidf')
        env_mode = os.environ.get("NOVELTY_SCORING_MODE", "tfidf")
        mode_raw = scoring_mode if scoring_mode is not None else env_mode
        mode_clean = str(mode_raw).strip().lower()
        if mode_clean not in {"tfidf", "hybrid"}:
            raise ValueError(f"Invalid scoring mode: '{mode_raw}'. Must be 'tfidf' or 'hybrid'.")
        self.scoring_mode = mode_clean

        # 2. Resolve hybrid alpha (default: 0.50)
        env_alpha = os.environ.get("NOVELTY_HYBRID_ALPHA", "0.50")
        alpha_raw = hybrid_alpha if hybrid_alpha is not None else env_alpha
        try:
            alpha_val = float(alpha_raw)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid hybrid_alpha: '{alpha_raw}'. Must be a float between 0.0 and 1.0.")
        if not (0.0 <= alpha_val <= 1.0):
            raise ValueError(f"hybrid_alpha must be between 0.0 and 1.0, got: {alpha_val}")
        self.hybrid_alpha = alpha_val

        # 3. Always verify baseline TF-IDF and Ridge production checkpoints
        if not self.tfidf_path.exists():
            raise FileNotFoundError(f"TF-IDF extractor checkpoint not found at: {self.tfidf_path}")
        if not self.ridge_path.exists():
            raise FileNotFoundError(f"Ridge model checkpoint not found at: {self.ridge_path}")

        # Load production models
        self.tfidf_extractor = joblib.load(self.tfidf_path)
        self.ridge_model = joblib.load(self.ridge_path)

        # Placeholders for lazy/hybrid loading
        self.dense_extractor = None
        self.candidate_model = None
        self._hybrid_explainer = None

        # 4. Initialize production TF-IDF explainer
        self._tfidf_explainer = NoveltyExplainer(
            tfidf_extractor=self.tfidf_extractor,
            model=self.ridge_model,
            scoring_mode="tfidf",
            hybrid_alpha=self.hybrid_alpha
        )

        # 5. If hybrid mode configured at startup, eagerly initialize it
        if self.scoring_mode == "hybrid":
            self._ensure_hybrid_initialized()
            self.explainer = self._hybrid_explainer
        else:
            self.explainer = self._tfidf_explainer

        self.reference_corpus_size = len(self.explainer.projects_df)

    def _ensure_hybrid_initialized(self) -> None:
        """Eagerly or lazily initialize dense extractor and candidate fusion model."""
        if not self.dense_path.exists():
            raise RuntimeError(
                f"Cannot initialize hybrid scoring mode: dense extractor artifact not found at: {self.dense_path}"
            )
        if not self.candidate_path.exists():
            raise RuntimeError(
                f"Cannot initialize hybrid scoring mode: candidate fusion model artifact not found at: {self.candidate_path}"
            )

        if self.dense_extractor is None:
            self.dense_extractor = joblib.load(self.dense_path)
        if self.candidate_model is None:
            self.candidate_model = joblib.load(self.candidate_path)

        if self._hybrid_explainer is None or self._hybrid_explainer.hybrid_alpha != self.hybrid_alpha:
            self._hybrid_explainer = NoveltyExplainer(
                tfidf_extractor=self.tfidf_extractor,
                model=self.candidate_model,
                projects_df=self.tfidf_extractor.projects_df,
                dense_extractor=self.dense_extractor,
                scoring_mode="hybrid",
                hybrid_alpha=self.hybrid_alpha
            )

    def get_explainer_for_mode(
        self,
        mode: str,
        alpha: Optional[float] = None
    ) -> NoveltyExplainer:
        """Return the appropriate explainer for the requested mode and alpha."""
        mode_clean = str(mode).strip().lower()
        if mode_clean not in {"tfidf", "hybrid"}:
            raise ValueError(f"Invalid scoring mode: '{mode}'. Must be 'tfidf' or 'hybrid'.")

        if mode_clean == "tfidf":
            return self._tfidf_explainer

        # Hybrid mode requested
        self._ensure_hybrid_initialized()
        effective_alpha = float(alpha if alpha is not None else self.hybrid_alpha)
        if not (0.0 <= effective_alpha <= 1.0):
            raise ValueError(f"hybrid_alpha must be between 0.0 and 1.0, got: {effective_alpha}")

        if self._hybrid_explainer is None or self._hybrid_explainer.hybrid_alpha != effective_alpha:
            self._hybrid_explainer = NoveltyExplainer(
                tfidf_extractor=self.tfidf_extractor,
                model=self.candidate_model,
                projects_df=self.tfidf_extractor.projects_df,
                dense_extractor=self.dense_extractor,
                scoring_mode="hybrid",
                hybrid_alpha=effective_alpha
            )
        return self._hybrid_explainer

    def score_proposal(
        self,
        candidate_dict: Dict[str, Any],
        top_k: int = 5,
        scoring_mode: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
        exclude_self: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate candidate proposal against the historical reference corpus.
        Does NOT save the proposal to the database or mutate the reference corpus.
        Allows overriding scoring mode ('tfidf' or 'hybrid') and alpha per request.
        """
        mode = (scoring_mode or self.scoring_mode).lower()
        alpha = float(hybrid_alpha if hybrid_alpha is not None else self.hybrid_alpha)

        active_explainer = self.get_explainer_for_mode(mode=mode, alpha=alpha)

        candidate_copy = dict(candidate_dict)
        if not candidate_copy.get("project_id"):
            candidate_copy["project_id"] = "NEW_PROPOSAL"

        # Explain proposal against reference corpus
        report = active_explainer.explain_proposal(
            candidate_dict=candidate_copy,
            top_k=top_k,
            exclude_self=exclude_self,
            scoring_mode=mode,
            hybrid_alpha=alpha
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
    dense_path: Optional[Union[str, Path]] = None,
    candidate_path: Optional[Union[str, Path]] = None,
    scoring_mode: Optional[str] = None,
    hybrid_alpha: Optional[float] = None,
    force_reload: bool = False
) -> NoveltyService:
    """Retrieve or initialize the singleton NoveltyService instance."""
    global _novelty_service_instance
    if force_reload or _novelty_service_instance is None:
        _novelty_service_instance = NoveltyService(
            tfidf_path=tfidf_path,
            ridge_path=ridge_path,
            dense_path=dense_path,
            candidate_path=candidate_path,
            scoring_mode=scoring_mode,
            hybrid_alpha=hybrid_alpha
        )
    return _novelty_service_instance

def set_novelty_service(service: NoveltyService) -> None:
    """Explicitly set the service instance (useful for testing with mock models)."""
    global _novelty_service_instance
    _novelty_service_instance = service

