"""
Explainable AI (XAI) Module for Academic Project Novelty Detection.

Provides transparent, evidence-based explanations for why a candidate academic proposal
is similar to historical projects in the corpus:
1. Section-level similarity scores across all 8 text sections.
2. Ranking of sections by degree of overlap.
3. Informative shared keywords and n-grams with source text snippets.
4. Additive Ridge model feature contributions reconstructing the exact predicted score.
5. Strict exclusion of self-matching proposals.
6. Transparent communication regarding lexical vs semantic limitations and benchmark caveats.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from src.features.section_tfidf import SECTION_COLUMNS, FEATURE_NAMES
from src.utils.metrics import calculate_novelty_score, get_novelty_level

# Sensible low-information and generic academic boilerplate terms
GENERIC_LOW_INFO_TERMS = {
    "using", "use", "uses", "used", "based", "project", "projects", "proposed",
    "proposal", "proposes", "propose", "study", "studies", "paper", "papers",
    "ug", "mini", "solution", "solutions", "work", "works", "approach", "approaches",
    "method", "methods", "methodology", "methodologies", "application", "applications",
    "develop", "developed", "develops", "developing", "development", "build", "builds",
    "building", "implement", "implements", "implemented", "implementing", "implementation",
    "generate", "generates", "generated", "generating", "useful", "analysis", "analyzes",
    "analyzed", "analyzing", "analyze", "evaluation", "evaluates", "evaluated", "evaluating",
    "evaluate", "results", "result", "performance", "technique", "techniques", "system",
    "systems", "domain", "domains", "effort", "efforts", "provide", "provides", "provided",
    "providing", "includes", "including", "include", "included", "presents", "present",
    "presented", "presenting", "various", "different", "overall", "primary", "secondary",
    "aims", "aim", "focuses", "focus", "improving", "practical", "reducing", "manual",
    "decision", "support", "classifications", "classification", "predictions", "prediction"
}

def is_informative_term(term: str) -> bool:
    """
    Check if a term or n-gram is informative (i.e. not a stop-word or generic filler).
    Preserves legitimate technical terms (algorithm names, datasets, frameworks).
    """
    if not term:
        return False
    term_lower = term.strip().lower()
    if len(term_lower) < 2:
        return False

    tokens = term_lower.split()
    # Unigram: reject if in stop words, generic list, or purely digits
    if len(tokens) == 1:
        if term_lower in ENGLISH_STOP_WORDS or term_lower in GENERIC_LOW_INFO_TERMS:
            return False
        if term_lower.isdigit():
            return False
        return True

    # N-gram:
    # 1. Reject if all tokens are generic or stop words
    if all(t in ENGLISH_STOP_WORDS or t in GENERIC_LOW_INFO_TERMS for t in tokens):
        return False
    # 2. Reject if starts or ends with generic filler/stop word
    if tokens[0] in ENGLISH_STOP_WORDS or tokens[0] in GENERIC_LOW_INFO_TERMS:
        return False
    if tokens[-1] in ENGLISH_STOP_WORDS or tokens[-1] in GENERIC_LOW_INFO_TERMS:
        return False
    return True

def extract_snippet(text: str, term: str, window: int = 45) -> str:
    """
    Extract a concise snippet from source text centered around the search term.
    Preserves exact source wording, uses word-boundary-aware context windows,
    and adds ellipses only when text has been omitted.
    """
    if not text or not term:
        return ""
    text_str = str(text).strip()
    if not text_str:
        return ""

    match = re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text_str.lower())
    if match:
        term_start, term_end = match.span()
    else:
        idx = text_str.lower().find(term.lower())
        if idx == -1:
            return ""
        term_start, term_end = idx, idx + len(term)

    raw_start = max(0, term_start - window)
    raw_end = min(len(text_str), term_end + window)

    # Word-boundary snapping for start
    if raw_start <= 0:
        start = 0
    else:
        space_before = -1
        for i in range(raw_start - 1, -1, -1):
            if text_str[i].isspace():
                space_before = i
                break
        space_after = -1
        for i in range(raw_start, term_start):
            if text_str[i].isspace():
                space_after = i
                break

        if space_after != -1 and (space_after - raw_start <= 15):
            start = space_after + 1
        elif space_before != -1:
            start = space_before + 1
        elif space_after != -1:
            start = space_after + 1
        else:
            start = 0

    # Word-boundary snapping for end
    if raw_end >= len(text_str):
        end = len(text_str)
    else:
        space_after = -1
        for i in range(raw_end, len(text_str)):
            if text_str[i].isspace():
                space_after = i
                break
        space_before = -1
        for i in range(raw_end - 1, term_end - 1, -1):
            if text_str[i].isspace():
                space_before = i
                break

        if space_after != -1 and (space_after - raw_end <= 15):
            end = space_after
        elif space_before != -1:
            end = space_before
        elif space_after != -1:
            end = space_after
        else:
            end = len(text_str)

    snippet_body = text_str[start:end].strip()
    if not snippet_body:
        return ""

    body_start = text_str.find(snippet_body, max(0, start - 5))
    if body_start == -1:
        body_start = text_str.find(snippet_body)
    body_end = body_start + len(snippet_body)

    prefix = "..." if body_start > 0 else ""
    suffix = "..." if body_end < len(text_str) else ""
    return prefix + snippet_body + suffix

def verify_snippet(snippet: str, source_text: str, term: str) -> bool:
    """Verify that a snippet matches the source text without mid-word truncations."""
    if not snippet or not source_text or not term:
        return False
    body = snippet
    if body.startswith("..."):
        body = body[3:]
    if body.endswith("..."):
        body = body[:-3]
    body = body.strip()
    return (body in source_text) and (term.lower() in body.lower())

class NoveltyExplainer:
    """
    Explainable AI Engine for Academic Project Novelty Detection.
    
    Attributes:
        tfidf_extractor: Fitted SectionTFIDFExtractor instance.
        model: Fitted Ridge regression model (or wrapper/dict with coef_ and intercept_).
        projects_df: DataFrame of historical projects in the reference corpus.
    """

    def __init__(
        self,
        tfidf_extractor,
        model,
        projects_df: Optional[pd.DataFrame] = None
    ):
        self.tfidf_extractor = tfidf_extractor
        self.projects_df = (
            projects_df if projects_df is not None else getattr(tfidf_extractor, "projects_df", None)
        )
        if self.projects_df is None:
            raise ValueError("Historical projects_df must be provided or attached to tfidf_extractor.")

        self.section_cols = getattr(tfidf_extractor, "section_cols", SECTION_COLUMNS)
        self.section_names = list(self.section_cols.keys())
        self.feature_names = [f"{sec}_similarity" for sec in self.section_names]

        # Extract model coefficients and intercept
        self._extract_model_parameters(model)

    def _extract_model_parameters(self, model: Any):
        """Extract coefficients and intercept from Ridge model, dict, or wrapper."""
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            self.model = model
            self.coef_ = np.asarray(model.coef_, dtype=float).ravel()
            self.intercept_ = float(model.intercept_)
        elif isinstance(model, dict) and "model" in model and hasattr(model["model"], "coef_"):
            self.model = model["model"]
            self.coef_ = np.asarray(self.model.coef_, dtype=float).ravel()
            self.intercept_ = float(self.model.intercept_)
        elif isinstance(model, dict) and "coefficients" in model and "intercept" in model:
            self.model = None
            coef_dict = model["coefficients"]
            self.coef_ = np.array([coef_dict.get(feat, 0.0) for feat in self.feature_names], dtype=float)
            self.intercept_ = float(model["intercept"])
        else:
            raise TypeError("Unsupported model type. Expected Ridge instance or dict with coef_/intercept_.")

        if len(self.coef_) != len(self.feature_names):
            raise ValueError(f"Coefficient dimension mismatch: expected {len(self.feature_names)}, got {len(self.coef_)}")

    def extract_shared_terms(
        self,
        sec_name: str,
        text_candidate: str,
        text_historical: str,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify shared informative n-grams using the fitted TF-IDF vectorizer for a section,
        verifying that every displayed keyword actually appears with whole-word boundaries in the source text.
        """
        if not text_candidate or not text_historical:
            return []

        text_cand_str = str(text_candidate).strip()
        text_hist_str = str(text_historical).strip()
        if not text_cand_str or not text_hist_str:
            return []

        vec: TfidfVectorizer = self.tfidf_extractor.vectorizers.get(sec_name)
        if vec is None:
            return []

        # Vectorize both texts with the fitted vocabulary
        v_cand = vec.transform([text_cand_str]).tocsr()
        v_hist = vec.transform([text_hist_str]).tocsr()

        # Element-wise product of normalized TF-IDF representations
        # In cosine similarity: v1 . v2 = sum(v1[k] * v2[k])
        overlap = v_cand.multiply(v_hist).tocoo()
        if overlap.nnz == 0:
            return []

        inv_vocab = {idx: term for term, idx in vec.vocabulary_.items()}
        shared = []

        cand_lower = text_cand_str.lower()
        hist_lower = text_hist_str.lower()

        for col_idx, val in zip(overlap.col, overlap.data):
            term = inv_vocab.get(col_idx)
            if not term:
                continue
            if not is_informative_term(term):
                continue
            term_lower = term.lower()
            # Strict verification: must physically appear with word boundaries in both source texts
            cand_pattern = rf"(?<!\w){re.escape(term_lower)}(?!\w)"
            hist_pattern = rf"(?<!\w){re.escape(term_lower)}(?!\w)"
            if re.search(cand_pattern, cand_lower) and re.search(hist_pattern, hist_lower):
                cand_snip = extract_snippet(text_cand_str, term)
                hist_snip = extract_snippet(text_hist_str, term)
                if cand_snip and hist_snip and verify_snippet(cand_snip, text_cand_str, term) and verify_snippet(hist_snip, text_hist_str, term):
                    shared.append({
                        "term": term,
                        "term_cosine_contribution": round(float(val), 5),
                        "candidate_snippet": cand_snip,
                        "historical_snippet": hist_snip
                    })

        # Sort terms by their contribution to the section's cosine similarity
        shared.sort(key=lambda x: x["term_cosine_contribution"], reverse=True)
        return shared[:top_n]

    def explain_pair(
        self,
        candidate_dict: Dict[str, Any],
        historical_row: Union[pd.Series, Dict[str, Any]],
        top_n_terms: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a transparent section-by-section and model-contribution breakdown
        for a pair of proposals (candidate vs historical).
        """
        if isinstance(historical_row, pd.Series):
            historical_dict = historical_row.to_dict()
        else:
            historical_dict = dict(historical_row)

        section_scores = {}
        section_contributions = {}
        section_evidence = {}
        unrounded_contribs = []

        # 1. Compute section similarities & signed model contributions
        for idx, sec_name in enumerate(self.section_names):
            col = self.section_cols[sec_name]
            vec: TfidfVectorizer = self.tfidf_extractor.vectorizers.get(sec_name)

            cand_text = str(candidate_dict.get(col, "")).strip()
            hist_text = str(historical_dict.get(col, "")).strip()

            if cand_text and hist_text and vec is not None:
                v_cand = vec.transform([cand_text])
                v_hist = vec.transform([hist_text])
                sim = float(cosine_similarity(v_cand, v_hist)[0, 0])
            else:
                sim = 0.0

            weight = float(self.coef_[idx])
            signed_contrib = sim * weight
            unrounded_contribs.append(signed_contrib)

            section_scores[sec_name] = round(sim, 4)
            section_contributions[sec_name] = round(signed_contrib, 4)

            # Keyword and snippet evidence
            shared_terms = self.extract_shared_terms(sec_name, cand_text, hist_text, top_n=top_n_terms)
            evidence_msg = (
                "Shared lexical evidence identified."
                if shared_terms
                else "No informative shared terms identified for this section."
            )
            section_evidence[sec_name] = {
                "similarity_score": round(sim, 4),
                "model_weight": round(weight, 4),
                "signed_contribution": round(signed_contrib, 4),
                "shared_terms_count": len(shared_terms),
                "shared_terms": shared_terms,
                "message": evidence_msg
            }

        # 2. Additive Reconstruction Check
        sum_of_contributions = float(sum(section_contributions.values()))
        raw_prediction = float(round(float(self.intercept_), 4) + sum_of_contributions)
        clipped_prediction = float(np.clip(raw_prediction, 0.0, 4.0))
        novelty_score_pct = calculate_novelty_score(clipped_prediction)
        novelty_tier = get_novelty_level(novelty_score_pct)

        # 3. Ranked Sections (by raw similarity descending)
        ranked_sections = sorted(
            [
                {
                    "section": sec,
                    "column": self.section_cols[sec],
                    "raw_similarity": section_scores[sec],
                    "model_weight": round(float(self.coef_[i]), 4),
                    "signed_contribution": section_contributions[sec]
                }
                for i, sec in enumerate(self.section_names)
            ],
            key=lambda x: x["raw_similarity"],
            reverse=True
        )

        return {
            "historical_project_id": str(historical_dict.get("project_id", "UNKNOWN")),
            "historical_project_title": str(historical_dict.get("project_title", "N/A")),
            "historical_domain": str(historical_dict.get("domain", "N/A")),
            "predicted_similarity": round(clipped_prediction, 4),
            "raw_similarity_prediction": round(raw_prediction, 4),
            "clipped_prediction": round(clipped_prediction, 4),
            "novelty_score_pct": round(novelty_score_pct, 2),
            "novelty_category": novelty_tier,
            "intercept": round(float(self.intercept_), 4),
            "sum_of_contributions": round(sum_of_contributions, 4),
            "reconstructed_raw_prediction": round(raw_prediction, 4),
            "section_similarity_scores": section_scores,
            "section_signed_contributions": section_contributions,
            "ranked_sections": ranked_sections,
            "section_evidence": section_evidence,
            "disclaimers": {
                "corpus_relativity": (
                    "Novelty score is an estimated heuristic relative to the current database. "
                    "It does not establish originality or prove plagiarism."
                ),
                "lexical_limitation": (
                    "TF-IDF measures exact lexical and n-gram overlap. High section similarity indicates shared terminology, "
                    "but low lexical overlap may miss paraphrased or conceptually similar ideas."
                ),
                "model_attribution": (
                    "Model contributions represent linear additive feature associations weighted by the fitted Ridge model, "
                    "not causal explanations of novelty, duplication, or copying."
                ),
                "evidence_integrity": (
                    "Shared keywords indicate lexical topic overlap, not academic plagiarism or definitive lack of novelty."
                )
            }
        }

    def explain_proposal(
        self,
        candidate_dict: Dict[str, Any],
        top_k: int = 3,
        exclude_self: bool = True
    ) -> Dict[str, Any]:
        """
        Rank historical projects against candidate proposal, exclude self-matching,
        and generate explainable reports for the top K similar prior projects.
        """
        # 1. Compute candidate similarities across all corpus projects
        sim_df = self.tfidf_extractor.compute_candidate_similarity(candidate_dict)

        # Merge additional metadata from projects_df if needed
        merged_df = pd.merge(
            sim_df,
            self.projects_df[["project_id", "project_title", "domain"]],
            on="project_id",
            how="left"
        )

        # 2. Compute model predictions across all projects
        X = merged_df[self.feature_names].to_numpy(dtype=float)
        raw_preds = self.intercept_ + (X @ self.coef_)
        clipped_preds = np.clip(raw_preds, 0.0, 4.0)

        merged_df["predicted_similarity"] = clipped_preds
        merged_df["raw_prediction"] = raw_preds

        # 3. Exclude self-matching
        eligible_df = merged_df.copy()
        candidate_id = candidate_dict.get("project_id")

        if exclude_self:
            exclude_mask = pd.Series(False, index=eligible_df.index)

            # Exclude by project ID match (case-insensitive string comparison)
            if candidate_id:
                cand_id_str = str(candidate_id).strip().lower()
                exclude_mask = exclude_mask | (eligible_df["project_id"].astype(str).str.strip().str.lower() == cand_id_str)

            # Exclude by exact title match if candidate title is non-empty
            cand_title = str(candidate_dict.get("project_title", "")).strip().lower()
            if cand_title and "project_title" in self.projects_df.columns:
                hist_titles = self.projects_df.set_index("project_id").loc[eligible_df["project_id"], "project_title"].fillna("").astype(str).str.strip().str.lower()
                exclude_mask = exclude_mask | (hist_titles.values == cand_title)

            # Exclude if all section similarity features are 1.0 (or nearly 1.0 within 1e-4)
            sim_cols = [f"{sec}_similarity" for sec in self.section_names]
            all_ones = (eligible_df[sim_cols] >= 0.9999).all(axis=1)
            exclude_mask = exclude_mask | all_ones

            eligible_df = eligible_df[~exclude_mask]

        if len(eligible_df) == 0:
            raise ValueError("No eligible historical projects available for comparison after self-exclusion.")

        # 4. Deterministic ranking: primary by predicted_similarity (descending),
        # secondary by raw_prediction (descending), tertiary by project_id (ascending)
        ranked_df = eligible_df.sort_values(
            by=["predicted_similarity", "raw_prediction", "project_id"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        top_k_actual = min(top_k, len(ranked_df))
        top_matches_df = ranked_df.iloc[:top_k_actual]

        # 5. Overall Novelty Assessment
        max_sim = float(ranked_df.iloc[0]["predicted_similarity"])
        overall_novelty_pct = calculate_novelty_score(max_sim)
        novelty_tier = get_novelty_level(overall_novelty_pct)

        # 6. Generate detailed explanations for each top match
        match_explanations = []
        for _, row in top_matches_df.iterrows():
            hist_proj_row = self.projects_df[self.projects_df["project_id"] == row["project_id"]].iloc[0]
            exp = self.explain_pair(candidate_dict, hist_proj_row)
            match_explanations.append(exp)

        return {
            "candidate_proposal_title": candidate_dict.get("project_title", "N/A"),
            "candidate_project_id": candidate_dict.get("project_id", "NEW_PROPOSAL"),
            "corpus_size_evaluated": len(eligible_df),
            "max_similarity_score": round(max_sim, 4),
            "overall_novelty_score_pct": round(overall_novelty_pct, 2),
            "novelty_assessment_tier": novelty_tier,
            "top_k_requested": top_k,
            "top_k_returned": top_k_actual,
            "top_matches": match_explanations,
            "framework_disclaimers": {
                "corpus_relativity": (
                    "Novelty score is an estimated heuristic relative to the current database, "
                    "not proof of absolute academic originality, and does not prove plagiarism."
                ),
                "threshold_notice": (
                    "Novelty assessment tiers are heuristic benchmark rules relative to the indexed database "
                    "and require faculty panel review and domain-specific validation."
                ),
                "synthetic_ground_truth": (
                    "All benchmark similarity ratings in this system originate from synthetic heuristic labels, "
                    "not verified faculty adjudications."
                ),
                "non_causal_attribution": (
                    "Linear model feature attributions represent statistical associations rather than causal evidence."
                )
            }
        }
