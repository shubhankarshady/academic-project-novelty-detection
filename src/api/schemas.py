"""
Pydantic schemas for the Academic Project Novelty Detection API.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator


class ProposalInput(BaseModel):
    """Input payload containing the 8 required academic proposal text sections."""
    project_title: str = Field(..., description="Academic project title", min_length=3)
    domain: Optional[str] = Field("General", description="Academic domain or department")
    abstract: str = Field(..., description="Project abstract summary", min_length=10)
    problem_statement: str = Field(..., description="Problem statement and motivation", min_length=10)
    objectives: str = Field(..., description="Specific project objectives", min_length=5)
    proposed_methodology: str = Field(..., description="Proposed technical methodology", min_length=10)
    dataset: str = Field(..., description="Dataset and benchmark resources used", min_length=3)
    algorithm_model: str = Field(..., description="Algorithms, models, and architectures", min_length=3)
    technology_tools: str = Field(..., description="Technologies, libraries, and tools", min_length=2)
    project_id: Optional[str] = Field(None, description="Optional custom proposal identifier")

    @field_validator(
        "project_title",
        "abstract",
        "problem_statement",
        "objectives",
        "proposed_methodology",
        "dataset",
        "algorithm_model",
        "technology_tools",
        mode="before"
    )
    @classmethod
    def strip_and_validate_non_empty(cls, v: Any) -> str:
        if v is None:
            raise ValueError("Section content cannot be null.")
        val_str = str(v).strip()
        if not val_str:
            raise ValueError("Section content cannot be empty or whitespace only.")
        return val_str


class HealthResponse(BaseModel):
    """System health check and corpus size statistics."""
    status: str = Field("healthy", description="Application status")
    reference_corpus_size: int = Field(..., description="Number of baseline historical projects in reference corpus")
    submitted_proposals_count: int = Field(..., description="Number of proposals stored in the submitted database")
    model_loaded: bool = Field(True, description="Whether ML and XAI models are loaded in memory")
    active_scoring_mode: Optional[str] = Field("tfidf", description="Active scoring mode ('tfidf', 'hybrid', or 'tuned_hybrid')")
    hybrid_alpha: Optional[float] = Field(0.50, description="Configured hybrid alpha")
    available_scoring_modes: List[str] = Field(default_factory=lambda: ["tfidf", "hybrid", "tuned_hybrid"], description="Supported scoring modes")
    tuned_hybrid_config: Optional[Dict[str, Any]] = Field(None, description="Tuned hybrid configuration metadata if available")
    message: str = Field(..., description="Status summary distinguishing reference corpus from user submissions")


class TermEvidenceItem(BaseModel):
    term: str
    term_cosine_contribution: float
    candidate_snippet: str
    historical_snippet: str


class SectionEvidenceItem(BaseModel):
    similarity_score: float
    model_weight: float
    signed_contribution: float
    shared_terms_count: int
    shared_terms: List[TermEvidenceItem] = Field(default_factory=list)
    message: Optional[str] = None


class RankedSectionItem(BaseModel):
    section: str
    column: str
    raw_similarity: float
    lexical_similarity: Optional[float] = Field(None, description="Lexical TF-IDF cosine similarity")
    semantic_similarity: Optional[float] = Field(None, description="Semantic Dense cosine similarity")
    hybrid_similarity: Optional[float] = Field(None, description="Convex hybrid similarity")
    model_weight: float
    signed_contribution: float


class SectionDifferenceSummary(BaseModel):
    high_overlap_sections: List[str] = Field(default_factory=list, description="Sections with similarity >= 0.30")
    differentiating_sections: List[str] = Field(default_factory=list, description="Sections with similarity < 0.15")


class TopMatchItem(BaseModel):
    historical_project_id: str
    historical_project_title: str
    historical_domain: str
    scoring_mode: Optional[str] = Field("tfidf", description="Scoring mode for this match ('tfidf', 'hybrid', or 'tuned_hybrid')")
    hybrid_alpha: Optional[float] = Field(1.0, description="Hybrid weighting factor alpha")
    predicted_similarity: float
    raw_similarity_prediction: float
    clipped_prediction: float
    novelty_score_pct: float
    novelty_category: str
    intercept: float
    sum_of_contributions: float
    reconstructed_raw_prediction: float
    section_similarity_scores: Dict[str, float]
    section_lexical_scores: Optional[Dict[str, float]] = Field(None, description="Lexical similarities per section")
    section_semantic_scores: Optional[Dict[str, float]] = Field(None, description="Semantic similarities per section")
    section_hybrid_scores: Optional[Dict[str, float]] = Field(None, description="Hybrid similarities per section")
    section_signed_contributions: Dict[str, float]
    ranked_sections: List[RankedSectionItem]
    section_evidence: Dict[str, SectionEvidenceItem]
    difference_summary: SectionDifferenceSummary
    disclaimers: Dict[str, str]


class ProposalEvaluationResponse(BaseModel):
    """Complete Explainable AI evaluation payload returned by /api/score."""
    candidate_proposal_title: str
    candidate_project_id: Optional[str] = Field("NEW_PROPOSAL", description="Candidate proposal identifier")
    scoring_mode: Optional[str] = Field("tfidf", description="Scoring mode evaluated ('tfidf', 'hybrid', or 'tuned_hybrid')")
    hybrid_alpha: Optional[float] = Field(1.0, description="Hybrid weighting factor alpha")
    scoring_formula: Optional[str] = Field(None, description="Mathematical scoring formula applied")
    model_version: Optional[str] = Field(None, description="Model version or identifier")
    corpus_size_evaluated: int
    max_similarity_score: float
    overall_novelty_score_pct: float
    novelty_assessment_tier: str
    top_k_requested: int
    top_k_returned: int
    top_matches: List[TopMatchItem]
    framework_disclaimers: Dict[str, str]


class SaveProposalRequest(BaseModel):
    """Payload to save an evaluated proposal into the database."""
    proposal: ProposalInput
    evaluation_report: Dict[str, Any] = Field(..., description="Complete evaluation report returned from /api/score")


class SaveProposalResponse(BaseModel):
    """Response returned upon saving a proposal."""
    status: str = Field("saved", description="Save status")
    project_id: str = Field(..., description="Assigned unique project ID")
    project_title: str
    novelty_score_pct: float
    novelty_assessment_tier: str
    created_at: str
    message: str


class ProposalSummaryItem(BaseModel):
    """Summary item for listing historical user submissions."""
    id: int
    project_id: str
    project_title: str
    domain: str
    max_similarity_score: float
    raw_similarity_score: float
    novelty_score_pct: float
    novelty_assessment_tier: str
    top_match_project_id: Optional[str] = None
    top_match_title: Optional[str] = None
    created_at: str


class ProposalListResponse(BaseModel):
    """Paginated list of saved user proposals."""
    total: int
    limit: int
    offset: int
    proposals: List[ProposalSummaryItem]


# ============================================================================
# Groq-Powered Explainable AI (XAI) Schemas
# ============================================================================

class RecommendedImprovements(BaseModel):
    """Specific recommendations for each major proposal section."""
    objectives: str = Field(..., description="Actionable revision for project objectives")
    proposed_methodology: str = Field(..., description="Technical enhancement to methodology")
    dataset: str = Field(..., description="Recommendations for dataset collection, augmentation, or validation")
    algorithm_model: str = Field(..., description="Recommendations for algorithms, models, and baselines")

    @field_validator("proposed_methodology", mode="before")
    @classmethod
    def normalize_methodology(cls, v: Any, info) -> str:
        if isinstance(v, str) and v.strip():
            return v.strip()
        return "Refine baseline comparisons, specify training hyperparameters, and include quantitative ablation studies."

    @field_validator("algorithm_model", mode="before")
    @classmethod
    def normalize_algorithm(cls, v: Any, info) -> str:
        if isinstance(v, str) and v.strip():
            return v.strip()
        return "Introduce architectural enhancements, ensemble comparisons, and rigorous loss formulations."


class SectionOverlapItem(BaseModel):
    """Explanation of conceptual and lexical overlap for a specific section."""
    section: str = Field(..., description="Section name")
    overlap_level: str = Field("Moderate", description="Overlap level: High, Moderate, or Low")
    analysis: str = Field(..., description="Technical explanation of the overlap")


class GroqExplanation(BaseModel):
    """Structured 9-part academic Explainable AI response produced by Groq."""
    overall_novelty_interpretation: str = Field(..., description="Plain-language academic interpretation of novelty score and tier")
    what_is_missing: List[str] = Field(..., description="Specific elements missing in the proposal")
    why_similar: str = Field(..., description="Technical comparison explaining similarity to retrieved reference projects")
    section_overlap_explanation: List[SectionOverlapItem] = Field(..., description="Detailed section-level overlap explanations")
    potential_research_gaps: List[str] = Field(..., description="Identified unaddressed research challenges in the domain")
    specific_suggestions: List[str] = Field(..., description="Concrete, actionable recommendations to improve novelty")
    potential_novel_directions: List[str] = Field(..., description="Promising novel research directions and paradigms")
    recommended_improvements: RecommendedImprovements = Field(..., description="Recommended improvements across core technical sections")
    prioritized_action_plan: List[str] = Field(..., description="Step-by-step prioritized roadmap for the student")

    @field_validator("what_is_missing", "potential_research_gaps", "specific_suggestions", "potential_novel_directions", "prioritized_action_plan", mode="before")
    @classmethod
    def ensure_list_of_strings(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            items = []
            for item in v:
                if isinstance(item, str) and item.strip():
                    items.append(item.strip())
                elif isinstance(item, dict):
                    items.append(str(item))
            return items if items else ["Detailed analysis provided in technical sections."]
        elif isinstance(v, str) and v.strip():
            lines = [line.strip().lstrip("-*0123456789. ") for line in v.splitlines() if line.strip()]
            return lines if lines else [v.strip()]
        return ["Information available in full evaluation report."]

    @field_validator("recommended_improvements", mode="before")
    @classmethod
    def normalize_improvements(cls, v: Any) -> Dict[str, str]:
        if isinstance(v, dict):
            return {
                "objectives": v.get("objectives", "Clarify measurable project objectives and specific novelty targets."),
                "proposed_methodology": v.get("proposed_methodology") or v.get("methodology") or "Detail procedural pipeline and validation benchmarks.",
                "dataset": v.get("dataset", "Specify data source, sample size, train/test split, and augmentation strategy."),
                "algorithm_model": v.get("algorithm_model") or v.get("algorithm") or v.get("model") or "Detail model architectures, loss functions, and evaluation metrics."
            }
        return {
            "objectives": "Clarify measurable project objectives and specific novelty targets.",
            "proposed_methodology": "Detail procedural pipeline and validation benchmarks.",
            "dataset": "Specify data source, sample size, train/test split, and augmentation strategy.",
            "algorithm_model": "Detail model architectures, loss functions, and evaluation metrics."
        }

    @field_validator("section_overlap_explanation", mode="before")
    @classmethod
    def normalize_section_overlap(cls, v: Any) -> List[Dict[str, Any]]:
        if isinstance(v, list):
            out = []
            for item in v:
                if isinstance(item, dict):
                    out.append({
                        "section": item.get("section", "General Section"),
                        "overlap_level": item.get("overlap_level", "Moderate"),
                        "analysis": item.get("analysis", "Moderate conceptual alignment with reference project.")
                    })
                elif isinstance(item, str):
                    out.append({
                        "section": "Section Overlap",
                        "overlap_level": "Moderate",
                        "analysis": item
                    })
            return out if out else [{"section": "Methodology", "overlap_level": "Moderate", "analysis": "Shared machine learning pipeline concepts."}]
        elif isinstance(v, str):
            return [{"section": "Overall Overlap", "overlap_level": "Moderate", "analysis": v.strip()}]
        return [{"section": "Methodology", "overlap_level": "Moderate", "analysis": "Shared machine learning pipeline concepts."}]


class ExplainProposalRequest(BaseModel):
    """Payload to request an on-demand Groq AI explanation for an evaluated proposal."""
    proposal: ProposalInput
    evaluation_report: Dict[str, Any] = Field(..., description="Evaluation report previously returned by /api/score")
    force_regenerate: Optional[bool] = Field(False, description="Whether to bypass backend cache and call Groq again")


class ExplainProposalResponse(BaseModel):
    """Response containing on-demand Groq-powered Explainable AI analysis."""
    status: str = Field("success", description="Status of the XAI explanation")
    scoring_mode: str = Field(..., description="Scoring mode of the evaluation ('tfidf', 'hybrid', or 'tuned_hybrid')")
    hybrid_alpha: Optional[float] = Field(1.0, description="Hybrid alpha parameter used during evaluation")
    overall_novelty_score_pct: float = Field(..., description="Novelty percentage calculated by deterministic NLP model")
    novelty_assessment_tier: str = Field(..., description="Novelty assessment tier")
    top_match_id: Optional[str] = Field(None, description="Identifier of the top matching reference project")
    top_match_title: Optional[str] = Field(None, description="Title of the top matching reference project")
    model_used: str = Field(..., description="Groq model invoked (e.g. llama-3.3-70b-versatile)")
    cached: bool = Field(False, description="Whether this response was served from memory cache")
    created_at: str = Field(..., description="ISO timestamp of explanation generation")
    explanation: GroqExplanation = Field(..., description="Structured 9-part explanation")

