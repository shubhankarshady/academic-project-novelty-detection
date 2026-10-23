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
    model_weight: float
    signed_contribution: float


class SectionDifferenceSummary(BaseModel):
    high_overlap_sections: List[str] = Field(default_factory=list, description="Sections with similarity >= 0.30")
    differentiating_sections: List[str] = Field(default_factory=list, description="Sections with similarity < 0.15")


class TopMatchItem(BaseModel):
    historical_project_id: str
    historical_project_title: str
    historical_domain: str
    predicted_similarity: float
    raw_similarity_prediction: float
    clipped_prediction: float
    novelty_score_pct: float
    novelty_category: str
    intercept: float
    sum_of_contributions: float
    reconstructed_raw_prediction: float
    section_similarity_scores: Dict[str, float]
    section_signed_contributions: Dict[str, float]
    ranked_sections: List[RankedSectionItem]
    section_evidence: Dict[str, SectionEvidenceItem]
    difference_summary: SectionDifferenceSummary
    disclaimers: Dict[str, str]


class ProposalEvaluationResponse(BaseModel):
    """Complete Explainable AI evaluation payload returned by /api/score."""
    candidate_proposal_title: str
    candidate_project_id: Optional[str] = Field("NEW_PROPOSAL", description="Candidate proposal identifier")
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
