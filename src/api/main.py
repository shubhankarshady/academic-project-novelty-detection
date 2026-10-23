"""
FastAPI Backend Application for Academic Project Novelty Detection.
Provides endpoints for health monitoring, candidate proposal evaluation,
deferred non-leaking proposal saving, and historical submission audit retrieval.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    ProposalInput,
    HealthResponse,
    ProposalEvaluationResponse,
    SaveProposalRequest,
    SaveProposalResponse,
    ProposalListResponse,
    ProposalSummaryItem,
)
from src.api.service import NoveltyService, get_novelty_service
from src.storage.db import (
    DatabaseManager,
    get_default_manager,
    DuplicateProposalError,
    StorageError,
    normalize_title,
)


def get_db() -> DatabaseManager:
    """FastAPI dependency for database manager (can be overridden in tests)."""
    return get_default_manager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load machine learning models once at application startup."""
    try:
        get_novelty_service()
    except Exception as e:
        # Allow server to start even if models need lazy loading
        pass
    yield


app = FastAPI(
    title="Academic Project Novelty Detection API",
    description="Explainable AI API for scoring academic project proposals against historical reference corpora.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DuplicateProposalError)
async def duplicate_proposal_exception_handler(request, exc: DuplicateProposalError):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=exc.to_dict()
    )


@app.exception_handler(StorageError)
async def storage_error_exception_handler(request, exc: StorageError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "StorageError", "message": "A database storage error occurred."}
    )


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def get_health(
    service: NoveltyService = Depends(get_novelty_service),
    db: DatabaseManager = Depends(get_db)
):
    """
    Check API health and report reference corpus size vs. submitted user proposals count.
    """
    ref_count = service.reference_corpus_size
    sub_count = db.count_proposals()
    return HealthResponse(
        status="healthy",
        reference_corpus_size=ref_count,
        submitted_proposals_count=sub_count,
        model_loaded=True,
        message=(
            f"Active. Reference corpus contains {ref_count} baseline historical projects. "
            f"Database contains {sub_count} user-submitted proposals."
        )
    )


@app.post("/api/score", response_model=ProposalEvaluationResponse, tags=["Evaluation"])
def score_proposal(
    proposal: ProposalInput,
    service: NoveltyService = Depends(get_novelty_service)
):
    """
    Evaluate a candidate academic proposal against the historical reference corpus.
    Does NOT save the proposal to SQLite or mutate the reference corpus.
    Candidate is strictly excluded from matching itself.
    """
    try:
        candidate_dict = proposal.model_dump()
        result = service.score_proposal(candidate_dict=candidate_dict, top_k=5)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during proposal evaluation: {e}"
        )


@app.post("/api/save", response_model=SaveProposalResponse, status_code=status.HTTP_201_CREATED, tags=["Proposals"])
def save_proposal(
    request: SaveProposalRequest,
    db: DatabaseManager = Depends(get_db)
):
    """
    Save a previously evaluated proposal and its verified XAI report into SQLite.
    Verifies that the evaluation report matches the candidate proposal title.
    Rejects duplicate records with HTTP 409 Conflict without silent overwrites.
    """
    proposal_data = request.proposal.model_dump()
    evaluation_report = request.evaluation_report

    # Content verification: report candidate title must match proposal project_title
    report_title = evaluation_report.get("candidate_proposal_title", "")
    if normalize_title(proposal_data["project_title"]) != normalize_title(report_title):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Evaluation report does not match the submitted proposal content. "
                f"Proposal title is '{proposal_data['project_title']}', but report is for '{report_title}'."
            )
        )

    # Verification of evaluation report essential fields
    required_eval_keys = ["max_similarity_score", "novelty_assessment_tier", "top_matches"]
    for key in required_eval_keys:
        if key not in evaluation_report:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluation report is incomplete: missing required field '{key}'."
            )

    try:
        project_id = db.save_proposal(
            proposal_data=proposal_data,
            evaluation_report=evaluation_report
        )
        saved_record = db.get_proposal_by_id(project_id)
        if not saved_record:
            raise HTTPException(status_code=500, detail="Proposal saved but could not be retrieved.")

        return SaveProposalResponse(
            status="saved",
            project_id=project_id,
            project_title=saved_record["project_title"],
            novelty_score_pct=saved_record["novelty_score_pct"],
            novelty_assessment_tier=saved_record["novelty_assessment_tier"],
            created_at=str(saved_record["created_at"]),
            message=f"Proposal '{saved_record['project_title']}' successfully saved as {project_id}."
        )
    except DuplicateProposalError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.to_dict()
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )


@app.get("/api/proposals", response_model=ProposalListResponse, tags=["Proposals"])
def list_proposals(
    limit: int = Query(50, ge=1, le=200, description="Number of proposals to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: DatabaseManager = Depends(get_db)
):
    """
    List previously submitted proposals, newest first, with pagination and summary metrics.
    """
    rows = db.list_proposals(limit=limit, offset=offset)
    total = db.count_proposals()

    summary_items = [
        ProposalSummaryItem(
            id=row["id"],
            project_id=row["project_id"],
            project_title=row["project_title"],
            domain=row.get("domain") or "General",
            max_similarity_score=row["max_similarity_score"],
            raw_similarity_score=row["raw_similarity_score"],
            novelty_score_pct=row["novelty_score_pct"],
            novelty_assessment_tier=row["novelty_assessment_tier"],
            top_match_project_id=row.get("top_match_project_id"),
            top_match_title=row.get("top_match_title"),
            created_at=str(row["created_at"])
        )
        for row in rows
    ]

    return ProposalListResponse(
        total=total,
        limit=limit,
        offset=offset,
        proposals=summary_items
    )


@app.get("/api/proposals/{project_id}", tags=["Proposals"])
def get_proposal_by_id(
    project_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Retrieve one saved proposal and its complete XAI evaluation report.
    Returns HTTP 404 if not found.
    """
    record = db.get_proposal_by_id(project_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Proposal with ID '{project_id}' not found."
        )
    return record
