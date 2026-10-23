"""
API package for Academic Project Novelty Detection.
Exposes the FastAPI application and service dependencies.
"""

from src.api.main import app, get_db
from src.api.service import NoveltyService, get_novelty_service
from src.api.schemas import (
    ProposalInput,
    HealthResponse,
    ProposalEvaluationResponse,
    SaveProposalRequest,
    SaveProposalResponse,
    ProposalListResponse,
)

__all__ = [
    "app",
    "get_db",
    "NoveltyService",
    "get_novelty_service",
    "ProposalInput",
    "HealthResponse",
    "ProposalEvaluationResponse",
    "SaveProposalRequest",
    "SaveProposalResponse",
    "ProposalListResponse",
]
