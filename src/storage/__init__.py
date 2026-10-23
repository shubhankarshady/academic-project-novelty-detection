"""
Storage package for Academic Project Novelty Detection.
Provides database access and deduplication for submitted proposals.
"""

from src.storage.db import (
    DatabaseManager,
    init_db,
    save_proposal,
    list_proposals,
    get_proposal_by_id,
    check_duplicate,
    DuplicateProposalError,
    StorageError,
    DEFAULT_DB_PATH,
)

__all__ = [
    "DatabaseManager",
    "init_db",
    "save_proposal",
    "list_proposals",
    "get_proposal_by_id",
    "check_duplicate",
    "DuplicateProposalError",
    "StorageError",
    "DEFAULT_DB_PATH",
]
