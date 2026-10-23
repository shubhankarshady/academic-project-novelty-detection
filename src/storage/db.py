"""
Database storage layer for Academic Project Novelty Detection.
Manages storage, retrieval, and deduplication of submitted project proposals
and their XAI evaluation reports using SQLite.
"""

from typing import Dict, List, Any, Optional, Union
import sqlite3
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = ROOT_DIR / "data" / "proposals.db"

# The 8 academic proposal text sections
SECTION_FIELDS = [
    "project_title",
    "abstract",
    "problem_statement",
    "objectives",
    "proposed_methodology",
    "dataset",
    "algorithm_model",
    "technology_tools"
]

FIELD_ALIASES = {
    "title": "project_title",
    "problem": "problem_statement",
    "methodology": "proposed_methodology",
    "algorithm": "algorithm_model",
    "technology": "technology_tools"
}

class StorageError(Exception):
    """Base exception for storage operations."""
    pass

class DuplicateProposalError(StorageError):
    """Raised when a submitted proposal conflicts with an existing proposal."""
    def __init__(self, message: str, existing_project_id: str, existing_title: str, reason: str):
        super().__init__(message)
        self.existing_project_id = existing_project_id
        self.existing_title = existing_title
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "DuplicateProposalError",
            "message": str(self),
            "existing_project_id": self.existing_project_id,
            "existing_title": self.existing_title,
            "reason": self.reason
        }

def normalize_text(text: Optional[str]) -> str:
    """Normalize text by lowercasing, trimming, and collapsing whitespace."""
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())

def normalize_title(title: Optional[str]) -> str:
    """Normalize title for alphanumeric exact comparison."""
    clean = normalize_text(title)
    return re.sub(r"[^a-z0-9]", "", clean)

def extract_section_value(proposal_dict: Dict[str, Any], field: str) -> str:
    """Retrieve section value supporting both canonical and alias keys."""
    val = proposal_dict.get(field)
    if val is not None:
        return str(val)
    # Check aliases
    for alias, canonical in FIELD_ALIASES.items():
        if canonical == field and alias in proposal_dict:
            return str(proposal_dict[alias])
    return ""

def compute_content_hash(proposal_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash across all 8 normalized sections."""
    normalized_parts = [
        normalize_text(extract_section_value(proposal_dict, field))
        for field in SECTION_FIELDS
    ]
    combined = "||".join(normalized_parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

class DatabaseManager:
    """
    Manages SQLite storage for submitted academic proposals and XAI evaluation reports.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self._is_memory = str(db_path) == ":memory:"
        self._mem_conn = None
        if self._is_memory:
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite database connection."""
        if self._is_memory:
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", timeout=15.0, check_same_thread=False)
                self._mem_conn.row_factory = sqlite3.Row
                self._mem_conn.execute("PRAGMA foreign_keys = ON;")
            return self._mem_conn

        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def close(self) -> None:
        """Close database resources if active."""
        if self._mem_conn is not None:
            try:
                self._mem_conn.close()
            except Exception:
                pass
            self._mem_conn = None

    def init_db(self) -> None:
        """Initialize the database schema if tables do not exist."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS submitted_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT UNIQUE NOT NULL,
                    project_title TEXT NOT NULL,
                    normalized_title TEXT NOT NULL,
                    domain TEXT DEFAULT 'General',
                    abstract TEXT NOT NULL,
                    problem_statement TEXT NOT NULL,
                    objectives TEXT NOT NULL,
                    proposed_methodology TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    algorithm_model TEXT NOT NULL,
                    technology_tools TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    max_similarity_score REAL NOT NULL,
                    raw_similarity_score REAL NOT NULL,
                    novelty_score_pct REAL NOT NULL,
                    novelty_assessment_tier TEXT NOT NULL,
                    top_match_project_id TEXT,
                    top_match_title TEXT,
                    evaluation_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proposals_project_id ON submitted_proposals(project_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proposals_norm_title ON submitted_proposals(normalized_title);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proposals_content_hash ON submitted_proposals(content_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proposals_created ON submitted_proposals(created_at);")
            conn.commit()

    def check_duplicate(self, proposal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if a proposal duplicates an existing record based on:
        1. Exact project_id match (if a custom ID is specified)
        2. Exact content_hash match (identical 8 sections)
        3. Normalized title match (when title has >= 4 alphanumeric chars)
        
        Returns None if unique, or a dict describing the duplicate if found.
        """
        content_hash = compute_content_hash(proposal_data)
        title = extract_section_value(proposal_data, "project_title")
        norm_title = normalize_title(title)
        custom_id = str(proposal_data.get("project_id", "")).strip()

        with self.get_connection() as conn:
            # 1. Content Hash Check (exact content match)
            cursor = conn.execute(
                "SELECT project_id, project_title FROM submitted_proposals WHERE content_hash = ? LIMIT 1;",
                (content_hash,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "is_duplicate": True,
                    "reason": "Identical proposal text sections already exist in the database.",
                    "existing_project_id": row["project_id"],
                    "existing_title": row["project_title"]
                }

            # 2. Normalized Title Check
            if len(norm_title) >= 4:
                cursor = conn.execute(
                    "SELECT project_id, project_title FROM submitted_proposals WHERE normalized_title = ? LIMIT 1;",
                    (norm_title,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "is_duplicate": True,
                        "reason": f"A proposal with an identical normalized title ('{row['project_title']}') already exists.",
                        "existing_project_id": row["project_id"],
                        "existing_title": row["project_title"]
                    }

            # 3. Explicit Project ID collision (if specified and not 'NEW_PROPOSAL')
            if custom_id and custom_id.upper() not in {"NEW_PROPOSAL", "UNKNOWN", "NONE", ""}:
                cursor = conn.execute(
                    "SELECT project_id, project_title FROM submitted_proposals WHERE UPPER(project_id) = UPPER(?) LIMIT 1;",
                    (custom_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "is_duplicate": True,
                        "reason": f"Project ID '{custom_id}' is already assigned to an existing proposal.",
                        "existing_project_id": row["project_id"],
                        "existing_title": row["project_title"]
                    }

        return None

    def _generate_next_project_id(self, conn: sqlite3.Connection) -> str:
        """Generate the next sequential project ID (e.g. SUB-0001, SUB-0002)."""
        cursor = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM submitted_proposals;")
        next_num = cursor.fetchone()["next_id"]
        candidate_id = f"SUB-{next_num:04d}"
        
        # Ensure candidate_id is not already taken
        cursor = conn.execute("SELECT 1 FROM submitted_proposals WHERE project_id = ?;", (candidate_id,))
        while cursor.fetchone() is not None:
            next_num += 1
            candidate_id = f"SUB-{next_num:04d}"
            cursor = conn.execute("SELECT 1 FROM submitted_proposals WHERE project_id = ?;", (candidate_id,))
        return candidate_id

    def save_proposal(
        self,
        proposal_data: Dict[str, Any],
        evaluation_report: Dict[str, Any]
    ) -> str:
        """
        Save a proposal and its XAI evaluation report into SQLite.
        
        Args:
            proposal_data: Dictionary containing the 8 proposal sections and optional domain/title.
            evaluation_report: Structured dictionary output from NoveltyExplainer.
            
        Returns:
            The assigned unique project_id.
            
        Raises:
            ValueError: If required sections or reports are missing.
            DuplicateProposalError: If the proposal duplicates an existing record.
            StorageError: If database operations fail.
        """
        if not proposal_data or not isinstance(proposal_data, dict):
            raise ValueError("proposal_data must be a non-empty dictionary.")
        if not evaluation_report or not isinstance(evaluation_report, dict):
            raise ValueError("evaluation_report must be a non-empty dictionary.")

        # Check for duplicates before transaction
        dup_info = self.check_duplicate(proposal_data)
        if dup_info:
            raise DuplicateProposalError(
                message=dup_info["reason"],
                existing_project_id=dup_info["existing_project_id"],
                existing_title=dup_info["existing_title"],
                reason=dup_info["reason"]
            )

        title = extract_section_value(proposal_data, "project_title").strip()
        if not title:
            raise ValueError("Proposal must have a non-empty project_title.")

        norm_title = normalize_title(title)
        content_hash = compute_content_hash(proposal_data)
        domain = str(proposal_data.get("domain", "General")).strip() or "General"

        # Extract metric summaries from evaluation report
        novelty_score_pct = float(
            evaluation_report.get("overall_novelty_score_pct",
            evaluation_report.get("novelty_score_pct", 0.0))
        )
        novelty_tier = str(
            evaluation_report.get("novelty_assessment_tier",
            evaluation_report.get("novelty_category", "N/A"))
        )
        max_sim = float(
            evaluation_report.get("max_similarity_score",
            evaluation_report.get("predicted_similarity", 0.0))
        )
        raw_sim = float(
            evaluation_report.get("raw_similarity_prediction", max_sim)
        )

        top_matches = evaluation_report.get("top_matches", [])
        if top_matches and isinstance(top_matches, list):
            top_match_id = str(top_matches[0].get("historical_project_id", "N/A"))
            top_match_title = str(top_matches[0].get("historical_project_title", "N/A"))
        else:
            top_match_id = str(evaluation_report.get("historical_project_id", "N/A"))
            top_match_title = str(evaluation_report.get("historical_project_title", "N/A"))

        # Serialize evaluation report
        try:
            eval_json_str = json.dumps(evaluation_report, ensure_ascii=False)
        except Exception as e:
            raise StorageError(f"Failed to serialize evaluation report to JSON: {e}") from e

        conn = self.get_connection()
        try:
            with conn:
                # Assign project ID if not specified
                custom_id = str(proposal_data.get("project_id", "")).strip()
                if custom_id and custom_id.upper() not in {"NEW_PROPOSAL", "UNKNOWN", "NONE", ""}:
                    project_id = custom_id
                else:
                    project_id = self._generate_next_project_id(conn)

                cursor = conn.execute("""
                    INSERT INTO submitted_proposals (
                        project_id,
                        project_title,
                        normalized_title,
                        domain,
                        abstract,
                        problem_statement,
                        objectives,
                        proposed_methodology,
                        dataset,
                        algorithm_model,
                        technology_tools,
                        content_hash,
                        max_similarity_score,
                        raw_similarity_score,
                        novelty_score_pct,
                        novelty_assessment_tier,
                        top_match_project_id,
                        top_match_title,
                        evaluation_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    project_id,
                    title,
                    norm_title,
                    domain,
                    extract_section_value(proposal_data, "abstract"),
                    extract_section_value(proposal_data, "problem_statement"),
                    extract_section_value(proposal_data, "objectives"),
                    extract_section_value(proposal_data, "proposed_methodology"),
                    extract_section_value(proposal_data, "dataset"),
                    extract_section_value(proposal_data, "algorithm_model"),
                    extract_section_value(proposal_data, "technology_tools"),
                    content_hash,
                    round(max_sim, 4),
                    round(raw_sim, 4),
                    round(novelty_score_pct, 2),
                    novelty_tier,
                    top_match_id,
                    top_match_title,
                    eval_json_str
                ))
            return project_id
        except sqlite3.IntegrityError as e:
            conn.rollback()
            # Handle race condition duplicate if caught by DB constraints
            err_msg = str(e).lower()
            if "content_hash" in err_msg:
                raise DuplicateProposalError(
                    message="Identical proposal text sections already exist in the database.",
                    existing_project_id="UNKNOWN",
                    existing_title=title,
                    reason="Database content_hash constraint violation."
                ) from e
            elif "project_id" in err_msg:
                raise DuplicateProposalError(
                    message=f"Project ID '{project_id}' already exists.",
                    existing_project_id=project_id,
                    existing_title=title,
                    reason="Database project_id unique constraint violation."
                ) from e
            else:
                raise StorageError(f"Database integrity error: {e}") from e
        except Exception as e:
            conn.rollback()
            raise StorageError(f"Failed to save proposal: {e}") from e
        finally:
            if not self._is_memory:
                conn.close()

    def list_proposals(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List saved proposals ordered by newest first.
        Returns summary metadata without the large evaluation_json payload.
        """
        query = """
            SELECT
                id,
                project_id,
                project_title,
                domain,
                max_similarity_score,
                raw_similarity_score,
                novelty_score_pct,
                novelty_assessment_tier,
                top_match_project_id,
                top_match_title,
                created_at
            FROM submitted_proposals
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?;
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (limit, offset))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def count_proposals(self) -> int:
        """Return total number of submitted proposals in the database."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) AS total FROM submitted_proposals;")
            return int(cursor.fetchone()["total"])

    def get_proposal_by_id(self, project_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single proposal by project_id or numeric primary key id,
        including all sections, metrics, and deserialized evaluation report.
        """
        if not project_id:
            return None

        query = """
            SELECT * FROM submitted_proposals
            WHERE UPPER(project_id) = UPPER(?) OR id = ?
            LIMIT 1;
        """
        str_id = str(project_id).strip()
        num_id = int(str_id) if str_id.isdigit() else -1

        with self.get_connection() as conn:
            cursor = conn.execute(query, (str_id, num_id))
            row = cursor.fetchone()
            if not row:
                return None

            result = dict(row)
            # Parse evaluation_json into Python dict
            try:
                result["evaluation_report"] = json.loads(result.pop("evaluation_json", "{}"))
            except Exception:
                result["evaluation_report"] = {}

            return result


# Module-level convenience functions using default database
_default_manager: Optional[DatabaseManager] = None

def get_default_manager(db_path: Optional[Union[str, Path]] = None) -> DatabaseManager:
    """Retrieve or create singleton DatabaseManager instance."""
    global _default_manager
    if db_path is not None:
        return DatabaseManager(db_path)
    if _default_manager is None:
        _default_manager = DatabaseManager(DEFAULT_DB_PATH)
    return _default_manager

def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize the SQLite database."""
    manager = get_default_manager(db_path)
    manager.init_db()

def save_proposal(
    proposal_data: Dict[str, Any],
    evaluation_report: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None
) -> str:
    """Save a proposal and its evaluation report into the database."""
    manager = get_default_manager(db_path)
    return manager.save_proposal(proposal_data, evaluation_report)

def list_proposals(
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """List saved proposals with summary information."""
    manager = get_default_manager(db_path)
    return manager.list_proposals(limit=limit, offset=offset)

def get_proposal_by_id(
    project_id: Union[str, int],
    db_path: Optional[Union[str, Path]] = None
) -> Optional[Dict[str, Any]]:
    """Retrieve a proposal and its full evaluation report by ID."""
    manager = get_default_manager(db_path)
    return manager.get_proposal_by_id(project_id)

def check_duplicate(
    proposal_data: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None
) -> Optional[Dict[str, Any]]:
    """Check if a proposal duplicates an existing record."""
    manager = get_default_manager(db_path)
    return manager.check_duplicate(proposal_data)
