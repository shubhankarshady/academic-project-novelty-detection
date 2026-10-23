"""
Unit and Integration Tests for SQLite Storage Layer (src/storage/db.py).

Tests:
1. Database initialization and table creation (file and in-memory).
2. Proposal save and full retrieval (verifying all 8 sections, metrics, and JSON report).
3. Sequential project ID auto-generation (SUB-0001, SUB-0002).
4. List proposals ordering (newest first) and pagination (limit/offset).
5. Duplicate prevention:
   - Content hash collision (exact 8 sections match).
   - Normalized title collision (case/punctuation normalized).
   - Custom Project ID collision.
   - Non-raising check_duplicate() inspection.
6. Error handling and transaction rollback (ensuring no corrupt/partial records).
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path

from src.storage.db import (
    DatabaseManager,
    DuplicateProposalError,
    StorageError,
    compute_content_hash,
    normalize_title,
    normalize_text,
    SECTION_FIELDS
)

class TestStorageLayer(unittest.TestCase):

    def setUp(self):
        """Create a clean temporary directory and database instance for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_proposals.db"
        self.db = DatabaseManager(self.db_path)

        # Standard sample proposal
        self.sample_proposal = {
            "project_title": "Deep Learning for Autonomous Drone Navigation",
            "domain": "Robotics & Autonomous Systems",
            "abstract": "We present a deep reinforcement learning framework for quadrotor indoor path planning.",
            "problem_statement": "GPS-denied environments lead to frequent drone collisions.",
            "objectives": "Achieve collision-free real-time navigation in obstacle-dense spaces.",
            "proposed_methodology": "Simulate quadrotor dynamics in Gazebo with PPO policy optimization.",
            "dataset": "Gazebo point cloud simulated telemetry.",
            "algorithm_model": "Proximal Policy Optimization (PPO).",
            "technology_tools": "ROS, Gazebo, PyTorch, Python, CUDA."
        }

        # Standard sample XAI evaluation report
        self.sample_evaluation = {
            "candidate_proposal_title": "Deep Learning for Autonomous Drone Navigation",
            "corpus_size_evaluated": 600,
            "max_similarity_score": 0.45,
            "raw_similarity_prediction": 0.4489,
            "overall_novelty_score_pct": 88.75,
            "novelty_assessment_tier": "High estimated novelty relative to the current database",
            "top_matches": [
                {
                    "historical_project_id": "SYN-0123",
                    "historical_project_title": "Reinforcement Learning for Quadrotors",
                    "historical_domain": "Robotics",
                    "predicted_similarity": 0.45,
                    "novelty_score_pct": 88.75
                }
            ],
            "framework_disclaimers": {
                "corpus_relativity": "Novelty score is an estimated heuristic relative to the current database."
            }
        }

    def tearDown(self):
        """Clean up temporary directory and files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initialization(self):
        """Verify database initializes properly and creates required schema and indexes."""
        self.assertTrue(self.db_path.exists())

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='submitted_proposals';"
            )
            table_row = cursor.fetchone()
            self.assertIsNotNone(table_row)
            self.assertEqual(table_row["name"], "submitted_proposals")

            # Check indexes
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index';")
            index_names = [row["name"] for row in cursor.fetchall()]
            self.assertIn("idx_proposals_project_id", index_names)
            self.assertIn("idx_proposals_norm_title", index_names)
            self.assertIn("idx_proposals_content_hash", index_names)
            self.assertIn("idx_proposals_created", index_names)

    def test_in_memory_database(self):
        """Verify DatabaseManager supports in-memory ':memory:' mode."""
        mem_db = DatabaseManager(":memory:")
        pid = mem_db.save_proposal(self.sample_proposal, self.sample_evaluation)
        self.assertEqual(pid, "SUB-0001")
        record = mem_db.get_proposal_by_id(pid)
        self.assertIsNotNone(record)
        self.assertEqual(record["project_title"], self.sample_proposal["project_title"])

    def test_save_and_retrieve_proposal(self):
        """Verify saving a proposal and retrieving it restores all 8 sections and the JSON report."""
        assigned_id = self.db.save_proposal(self.sample_proposal, self.sample_evaluation)
        self.assertEqual(assigned_id, "SUB-0001")

        # Retrieve by project_id string
        retrieved = self.db.get_proposal_by_id("SUB-0001")
        self.assertIsNotNone(retrieved)

        # Check section fields
        for field in SECTION_FIELDS:
            self.assertEqual(
                retrieved[field],
                self.sample_proposal[field],
                f"Field '{field}' did not match saved value!"
            )

        # Check metrics
        self.assertEqual(retrieved["domain"], self.sample_proposal["domain"])
        self.assertEqual(retrieved["max_similarity_score"], 0.45)
        self.assertEqual(retrieved["novelty_score_pct"], 88.75)
        self.assertEqual(
            retrieved["novelty_assessment_tier"],
            "High estimated novelty relative to the current database"
        )
        self.assertEqual(retrieved["top_match_project_id"], "SYN-0123")
        self.assertEqual(retrieved["top_match_title"], "Reinforcement Learning for Quadrotors")

        # Check deserialized evaluation report
        eval_report = retrieved["evaluation_report"]
        self.assertIsInstance(eval_report, dict)
        self.assertEqual(eval_report["candidate_proposal_title"], self.sample_proposal["project_title"])
        self.assertIn("framework_disclaimers", eval_report)

        # Retrieve by numeric ID
        retrieved_by_num = self.db.get_proposal_by_id(1)
        self.assertIsNotNone(retrieved_by_num)
        self.assertEqual(retrieved_by_num["project_id"], "SUB-0001")

    def test_sequential_id_generation(self):
        """Verify sequential IDs are correctly generated (SUB-0001, SUB-0002, SUB-0003)."""
        prop_1 = dict(self.sample_proposal, project_title="Project One Title")
        prop_2 = dict(self.sample_proposal, project_title="Project Two Title")
        prop_3 = dict(self.sample_proposal, project_title="Project Three Title")

        id_1 = self.db.save_proposal(prop_1, self.sample_evaluation)
        id_2 = self.db.save_proposal(prop_2, self.sample_evaluation)
        id_3 = self.db.save_proposal(prop_3, self.sample_evaluation)

        self.assertEqual(id_1, "SUB-0001")
        self.assertEqual(id_2, "SUB-0002")
        self.assertEqual(id_3, "SUB-0003")
        self.assertEqual(self.db.count_proposals(), 3)

    def test_custom_project_id(self):
        """Verify custom project_id is honored if provided and valid."""
        custom_prop = dict(self.sample_proposal, project_id="PROPOSAL-2026-ALPHA")
        assigned_id = self.db.save_proposal(custom_prop, self.sample_evaluation)
        self.assertEqual(assigned_id, "PROPOSAL-2026-ALPHA")

        record = self.db.get_proposal_by_id("PROPOSAL-2026-ALPHA")
        self.assertIsNotNone(record)
        self.assertEqual(record["project_id"], "PROPOSAL-2026-ALPHA")

    def test_list_proposals_ordering_and_pagination(self):
        """Verify listing proposals returns newest first and respects pagination."""
        for i in range(1, 6):
            prop = dict(
                self.sample_proposal,
                project_title=f"Autonomous Robotics Research Part {i}",
                abstract=f"Abstract part {i} with distinct text content."
            )
            self.db.save_proposal(prop, self.sample_evaluation)

        # Fetch limit 3
        page_1 = self.db.list_proposals(limit=3, offset=0)
        self.assertEqual(len(page_1), 3)
        # Newest first: SUB-0005, SUB-0004, SUB-0003
        self.assertEqual(page_1[0]["project_id"], "SUB-0005")
        self.assertEqual(page_1[1]["project_id"], "SUB-0004")
        self.assertEqual(page_1[2]["project_id"], "SUB-0003")

        # Page 2: SUB-0002, SUB-0001
        page_2 = self.db.list_proposals(limit=3, offset=3)
        self.assertEqual(len(page_2), 2)
        self.assertEqual(page_2[0]["project_id"], "SUB-0002")
        self.assertEqual(page_2[1]["project_id"], "SUB-0001")

    def test_duplicate_detection_exact_content_hash(self):
        """Verify submitting identical 8 sections raises DuplicateProposalError and does not overwrite."""
        self.db.save_proposal(self.sample_proposal, self.sample_evaluation)
        self.assertEqual(self.db.count_proposals(), 1)

        # Attempt to save exact same content again
        with self.assertRaises(DuplicateProposalError) as ctx:
            self.db.save_proposal(self.sample_proposal, self.sample_evaluation)

        self.assertIn("SUB-0001", ctx.exception.existing_project_id)
        # Ensure count is still 1 (no duplicate inserted)
        self.assertEqual(self.db.count_proposals(), 1)

    def test_duplicate_detection_normalized_title(self):
        """Verify normalized title collision (ignoring casing, whitespace, and punctuation) is blocked."""
        self.db.save_proposal(self.sample_proposal, self.sample_evaluation)

        # Slightly altered punctuation/casing in title, but different abstract
        duplicate_title_prop = dict(
            self.sample_proposal,
            project_title="deep-learning: FOR autonomous drone navigation!!!",
            abstract="Completely different abstract text here."
        )

        with self.assertRaises(DuplicateProposalError) as ctx:
            self.db.save_proposal(duplicate_title_prop, self.sample_evaluation)

        self.assertIn("SUB-0001", ctx.exception.existing_project_id)
        self.assertEqual(self.db.count_proposals(), 1)

    def test_duplicate_detection_custom_id_collision(self):
        """Verify custom project ID collisions are blocked."""
        prop_1 = dict(self.sample_proposal, project_id="PROPOSAL-UNIQUE-99")
        self.db.save_proposal(prop_1, self.sample_evaluation)

        prop_2 = {
            "project_id": "PROPOSAL-UNIQUE-99",
            "project_title": "Totally Different Medical Imaging Project",
            "abstract": "Distinct abstract text.",
            "problem_statement": "Distinct problem.",
            "objectives": "Distinct objectives.",
            "proposed_methodology": "Distinct methodology.",
            "dataset": "Distinct dataset.",
            "algorithm_model": "Distinct algorithm.",
            "technology_tools": "Distinct technology."
        }

        with self.assertRaises(DuplicateProposalError) as ctx:
            self.db.save_proposal(prop_2, self.sample_evaluation)

        self.assertIn("PROPOSAL-UNIQUE-99", ctx.exception.existing_project_id)
        self.assertEqual(self.db.count_proposals(), 1)

    def test_check_duplicate_inspection_helper(self):
        """Verify check_duplicate() returns None for unique proposals and dict for duplicates."""
        # Unsaved proposal is unique
        self.assertIsNone(self.db.check_duplicate(self.sample_proposal))

        # Save proposal
        self.db.save_proposal(self.sample_proposal, self.sample_evaluation)

        # Now check_duplicate detects it
        dup_info = self.db.check_duplicate(self.sample_proposal)
        self.assertIsNotNone(dup_info)
        self.assertTrue(dup_info["is_duplicate"])
        self.assertEqual(dup_info["existing_project_id"], "SUB-0001")

        # Distinct new proposal is unique
        new_prop = {
            "project_title": "Blockchain IoT Supply Chain System",
            "abstract": "Ethereum smart contracts for supply chains.",
            "problem_statement": "Counterfeits in electronics.",
            "objectives": "Immutable audit logging.",
            "proposed_methodology": "Solidity smart contracts.",
            "dataset": "RFID sensor logs.",
            "algorithm_model": "Proof of Authority.",
            "technology_tools": "Solidity, Web3.js."
        }
        self.assertIsNone(self.db.check_duplicate(new_prop))

    def test_error_handling_empty_title_rollback(self):
        """Verify saving with empty title raises ValueError and does not commit anything."""
        invalid_prop = dict(self.sample_proposal, project_title="   ")
        with self.assertRaises(ValueError):
            self.db.save_proposal(invalid_prop, self.sample_evaluation)

        self.assertEqual(self.db.count_proposals(), 0)

    def test_error_handling_invalid_inputs(self):
        """Verify invalid proposal_data or evaluation_report raises ValueError."""
        with self.assertRaises(ValueError):
            self.db.save_proposal({}, self.sample_evaluation)

        with self.assertRaises(ValueError):
            self.db.save_proposal(self.sample_proposal, {})

        with self.assertRaises(ValueError):
            self.db.save_proposal(None, self.sample_evaluation)

    def test_get_nonexistent_proposal(self):
        """Verify get_proposal_by_id returns None for nonexistent IDs."""
        self.assertIsNone(self.db.get_proposal_by_id("SUB-9999"))
        self.assertIsNone(self.db.get_proposal_by_id(""))
        self.assertIsNone(self.db.get_proposal_by_id(None))

    def test_field_alias_support(self):
        """Verify saving proposals using shorthand field aliases (problem, methodology, etc.)."""
        alias_prop = {
            "title": "Quantum Cryptography for Secure Communication",
            "abstract": "BB84 protocol implementation.",
            "problem": "Quantum computer attack risks.",
            "objectives": "Generate entangled pairs.",
            "methodology": "Simulate quantum channel in Qiskit.",
            "dataset": "Photonic state logs.",
            "algorithm": "BB84 Key Distribution.",
            "technology": "Python, Qiskit."
        }
        pid = self.db.save_proposal(alias_prop, self.sample_evaluation)
        retrieved = self.db.get_proposal_by_id(pid)

        self.assertEqual(retrieved["project_title"], alias_prop["title"])
        self.assertEqual(retrieved["problem_statement"], alias_prop["problem"])
        self.assertEqual(retrieved["proposed_methodology"], alias_prop["methodology"])
        self.assertEqual(retrieved["algorithm_model"], alias_prop["algorithm"])
        self.assertEqual(retrieved["technology_tools"], alias_prop["technology"])

if __name__ == "__main__":
    unittest.main()
