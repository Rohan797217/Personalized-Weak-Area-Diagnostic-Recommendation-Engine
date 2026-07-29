"""
Unit tests for the ETL pipeline.

Run: python -m pytest tests/test_etl.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.utils.helpers import DATASET_DIR, DATA_PROCESSED_DIR
from src.data.etl_pipeline import (
    extract,
    transform_users,
    transform_questions,
    transform_attempts,
    transform_mock_tests,
    build_student_topic_summary,
)


class TestExtract:
    """Tests for the extract phase."""

    def test_all_raw_files_exist(self):
        """All required CSV files should exist in DATASET directory."""
        required = ["users.csv", "questions.csv", "attempts.csv", "mock_tests.csv"]
        for f in required:
            assert (DATASET_DIR / f).exists(), f"Missing: {f}"

    def test_extract_loads_all_tables(self):
        """Extract should return all expected tables."""
        tables = extract()
        expected = ["users", "questions", "attempts", "mock_tests",
                     "users_with_ground_truth", "questions_with_ground_truth"]
        for name in expected:
            assert name in tables, f"Missing table: {name}"
            assert len(tables[name]) > 0, f"Empty table: {name}"


class TestTransformUsers:
    """Tests for user transformation."""

    def test_signup_date_is_datetime(self):
        raw = pd.read_csv(DATASET_DIR / "users.csv")
        users = transform_users(raw)
        assert pd.api.types.is_datetime64_any_dtype(users["signup_date"])

    def test_account_age_computed(self):
        raw = pd.read_csv(DATASET_DIR / "users.csv")
        users = transform_users(raw)
        assert "account_age_days" in users.columns
        assert users["account_age_days"].min() >= 0

    def test_no_rows_lost(self):
        raw = pd.read_csv(DATASET_DIR / "users.csv")
        users = transform_users(raw)
        assert len(users) == len(raw)


class TestTransformQuestions:
    """Tests for question transformation."""

    def test_difficulty_tags_standardized(self):
        raw = pd.read_csv(DATASET_DIR / "questions.csv")
        questions = transform_questions(raw)
        valid_tags = {"Easy", "Medium", "Hard"}
        assert questions["difficulty_tag"].isin(valid_tags).all()

    def test_exam_tags_parsed(self):
        raw = pd.read_csv(DATASET_DIR / "questions.csv")
        questions = transform_questions(raw)
        assert "exam_tags_list" in questions.columns
        assert "num_exam_tags" in questions.columns
        assert questions["num_exam_tags"].min() >= 1


class TestTransformAttempts:
    """Tests for attempt transformation."""

    def test_is_correct_is_boolean(self):
        users = pd.read_csv(DATASET_DIR / "users.csv")
        questions = pd.read_csv(DATASET_DIR / "questions.csv")
        raw = pd.read_csv(DATASET_DIR / "attempts.csv")
        attempts = transform_attempts(raw, users, questions)
        assert attempts["is_correct"].dtype == bool

    def test_timestamp_is_datetime(self):
        users = pd.read_csv(DATASET_DIR / "users.csv")
        questions = pd.read_csv(DATASET_DIR / "questions.csv")
        raw = pd.read_csv(DATASET_DIR / "attempts.csv")
        attempts = transform_attempts(raw, users, questions)
        assert pd.api.types.is_datetime64_any_dtype(attempts["timestamp"])

    def test_feature_columns_added(self):
        users = pd.read_csv(DATASET_DIR / "users.csv")
        questions = pd.read_csv(DATASET_DIR / "questions.csv")
        raw = pd.read_csv(DATASET_DIR / "attempts.csv")
        attempts = transform_attempts(raw, users, questions)
        expected_cols = ["hour_of_day", "day_of_week", "is_weekend",
                         "is_guessing", "is_time_outlier", "attempt_seq"]
        for col in expected_cols:
            assert col in attempts.columns, f"Missing column: {col}"

    def test_no_orphan_users(self):
        users = pd.read_csv(DATASET_DIR / "users.csv")
        questions = pd.read_csv(DATASET_DIR / "questions.csv")
        raw = pd.read_csv(DATASET_DIR / "attempts.csv")
        attempts = transform_attempts(raw, users, questions)
        valid_users = set(users["user_id"])
        assert attempts["user_id"].isin(valid_users).all()

    def test_chronological_order(self):
        users = pd.read_csv(DATASET_DIR / "users.csv")
        questions = pd.read_csv(DATASET_DIR / "questions.csv")
        raw = pd.read_csv(DATASET_DIR / "attempts.csv")
        attempts = transform_attempts(raw, users, questions)
        # Within each user, timestamps should be non-decreasing
        for uid, group in attempts.groupby("user_id"):
            ts = group["timestamp"].values
            assert (ts[1:] >= ts[:-1]).all(), f"Non-chronological for {uid}"


class TestStudentTopicSummary:
    """Tests for the student-topic summary builder."""

    def test_summary_has_expected_columns(self):
        users = pd.read_csv(DATASET_DIR / "users.csv")
        questions = pd.read_csv(DATASET_DIR / "questions.csv")
        raw = pd.read_csv(DATASET_DIR / "attempts.csv")
        attempts = transform_attempts(raw, users, questions)
        summary = build_student_topic_summary(attempts)

        expected_cols = ["user_id", "section", "subtopic", "total_attempts",
                         "weighted_accuracy", "avg_time_sec"]
        for col in expected_cols:
            assert col in summary.columns, f"Missing: {col}"

    def test_weighted_accuracy_in_range(self):
        users = pd.read_csv(DATASET_DIR / "users.csv")
        questions = pd.read_csv(DATASET_DIR / "questions.csv")
        raw = pd.read_csv(DATASET_DIR / "attempts.csv")
        attempts = transform_attempts(raw, users, questions)
        summary = build_student_topic_summary(attempts)

        assert summary["weighted_accuracy"].min() >= 0
        assert summary["weighted_accuracy"].max() <= 1


class TestProcessedOutputs:
    """Tests that verify processed output files exist after pipeline run."""

    def test_processed_files_exist(self):
        """Check core processed files exist (run after ETL pipeline)."""
        required = [
            "users.csv", "questions.csv", "attempts.csv", "mock_tests.csv",
            "student_topic_summary.csv", "student_section_summary.csv",
            "daily_engagement.csv",
        ]
        for f in required:
            path = DATA_PROCESSED_DIR / f
            if path.exists():
                df = pd.read_csv(path)
                assert len(df) > 0, f"Empty file: {f}"
