"""
ETL Pipeline for AptiDude DS Project.

Extracts raw CSV data, transforms it (cleaning, validation, feature engineering),
and loads analytics-ready tables to data/processed/ and optionally to Neon PostgreSQL.

Usage:
    python -m src.data.etl_pipeline
    python -m src.data.etl_pipeline --upload-db   # also push to Neon
"""

import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# Resolve imports whether run as module or script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.helpers import (
    get_logger,
    load_raw_csv,
    save_processed_csv,
    upload_df_to_db,
    DATASET_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logger = get_logger("etl_pipeline")


# ===================================================================
# EXTRACT
# ===================================================================


def extract() -> dict[str, pd.DataFrame]:
    """Load all raw CSV files from the DATASET directory."""
    logger.info("=" * 60)
    logger.info("EXTRACT — Loading raw CSV files")
    logger.info("=" * 60)

    tables = {}
    for name in ["users", "questions", "attempts", "mock_tests"]:
        df = load_raw_csv(f"{name}.csv")
        logger.info(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        tables[name] = df

    # Also load ground truth files for validation
    for name in ["users_with_ground_truth", "questions_with_ground_truth"]:
        df = load_raw_csv(f"{name}.csv")
        logger.info(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        tables[name] = df

    return tables


# ===================================================================
# TRANSFORM
# ===================================================================


def transform_users(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the users table."""
    logger.info("Transforming users...")
    df = df.copy()

    # Parse dates
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    # Validate engagement levels
    valid_engagement = {"low", "medium", "high"}
    invalid = df[~df["engagement_level"].isin(valid_engagement)]
    if len(invalid) > 0:
        logger.warning(f"  {len(invalid)} users with invalid engagement_level")

    # Compute account age in days (from signup to today)
    df["account_age_days"] = (pd.Timestamp.now() - df["signup_date"]).dt.days

    logger.info(f"  Users cleaned: {len(df)} rows")
    return df


def transform_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the questions table."""
    logger.info("Transforming questions...")
    df = df.copy()

    # Standardize difficulty tags
    df["difficulty_tag"] = df["difficulty_tag"].str.strip().str.title()

    # Parse exam_tags from comma-separated string to list
    df["exam_tags_list"] = df["exam_tags"].apply(
        lambda x: [t.strip() for t in str(x).split(",")]
    )
    df["num_exam_tags"] = df["exam_tags_list"].apply(len)

    # Section and subtopic cleanup
    df["section"] = df["section"].str.strip()
    df["subtopic"] = df["subtopic"].str.strip()

    logger.info(f"  Questions cleaned: {len(df)} rows")
    logger.info(f"  Sections: {df['section'].nunique()} unique")
    logger.info(f"  Subtopics: {df['subtopic'].nunique()} unique")
    logger.info(f"  Difficulty distribution:\n{df['difficulty_tag'].value_counts().to_string()}")
    return df


def transform_attempts(df: pd.DataFrame, users: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and feature-engineer the attempts table.
    This is the core transformation — most analytics derive from this.
    """
    logger.info("Transforming attempts...")
    df = df.copy()

    # --- Type conversions ---
    df["is_correct"] = df["is_correct"].astype(bool)
    df["time_taken_sec"] = pd.to_numeric(df["time_taken_sec"], errors="coerce").fillna(0).astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # --- Foreign key validation ---
    valid_users = set(users["user_id"])
    valid_questions = set(questions["question_id"])

    orphan_users = df[~df["user_id"].isin(valid_users)]
    orphan_questions = df[~df["question_id"].isin(valid_questions)]
    if len(orphan_users) > 0:
        logger.warning(f"  {len(orphan_users)} attempts with unknown user_id — dropping")
        df = df[df["user_id"].isin(valid_users)]
    if len(orphan_questions) > 0:
        logger.warning(f"  {len(orphan_questions)} attempts with unknown question_id — dropping")
        df = df[df["question_id"].isin(valid_questions)]

    # --- Remove duplicates ---
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "question_id", "timestamp"])
    if len(df) < before:
        logger.info(f"  Removed {before - len(df)} duplicate attempts")

    # --- Feature engineering ---

    # Time-based features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    # Guessing detection: very fast + wrong (below 10 seconds and wrong)
    df["is_guessing"] = (~df["is_correct"]) & (df["time_taken_sec"] < 10)

    # Time outlier flag (> 600 seconds = 10 minutes on a single question)
    df["is_time_outlier"] = df["time_taken_sec"] > 600

    # Standardize source
    df["source"] = df["source"].str.strip().str.lower().str.replace(" ", "_")

    # Sort chronologically
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Compute attempt sequence number per user
    df["attempt_seq"] = df.groupby("user_id").cumcount() + 1

    logger.info(f"  Attempts cleaned: {len(df):,} rows")
    logger.info(f"  Guessing attempts: {df['is_guessing'].sum():,} ({df['is_guessing'].mean()*100:.1f}%)")
    logger.info(f"  Time outliers: {df['is_time_outlier'].sum():,}")
    logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def transform_mock_tests(df: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich mock test data."""
    logger.info("Transforming mock tests...")
    df = df.copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Numeric columns
    for col in ["varc_score", "dilr_score", "qa_score", "overall_score", "percentile_est"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # FK validation
    valid_users = set(users["user_id"])
    orphans = df[~df["user_id"].isin(valid_users)]
    if len(orphans) > 0:
        logger.warning(f"  {len(orphans)} mock tests with unknown user_id — dropping")
        df = df[df["user_id"].isin(valid_users)]

    # Sort by user and date
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    # Compute mock test sequence per user
    df["mock_seq"] = df.groupby("user_id").cumcount() + 1

    logger.info(f"  Mock tests cleaned: {len(df):,} rows")
    logger.info(f"  Score range: {df['overall_score'].min():.1f} – {df['overall_score'].max():.1f}")
    return df


def build_student_topic_summary(attempts: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-student, per-section, per-subtopic performance summary.

    This is the key analytics table that powers the diagnostic model.
    Applies exponential recency weighting — recent attempts matter more.
    """
    logger.info("Building student-topic performance summary...")
    df = attempts.copy()

    # Compute recency weight: exponential decay with half-life of 30 days
    max_ts = df["timestamp"].max()
    df["days_ago"] = (max_ts - df["timestamp"]).dt.total_seconds() / 86400
    half_life = 30.0
    df["recency_weight"] = np.exp(-np.log(2) * df["days_ago"] / half_life)

    # Weighted accuracy per (user, section, subtopic)
    grouped = df.groupby(["user_id", "section", "subtopic"]).agg(
        total_attempts=("is_correct", "count"),
        raw_correct=("is_correct", "sum"),
        raw_accuracy=("is_correct", "mean"),
        weighted_correct=("is_correct", lambda x: (x * df.loc[x.index, "recency_weight"]).sum()),
        total_weight=("recency_weight", "sum"),
        avg_time_sec=("time_taken_sec", "mean"),
        median_time_sec=("time_taken_sec", "median"),
        guessing_count=("is_guessing", "sum"),
        last_attempt=("timestamp", "max"),
    ).reset_index()

    # Weighted accuracy
    grouped["weighted_accuracy"] = grouped["weighted_correct"] / grouped["total_weight"]
    grouped["weighted_accuracy"] = grouped["weighted_accuracy"].clip(0, 1)

    # Days since last attempt
    grouped["days_since_last"] = (max_ts - grouped["last_attempt"]).dt.total_seconds() / 86400

    # Guessing rate
    grouped["guessing_rate"] = grouped["guessing_count"] / grouped["total_attempts"]

    logger.info(f"  Student-topic summary: {len(grouped):,} rows")
    logger.info(f"  Covering {grouped['user_id'].nunique()} users × {grouped['subtopic'].nunique()} subtopics")
    return grouped


def build_student_section_summary(topic_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up the subtopic summary to section level for each student.
    This powers the section-level radar charts.
    """
    logger.info("Building student-section summary...")
    grouped = topic_summary.groupby(["user_id", "section"]).agg(
        total_attempts=("total_attempts", "sum"),
        raw_correct=("raw_correct", "sum"),
        num_subtopics_attempted=("subtopic", "nunique"),
        avg_weighted_accuracy=("weighted_accuracy", "mean"),
        avg_time_sec=("avg_time_sec", "mean"),
        total_guessing=("guessing_count", "sum"),
    ).reset_index()

    grouped["raw_accuracy"] = grouped["raw_correct"] / grouped["total_attempts"]
    grouped["guessing_rate"] = grouped["total_guessing"] / grouped["total_attempts"]

    logger.info(f"  Student-section summary: {len(grouped):,} rows")
    return grouped


def build_daily_engagement(attempts: pd.DataFrame) -> pd.DataFrame:
    """Build a daily engagement table for trend analysis."""
    logger.info("Building daily engagement stats...")
    df = attempts.copy()
    df["date"] = df["timestamp"].dt.date

    daily = df.groupby("date").agg(
        active_users=("user_id", "nunique"),
        total_attempts=("attempt_id", "count"),
        correct_attempts=("is_correct", "sum"),
        avg_time_sec=("time_taken_sec", "mean"),
        guessing_attempts=("is_guessing", "sum"),
    ).reset_index()

    daily["accuracy"] = daily["correct_attempts"] / daily["total_attempts"]
    daily["guessing_rate"] = daily["guessing_attempts"] / daily["total_attempts"]
    daily["date"] = pd.to_datetime(daily["date"])

    logger.info(f"  Daily engagement: {len(daily)} days")
    return daily


# ===================================================================
# LOAD
# ===================================================================


def load_to_csv(tables: dict[str, pd.DataFrame]):
    """Save all processed tables to data/processed/."""
    logger.info("=" * 60)
    logger.info("LOAD — Saving processed tables to CSV")
    logger.info("=" * 60)

    for name, df in tables.items():
        save_processed_csv(df, f"{name}.csv")


def load_to_database(tables: dict[str, pd.DataFrame]):
    """Upload all processed tables to Neon PostgreSQL."""
    logger.info("=" * 60)
    logger.info("LOAD — Uploading to Neon PostgreSQL")
    logger.info("=" * 60)

    for name, df in tables.items():
        # Remove list columns that PostgreSQL can't handle directly
        df_upload = df.copy()
        for col in df_upload.columns:
            if df_upload[col].apply(lambda x: isinstance(x, list)).any():
                df_upload[col] = df_upload[col].apply(str)
        upload_df_to_db(df_upload, name)


# ===================================================================
# MAIN PIPELINE
# ===================================================================


def run_pipeline(upload_db: bool = False):
    """Execute the full ETL pipeline."""
    start = datetime.now()
    logger.info("=" * 60)
    logger.info("AptiDude ETL Pipeline — Starting")
    logger.info(f"Timestamp: {start.isoformat()}")
    logger.info("=" * 60)

    # --- EXTRACT ---
    raw = extract()

    # --- TRANSFORM ---
    logger.info("=" * 60)
    logger.info("TRANSFORM — Cleaning and enriching data")
    logger.info("=" * 60)

    users = transform_users(raw["users"])
    questions = transform_questions(raw["questions"])
    attempts = transform_attempts(raw["attempts"], users, questions)
    mock_tests = transform_mock_tests(raw["mock_tests"], users)

    # Build analytics tables
    student_topic_summary = build_student_topic_summary(attempts)
    student_section_summary = build_student_section_summary(student_topic_summary)
    daily_engagement = build_daily_engagement(attempts)

    # Bundle all processed tables
    processed = {
        "users": users,
        "questions": questions,
        "attempts": attempts,
        "mock_tests": mock_tests,
        "users_ground_truth": raw["users_with_ground_truth"],
        "questions_ground_truth": raw["questions_with_ground_truth"],
        "student_topic_summary": student_topic_summary,
        "student_section_summary": student_section_summary,
        "daily_engagement": daily_engagement,
    }

    # --- LOAD ---
    load_to_csv(processed)

    if upload_db:
        try:
            load_to_database(processed)
        except Exception as e:
            logger.error(f"Database upload failed: {e}")
            logger.info("CSV files were saved successfully — you can retry DB upload later.")

    # --- SUMMARY ---
    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=" * 60)
    logger.info(f"ETL Pipeline complete in {elapsed:.1f}s")
    logger.info("Processed tables:")
    for name, df in processed.items():
        logger.info(f"  {name}: {len(df):,} rows × {df.shape[1]} cols")
    logger.info("=" * 60)

    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AptiDude ETL Pipeline")
    parser.add_argument(
        "--upload-db",
        action="store_true",
        help="Also upload processed tables to Neon PostgreSQL",
    )
    args = parser.parse_args()
    run_pipeline(upload_db=args.upload_db)
