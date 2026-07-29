"""
Synthetic Data Generator for AptiDude.

Generates realistic synthetic data for the AptiDude platform:
- Users with varied engagement levels and target exams
- Questions across sections/subtopics with difficulty distributions
- Attempts with realistic timing, correctness patterns, and guessing
- Mock tests with section scores and percentiles

This de-risks the project by allowing development even if real data
access is delayed.

Usage:
    python -m src.data.generate_synthetic_data
    python -m src.data.generate_synthetic_data --n-users 1000 --n-questions 15000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.utils.helpers import get_logger, DATA_SYNTHETIC_DIR

logger = get_logger("synthetic_data_generator")

# Seed for reproducibility
RNG = np.random.default_rng(42)

# =====================================================================
# CONFIGURATION
# =====================================================================

SECTIONS = {
    "Quantitative Aptitude": [
        "Number System", "Averages", "Percentages", "Profit & Loss",
        "Simple Interest", "Compound Interest", "Ratio & Proportion",
        "Time & Work", "Time, Speed & Distance", "Geometry",
        "Mensuration", "Algebra", "Trigonometry", "Progressions (AP/GP)",
        "Logarithms", "Permutations & Combinations", "Probability",
    ],
    "Logical Reasoning": [
        "Puzzles", "Syllogisms", "Blood Relations", "Coding-Decoding",
        "Direction Sense", "Ranking & Order", "Seating Arrangement",
        "Clocks & Calendars", "Cubes & Dice", "Series Completion",
        "Analogies", "Classification",
    ],
    "Verbal Ability": [
        "Reading Comprehension", "Para Jumbles", "Para Summary",
        "Sentence Completion", "Critical Reasoning", "Vocabulary",
        "Fill in the Blanks", "Idioms & Phrases", "Grammar",
        "Error Spotting",
    ],
    "Data Interpretation": [
        "Bar Charts", "Pie Charts", "Line Graphs", "Tables",
        "Caselets", "Mixed DI", "Data Sufficiency",
    ],
}

EXAMS = ["CAT", "Placements", "TCS NQT", "General"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]
ENGAGEMENT_LEVELS = ["low", "medium", "high"]
SOURCES = ["practice", "mock_test", "daily_drill"]


# =====================================================================
# GENERATORS
# =====================================================================


def generate_users(n: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic user profiles with hidden skill scores."""
    logger.info(f"Generating {n} users...")

    users = []
    users_gt = []

    for i in range(n):
        user_id = f"U{i+1:05d}"
        signup_date = datetime(2025, 6, 1) + timedelta(days=RNG.integers(0, 400))
        target_exam = RNG.choice(EXAMS, p=[0.35, 0.30, 0.20, 0.15])
        engagement = RNG.choice(ENGAGEMENT_LEVELS, p=[0.25, 0.45, 0.30])

        # Hidden ground truth: skill levels per section (0 to 1)
        base_skill = RNG.beta(2, 2)  # baseline skill
        skill_quant = np.clip(base_skill + RNG.normal(0, 0.15), 0.05, 0.95)
        skill_lr = np.clip(base_skill + RNG.normal(0, 0.15), 0.05, 0.95)
        skill_va = np.clip(base_skill + RNG.normal(0, 0.15), 0.05, 0.95)
        skill_di = np.clip(base_skill + RNG.normal(0, 0.15), 0.05, 0.95)

        users.append({
            "user_id": user_id,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "target_exam": target_exam,
            "engagement_level": engagement,
        })

        users_gt.append({
            "user_id": user_id,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "target_exam": target_exam,
            "engagement_level": engagement,
            "_skill_quant": round(skill_quant, 3),
            "_skill_lr": round(skill_lr, 3),
            "_skill_va": round(skill_va, 3),
            "_skill_di": round(skill_di, 3),
        })

    return pd.DataFrame(users), pd.DataFrame(users_gt)


def generate_questions(n: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic question catalog with hidden difficulty."""
    logger.info(f"Generating {n} questions...")

    questions = []
    questions_gt = []

    for i in range(n):
        question_id = f"Q{i+1:05d}"
        section = RNG.choice(list(SECTIONS.keys()))
        subtopic = RNG.choice(SECTIONS[section])
        difficulty = RNG.choice(DIFFICULTIES, p=[0.35, 0.45, 0.20])

        # Determine exam tags
        n_tags = RNG.integers(1, 4)
        exam_tags = list(RNG.choice(EXAMS, size=n_tags, replace=False))

        # Hidden ground truth: true correct probability
        diff_map = {"Easy": 0.75, "Medium": 0.50, "Hard": 0.30}
        true_prob = np.clip(
            diff_map[difficulty] + RNG.normal(0, 0.12), 0.05, 0.95
        )

        questions.append({
            "question_id": question_id,
            "section": section,
            "subtopic": subtopic,
            "difficulty_tag": difficulty,
            "exam_tags": ",".join(exam_tags),
        })

        questions_gt.append({
            "question_id": question_id,
            "section": section,
            "subtopic": subtopic,
            "difficulty_tag": difficulty,
            "exam_tags": ",".join(exam_tags),
            "_true_correct_prob": round(true_prob, 3),
        })

    return pd.DataFrame(questions), pd.DataFrame(questions_gt)


def generate_attempts(
    users_gt: pd.DataFrame,
    questions_gt: pd.DataFrame,
    avg_attempts_per_user: int = 170,
) -> pd.DataFrame:
    """
    Generate synthetic attempt logs.

    The probability of a correct answer depends on:
    - Student skill (from ground truth)
    - Question difficulty (from ground truth)
    Combined via a logistic-like model.
    """
    logger.info("Generating attempts...")

    # Map section to skill column
    section_to_skill = {
        "Quantitative Aptitude": "_skill_quant",
        "Logical Reasoning": "_skill_lr",
        "Verbal Ability": "_skill_va",
        "Data Interpretation": "_skill_di",
    }

    attempts = []
    q_gt = questions_gt.set_index("question_id")

    for _, user in users_gt.iterrows():
        # Number of attempts varies by engagement
        eng_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}
        n_attempts = int(
            avg_attempts_per_user * eng_multiplier[user["engagement_level"]]
            * RNG.uniform(0.7, 1.3)
        )

        signup = datetime.strptime(user["signup_date"], "%Y-%m-%d")
        days_active = max(30, (datetime(2026, 7, 29) - signup).days)

        # Sample questions
        q_ids = RNG.choice(questions_gt["question_id"].values, size=n_attempts, replace=True)

        for q_id in q_ids:
            q_info = q_gt.loc[q_id]
            skill_col = section_to_skill.get(q_info["section"], "_skill_quant")
            student_skill = user[skill_col]
            q_difficulty = q_info["_true_correct_prob"]

            # P(correct) based on skill and difficulty
            p_correct = np.clip(
                student_skill * 0.6 + q_difficulty * 0.4 + RNG.normal(0, 0.08),
                0.01, 0.99
            )
            is_correct = RNG.random() < p_correct

            # Time taken depends on difficulty and correctness
            base_time = {"Easy": 25, "Medium": 45, "Hard": 70}
            time_taken = max(3, int(
                base_time.get(q_info["difficulty_tag"], 45)
                * RNG.lognormal(0, 0.4)
                * (0.85 if is_correct else 1.15)
            ))

            # Small chance of guessing (< 10 seconds, usually wrong)
            if RNG.random() < 0.015:
                time_taken = RNG.integers(3, 10)
                is_correct = RNG.random() < 0.2

            timestamp = signup + timedelta(
                days=RNG.integers(0, days_active),
                hours=RNG.integers(6, 23),
                minutes=RNG.integers(0, 60),
                seconds=RNG.integers(0, 60),
            )

            source = RNG.choice(SOURCES, p=[0.55, 0.25, 0.20])

            attempts.append({
                "attempt_id": str(uuid.uuid4())[:11],
                "user_id": user["user_id"],
                "question_id": q_id,
                "section": q_info["section"],
                "subtopic": q_info["subtopic"],
                "difficulty_tag": q_info["difficulty_tag"],
                "is_correct": is_correct,
                "time_taken_sec": time_taken,
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": source,
            })

    df = pd.DataFrame(attempts)
    logger.info(f"  Generated {len(df):,} attempts for {users_gt['user_id'].nunique()} users")
    return df


def generate_mock_tests(
    users_gt: pd.DataFrame,
    avg_mocks_per_user: int = 7,
) -> pd.DataFrame:
    """Generate synthetic mock test results."""
    logger.info("Generating mock tests...")

    section_to_skill = {
        "varc": "_skill_va",
        "dilr": "_skill_di",
        "qa": "_skill_quant",
    }

    mocks = []

    for _, user in users_gt.iterrows():
        n_mocks = max(1, int(avg_mocks_per_user * RNG.uniform(0.5, 1.5)))
        signup = datetime.strptime(user["signup_date"], "%Y-%m-%d")

        for m in range(n_mocks):
            mock_date = signup + timedelta(days=RNG.integers(14, 400))
            if mock_date > datetime(2026, 7, 29):
                continue

            # Section scores (max ~25 each, total ~75)
            varc = np.clip(user["_skill_va"] * 25 * RNG.uniform(0.6, 1.2), 2, 25)
            dilr = np.clip(user["_skill_di"] * 25 * RNG.uniform(0.6, 1.2), 2, 25)
            qa = np.clip(user["_skill_quant"] * 25 * RNG.uniform(0.6, 1.2), 2, 25)
            overall = varc + dilr + qa

            # Percentile estimate based on overall score
            percentile = np.clip(overall / 75 * 100 * RNG.uniform(0.85, 1.15), 5, 99.9)

            mocks.append({
                "mock_id": str(uuid.uuid4())[:11],
                "user_id": user["user_id"],
                "date": mock_date.strftime("%Y-%m-%d"),
                "varc_score": round(varc, 1),
                "dilr_score": round(dilr, 1),
                "qa_score": round(qa, 1),
                "overall_score": round(overall, 1),
                "percentile_est": round(percentile, 1),
            })

    df = pd.DataFrame(mocks)
    logger.info(f"  Generated {len(df):,} mock tests")
    return df


# =====================================================================
# MAIN
# =====================================================================


def generate_all(n_users: int = 600, n_questions: int = 10000):
    """Generate all synthetic data tables and save to data/synthetic/."""
    DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Synthetic Data Generator - Starting")
    logger.info(f"  Users: {n_users}, Questions: {n_questions}")
    logger.info("=" * 60)

    # Generate
    users, users_gt = generate_users(n_users)
    questions, questions_gt = generate_questions(n_questions)
    attempts = generate_attempts(users_gt, questions_gt)
    mock_tests = generate_mock_tests(users_gt)

    # Save
    users.to_csv(DATA_SYNTHETIC_DIR / "users.csv", index=False)
    users_gt.to_csv(DATA_SYNTHETIC_DIR / "users_with_ground_truth.csv", index=False)
    questions.to_csv(DATA_SYNTHETIC_DIR / "questions.csv", index=False)
    questions_gt.to_csv(DATA_SYNTHETIC_DIR / "questions_with_ground_truth.csv", index=False)
    attempts.to_csv(DATA_SYNTHETIC_DIR / "attempts.csv", index=False)
    mock_tests.to_csv(DATA_SYNTHETIC_DIR / "mock_tests.csv", index=False)

    logger.info("=" * 60)
    logger.info("Synthetic Data Generator - Complete")
    logger.info(f"  Users:     {len(users):,}")
    logger.info(f"  Questions: {len(questions):,}")
    logger.info(f"  Attempts:  {len(attempts):,}")
    logger.info(f"  Mocks:     {len(mock_tests):,}")
    logger.info(f"  Output:    {DATA_SYNTHETIC_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AptiDude Synthetic Data Generator")
    parser.add_argument("--n-users", type=int, default=600, help="Number of users")
    parser.add_argument("--n-questions", type=int, default=10000, help="Number of questions")
    args = parser.parse_args()
    generate_all(n_users=args.n_users, n_questions=args.n_questions)
