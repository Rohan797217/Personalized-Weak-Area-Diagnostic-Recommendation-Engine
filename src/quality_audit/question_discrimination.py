"""
Question Quality Audit — Discrimination Index Analysis.

Flags questions with abnormally low discrimination (strong students get it wrong
as often as weak students → signals bad/ambiguous question).

Usage:
    python -m src.quality_audit.question_discrimination
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.helpers import (
    get_logger,
    load_processed_csv,
    save_processed_csv,
    DATASET_DIR,
)

logger = get_logger("question_audit")


def compute_discrimination_index(attempts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the discrimination index for each question.

    Method: Point-biserial correlation between item score (correct/incorrect)
    and the student's overall score (total accuracy across all attempts).

    A good question should have high discrimination: strong students get it right
    more often than weak students. Low/negative discrimination = problematic question.
    """
    logger.info("Computing question discrimination indices...")

    # Compute overall student ability (total accuracy)
    student_accuracy = attempts.groupby("user_id")["is_correct"].mean().reset_index()
    student_accuracy.columns = ["user_id", "overall_accuracy"]

    # Merge student ability back into attempts
    df = attempts.merge(student_accuracy, on="user_id", how="left")

    # For each question, compute point-biserial correlation
    results = []
    for qid, group in df.groupby("question_id"):
        n_attempts = len(group)
        if n_attempts < 10:
            # Need enough attempts for reliable discrimination
            results.append({
                "question_id": qid,
                "n_attempts": n_attempts,
                "discrimination_index": np.nan,
                "p_value": np.nan,
                "correct_rate": group["is_correct"].mean(),
                "avg_time_sec": group["time_taken_sec"].mean(),
                "flag": "insufficient_data",
            })
            continue

        correct = group["is_correct"].astype(int)
        ability = group["overall_accuracy"]

        # Point-biserial correlation
        if correct.std() == 0 or ability.std() == 0:
            disc = 0.0
            pval = 1.0
        else:
            disc, pval = stats.pointbiserialr(correct, ability)

        # Classify question quality
        if disc < 0:
            flag = "negative_discrimination"  # 🔴 strong students fail more
        elif disc < 0.1:
            flag = "low_discrimination"  # 🟡 doesn't distinguish well
        elif disc < 0.2:
            flag = "moderate_discrimination"  # 🟠 acceptable but could be better
        else:
            flag = "good_discrimination"  # 🟢 effective question

        results.append({
            "question_id": qid,
            "n_attempts": n_attempts,
            "discrimination_index": round(disc, 4),
            "p_value": round(pval, 6),
            "correct_rate": round(group["is_correct"].mean(), 4),
            "avg_time_sec": round(group["time_taken_sec"].mean(), 1),
            "flag": flag,
        })

    results_df = pd.DataFrame(results)

    # Merge with question metadata
    questions = load_processed_csv("questions.csv")
    results_df = results_df.merge(
        questions[["question_id", "section", "subtopic", "difficulty_tag"]],
        on="question_id", how="left"
    )

    # Sort by discrimination (worst first)
    results_df = results_df.sort_values("discrimination_index", ascending=True)

    logger.info(f"  Total questions analyzed: {len(results_df)}")
    logger.info(f"  Flag distribution:")
    logger.info(results_df["flag"].value_counts().to_string())

    return results_df


def compute_difficulty_vs_tagged(questions_audit: pd.DataFrame) -> pd.DataFrame:
    """
    Compare actual correct rates with tagged difficulty.
    Identifies mistagged questions (e.g., tagged 'Easy' but only 20% get it right).
    """
    logger.info("Checking difficulty tag accuracy...")

    df = questions_audit.dropna(subset=["correct_rate", "difficulty_tag"]).copy()

    # Expected correct-rate ranges by difficulty
    expected_ranges = {
        "Easy": (0.6, 1.0),
        "Medium": (0.35, 0.7),
        "Hard": (0.0, 0.45),
    }

    def check_tag(row):
        expected = expected_ranges.get(row["difficulty_tag"])
        if expected is None:
            return "unknown_tag"
        low, high = expected
        if row["correct_rate"] < low:
            return "harder_than_tagged"
        elif row["correct_rate"] > high:
            return "easier_than_tagged"
        return "correctly_tagged"

    df["tag_accuracy"] = df.apply(check_tag, axis=1)

    logger.info(f"  Tag accuracy distribution:")
    logger.info(df["tag_accuracy"].value_counts().to_string())

    return df


def validate_against_ground_truth(questions_audit: pd.DataFrame) -> pd.DataFrame:
    """
    Validate IRT difficulty estimates against ground-truth correct probabilities.
    """
    logger.info("Validating against ground truth question difficulties...")

    gt = pd.read_csv(DATASET_DIR / "questions_with_ground_truth.csv")

    merged = questions_audit.merge(
        gt[["question_id", "_true_correct_prob"]],
        on="question_id", how="inner"
    )

    if len(merged) > 10:
        corr, pval = stats.pearsonr(
            merged["correct_rate"].dropna(),
            merged.loc[merged["correct_rate"].notna(), "_true_correct_prob"]
        )
        logger.info(f"  Observed correct_rate vs true_correct_prob: r={corr:.3f} (p={pval:.4f})")

    return merged


def run_quality_audit():
    """Run the full question quality audit."""
    logger.info("=" * 60)
    logger.info("Question Quality Audit — Starting")
    logger.info("=" * 60)

    attempts = load_processed_csv("attempts.csv")
    attempts["is_correct"] = attempts["is_correct"].astype(bool)

    # Core discrimination analysis
    audit = compute_discrimination_index(attempts)
    save_processed_csv(audit, "question_quality_audit.csv")

    # Difficulty tag validation
    tag_check = compute_difficulty_vs_tagged(audit)
    save_processed_csv(tag_check, "difficulty_tag_validation.csv")

    # Ground truth validation
    gt_validation = validate_against_ground_truth(audit)
    save_processed_csv(gt_validation, "question_gt_validation.csv")

    # Summary for founders
    flagged = audit[audit["flag"].isin(["negative_discrimination", "low_discrimination"])]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"FLAGGED QUESTIONS: {len(flagged)} questions need review")
    logger.info(f"  Negative discrimination: {len(audit[audit['flag'] == 'negative_discrimination'])}")
    logger.info(f"  Low discrimination: {len(audit[audit['flag'] == 'low_discrimination'])}")
    logger.info(f"\nTop 10 most problematic questions:")
    if len(flagged) > 0:
        logger.info(flagged.head(10)[
            ["question_id", "section", "subtopic", "difficulty_tag",
             "discrimination_index", "correct_rate", "n_attempts", "flag"]
        ].to_string(index=False))

    logger.info(f"{'=' * 60}")
    logger.info("Question Quality Audit — Complete")
    logger.info(f"{'=' * 60}")

    return audit


if __name__ == "__main__":
    run_quality_audit()
