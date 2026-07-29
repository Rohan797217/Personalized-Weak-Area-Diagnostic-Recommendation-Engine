"""
Recommendation Engine for AptiDude.

Recommends next-best questions/topics for each student based on:
1. Weak-area targeting — prioritize weakest subtopics
2. Difficulty progression — easy → medium → hard within weak areas
3. Spaced repetition — resurface previously-wrong questions after a gap
4. Diversity — balance across sections to avoid fatigue

Usage:
    python -m src.models.recommendation_engine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.utils.helpers import (
    get_logger,
    load_processed_csv,
    save_processed_csv,
)

logger = get_logger("recommendation_engine")


class RecommendationEngine:
    """
    Rule-based recommendation engine with spaced repetition.

    Strategy:
    1. Identify student's weakest subtopics
    2. For each weak subtopic, select unattempted or previously-wrong questions
    3. Order by difficulty progression (confidence building)
    4. Mix in spaced-repetition candidates (old mistakes to revisit)
    5. Diversify across sections
    """

    def __init__(
        self,
        n_weak_subtopics: int = 5,
        spaced_rep_gap_days: int = 7,
        difficulty_order: list[str] = None,
        max_recommendations: int = 20,
    ):
        self.n_weak_subtopics = n_weak_subtopics
        self.spaced_rep_gap_days = spaced_rep_gap_days
        self.difficulty_order = difficulty_order or ["Easy", "Medium", "Hard"]
        self.max_recommendations = max_recommendations

        self.attempts = None
        self.questions = None
        self.topic_summary = None
        self.question_quality = None
        self.now = None

    def fit(
        self,
        attempts: pd.DataFrame,
        questions: pd.DataFrame,
        topic_summary: pd.DataFrame,
        question_quality: pd.DataFrame = None,
    ):
        """Load all data needed for recommendations."""
        logger.info("Loading recommendation engine data...")

        self.attempts = attempts.copy()
        self.attempts["timestamp"] = pd.to_datetime(self.attempts["timestamp"])

        self.questions = questions.copy()
        self.topic_summary = topic_summary.copy()
        self.question_quality = question_quality

        self.now = self.attempts["timestamp"].max()

        # Pre-compute: for each question, who attempted it and when
        self.user_attempts = {}
        for uid, group in self.attempts.groupby("user_id"):
            self.user_attempts[uid] = {
                "attempted_questions": set(group["question_id"]),
                "wrong_questions": set(group[~group["is_correct"]]["question_id"]),
                "attempt_history": group[["question_id", "is_correct", "timestamp"]].copy(),
            }

        # Remove low-quality questions from recommendation pool
        if self.question_quality is not None:
            bad_questions = set(
                self.question_quality[
                    self.question_quality["flag"].isin(["negative_discrimination"])
                ]["question_id"]
            )
            self.questions = self.questions[~self.questions["question_id"].isin(bad_questions)]
            logger.info(f"  Excluded {len(bad_questions)} low-quality questions from pool")

        logger.info(f"  {len(self.user_attempts)} students, {len(self.questions)} questions in pool")
        return self

    def recommend(self, user_id: str) -> list[dict]:
        """
        Generate personalized recommendations for a student.

        Returns a ranked list of dicts:
        [
            {
                "question_id": str,
                "section": str,
                "subtopic": str,
                "difficulty_tag": str,
                "reason": str,
                "priority": float,
            },
            ...
        ]
        """
        if user_id not in self.user_attempts:
            logger.warning(f"No attempt data for {user_id}")
            return []

        user_data = self.user_attempts[user_id]
        attempted = user_data["attempted_questions"]
        wrong = user_data["wrong_questions"]
        history = user_data["attempt_history"]

        recommendations = []

        # --- 1. Weak-area questions (unattempted) ---
        weak_recs = self._get_weak_area_recommendations(user_id, attempted)
        recommendations.extend(weak_recs)

        # --- 2. Spaced repetition (previously wrong, revisit after gap) ---
        spaced_recs = self._get_spaced_repetition_candidates(user_id, history)
        recommendations.extend(spaced_recs)

        # --- 3. Diversification (ensure section balance) ---
        recommendations = self._diversify_sections(recommendations)

        # --- 4. Sort by priority and limit ---
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        recommendations = recommendations[: self.max_recommendations]

        return recommendations

    def _get_weak_area_recommendations(
        self, user_id: str, attempted: set
    ) -> list[dict]:
        """Recommend unattempted questions from the student's weakest subtopics."""
        user_topics = self.topic_summary[
            (self.topic_summary["user_id"] == user_id)
            & (self.topic_summary["total_attempts"] >= 3)
        ].sort_values("weighted_accuracy")

        # Get weakest subtopics
        weak_subtopics = user_topics.head(self.n_weak_subtopics)
        recs = []

        for _, row in weak_subtopics.iterrows():
            section = row["section"]
            subtopic = row["subtopic"]
            weakness = 1.0 - row["weighted_accuracy"]  # higher = weaker

            # Find unattempted questions in this subtopic
            candidates = self.questions[
                (self.questions["section"] == section)
                & (self.questions["subtopic"] == subtopic)
                & (~self.questions["question_id"].isin(attempted))
            ].copy()

            if len(candidates) == 0:
                continue

            # Sort by difficulty progression (easy first for confidence)
            diff_rank = {d: i for i, d in enumerate(self.difficulty_order)}
            candidates["diff_rank"] = candidates["difficulty_tag"].map(diff_rank).fillna(1)
            candidates = candidates.sort_values("diff_rank")

            # Take top 4 per subtopic (mix of difficulties)
            for _, q in candidates.head(4).iterrows():
                recs.append({
                    "question_id": q["question_id"],
                    "section": section,
                    "subtopic": subtopic,
                    "difficulty_tag": q["difficulty_tag"],
                    "reason": f"Weak area: {subtopic} (accuracy: {row['weighted_accuracy']:.0%})",
                    "priority": weakness * 10 + (2 - q["diff_rank"]),  # weak + easy = highest priority
                })

        return recs

    def _get_spaced_repetition_candidates(
        self, user_id: str, history: pd.DataFrame
    ) -> list[dict]:
        """
        Resurface previously-wrong questions after a gap.

        Only includes questions the student got wrong AND hasn't attempted
        in the last `spaced_rep_gap_days` days.
        """
        gap = timedelta(days=self.spaced_rep_gap_days)
        cutoff = self.now - gap

        # Questions the student got wrong
        wrong_attempts = history[~history["is_correct"]].copy()
        if len(wrong_attempts) == 0:
            return []

        # Last attempt per question
        last_attempt = history.groupby("question_id")["timestamp"].max().reset_index()
        last_attempt.columns = ["question_id", "last_attempt_time"]

        # Wrong questions whose last attempt was before the gap
        wrong_qids = set(wrong_attempts["question_id"])
        eligible = last_attempt[
            (last_attempt["question_id"].isin(wrong_qids))
            & (last_attempt["last_attempt_time"] < cutoff)
        ]

        # Get question details
        recs = []
        for _, row in eligible.iterrows():
            qid = row["question_id"]
            q_info = self.questions[self.questions["question_id"] == qid]
            if len(q_info) == 0:
                continue
            q = q_info.iloc[0]

            days_since = (self.now - row["last_attempt_time"]).total_seconds() / 86400

            recs.append({
                "question_id": qid,
                "section": q["section"],
                "subtopic": q["subtopic"],
                "difficulty_tag": q["difficulty_tag"],
                "reason": f"Spaced repetition: got wrong {days_since:.0f} days ago",
                "priority": 5.0 + min(days_since / 30, 3),  # older = slightly higher priority
            })

        return recs

    def _diversify_sections(self, recommendations: list[dict]) -> list[dict]:
        """Ensure recommendations aren't all from the same section."""
        if not recommendations:
            return recommendations

        # Group by section
        by_section = {}
        for rec in recommendations:
            sec = rec["section"]
            by_section.setdefault(sec, []).append(rec)

        # Interleave: round-robin pick from each section
        diversified = []
        sections = list(by_section.keys())
        max_per_round = max(len(v) for v in by_section.values())

        for i in range(max_per_round):
            for sec in sections:
                if i < len(by_section[sec]):
                    diversified.append(by_section[sec][i])

        return diversified

    def batch_recommend(self, user_ids: list[str] = None) -> pd.DataFrame:
        """
        Generate recommendations for multiple students.
        Returns a DataFrame with all recommendations.
        """
        if user_ids is None:
            user_ids = list(self.user_attempts.keys())

        all_recs = []
        for uid in user_ids:
            recs = self.recommend(uid)
            for r in recs:
                r["user_id"] = uid
                all_recs.append(r)

        df = pd.DataFrame(all_recs)
        if len(df) > 0:
            df = df[["user_id", "question_id", "section", "subtopic",
                      "difficulty_tag", "reason", "priority"]]
        return df


# ===================================================================
# OFFLINE EVALUATION
# ===================================================================


def evaluate_recommendations_offline(
    engine: RecommendationEngine,
    attempts: pd.DataFrame,
) -> dict:
    """
    Offline evaluation: does practicing recommended topics lead to improvement?

    Methodology: For each student, identify their weak areas (from first 70% of attempts).
    Check if accuracy in those areas improves in the remaining 30%.
    Compare against random (non-weak) topics as a control.
    """
    logger.info("Running offline evaluation of recommendations...")

    df = attempts.sort_values("timestamp").copy()
    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Build summary from training period
    train_summary = train.groupby(["user_id", "section", "subtopic"]).agg(
        train_accuracy=("is_correct", "mean"),
        train_n=("is_correct", "count"),
    ).reset_index()
    train_summary = train_summary[train_summary["train_n"] >= 3]

    # Get test performance
    test_summary = test.groupby(["user_id", "section", "subtopic"]).agg(
        test_accuracy=("is_correct", "mean"),
        test_n=("is_correct", "count"),
    ).reset_index()
    test_summary = test_summary[test_summary["test_n"] >= 2]

    merged = train_summary.merge(test_summary, on=["user_id", "section", "subtopic"], how="inner")

    # Weak = train accuracy < 0.5
    weak = merged[merged["train_accuracy"] < 0.5]
    strong = merged[merged["train_accuracy"] >= 0.5]

    results = {
        "n_weak_pairs": len(weak),
        "n_strong_pairs": len(strong),
        "weak_train_acc": round(weak["train_accuracy"].mean(), 4) if len(weak) > 0 else None,
        "weak_test_acc": round(weak["test_accuracy"].mean(), 4) if len(weak) > 0 else None,
        "strong_train_acc": round(strong["train_accuracy"].mean(), 4) if len(strong) > 0 else None,
        "strong_test_acc": round(strong["test_accuracy"].mean(), 4) if len(strong) > 0 else None,
    }

    if len(weak) > 0:
        results["weak_improvement"] = round(
            results["weak_test_acc"] - results["weak_train_acc"], 4
        )

    logger.info(f"  Weak areas: {results['n_weak_pairs']} pairs")
    logger.info(f"    Train accuracy: {results.get('weak_train_acc')}")
    logger.info(f"    Test accuracy: {results.get('weak_test_acc')}")
    logger.info(f"    Improvement: {results.get('weak_improvement')}")

    return results


# ===================================================================
# MAIN
# ===================================================================


def run_recommendation_engine():
    """Run the full recommendation pipeline."""
    logger.info("=" * 60)
    logger.info("Recommendation Engine — Starting")
    logger.info("=" * 60)

    # Load processed data
    attempts = load_processed_csv("attempts.csv")
    attempts["timestamp"] = pd.to_datetime(attempts["timestamp"])
    questions = load_processed_csv("questions.csv")
    topic_summary = load_processed_csv("student_topic_summary.csv")

    # Load question quality (if available)
    try:
        question_quality = load_processed_csv("question_quality_audit.csv")
    except FileNotFoundError:
        question_quality = None
        logger.warning("Question quality audit not found — skipping quality filter")

    # Build engine
    engine = RecommendationEngine(
        n_weak_subtopics=5,
        spaced_rep_gap_days=7,
        max_recommendations=20,
    )
    engine.fit(attempts, questions, topic_summary, question_quality)

    # Generate recommendations for all students
    all_recs = engine.batch_recommend()
    save_processed_csv(all_recs, "all_recommendations.csv")
    logger.info(f"\nGenerated {len(all_recs):,} total recommendations for {all_recs['user_id'].nunique()} students")

    # Sample output
    sample_user = attempts["user_id"].iloc[0]
    sample_recs = engine.recommend(sample_user)
    logger.info(f"\nSample recommendations for {sample_user}:")
    for i, rec in enumerate(sample_recs[:5], 1):
        logger.info(f"  {i}. [{rec['difficulty_tag']}] {rec['subtopic']} — {rec['reason']}")

    # Offline evaluation
    eval_results = evaluate_recommendations_offline(engine, attempts)
    eval_df = pd.DataFrame([eval_results])
    save_processed_csv(eval_df, "recommendation_evaluation.csv")

    logger.info("=" * 60)
    logger.info("Recommendation Engine — Complete")
    logger.info("=" * 60)

    return engine


if __name__ == "__main__":
    run_recommendation_engine()
