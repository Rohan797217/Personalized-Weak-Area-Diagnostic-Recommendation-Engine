"""
Diagnostic Model for AptiDude.

Two-tier approach to diagnosing student weak areas:
1. Weighted Accuracy Model — recency-weighted accuracy per subtopic
2. IRT-lite Model — logistic regression separating student ability from question difficulty

Validates against ground-truth skill scores when available.

Usage:
    python -m src.models.diagnostic_model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

from src.utils.helpers import (
    get_logger,
    load_processed_csv,
    save_processed_csv,
    DATASET_DIR,
)

logger = get_logger("diagnostic_model")


# ===================================================================
# TIER 1: Weighted Accuracy Diagnostic
# ===================================================================


class WeightedAccuracyDiagnostic:
    """
    Diagnoses student weak areas using recency-weighted accuracy.

    For each student, computes a weighted accuracy score per section and subtopic
    where recent attempts are weighted more heavily (exponential decay).
    """

    def __init__(self, weak_threshold: float = 0.5, min_attempts: int = 3):
        """
        Args:
            weak_threshold: accuracy below this = weak area
            min_attempts: minimum attempts in a subtopic to count it
        """
        self.weak_threshold = weak_threshold
        self.min_attempts = min_attempts
        self.student_profiles = {}
        self.topic_summary = None
        self.section_summary = None

    def fit(self, topic_summary: pd.DataFrame, section_summary: pd.DataFrame):
        """Build diagnostic profiles from the pre-computed summaries."""
        logger.info("Fitting Weighted Accuracy Diagnostic...")

        self.topic_summary = topic_summary.copy()
        self.section_summary = section_summary.copy()

        # Filter to subtopics with enough attempts
        valid = self.topic_summary[self.topic_summary["total_attempts"] >= self.min_attempts]

        for user_id in valid["user_id"].unique():
            user_topics = valid[valid["user_id"] == user_id].copy()
            user_sections = self.section_summary[self.section_summary["user_id"] == user_id]

            # Identify weak subtopics (below threshold)
            weak_subtopics = user_topics[
                user_topics["weighted_accuracy"] < self.weak_threshold
            ].sort_values("weighted_accuracy")

            # Identify strong subtopics
            strong_subtopics = user_topics[
                user_topics["weighted_accuracy"] >= self.weak_threshold
            ].sort_values("weighted_accuracy", ascending=False)

            # Section-level strengths/weaknesses
            section_scores = dict(
                zip(user_sections["section"], user_sections["avg_weighted_accuracy"])
            )

            self.student_profiles[user_id] = {
                "section_scores": section_scores,
                "weak_subtopics": weak_subtopics[["section", "subtopic", "weighted_accuracy", "total_attempts"]].to_dict("records"),
                "strong_subtopics": strong_subtopics[["section", "subtopic", "weighted_accuracy", "total_attempts"]].head(5).to_dict("records"),
                "num_weak": len(weak_subtopics),
                "num_strong": len(strong_subtopics),
                "overall_accuracy": user_topics["weighted_accuracy"].mean(),
            }

        logger.info(f"  Profiles built for {len(self.student_profiles)} students")
        return self

    def diagnose(self, user_id: str) -> dict:
        """Get the diagnostic profile for a student."""
        if user_id not in self.student_profiles:
            return {"error": f"No profile found for {user_id}"}
        return self.student_profiles[user_id]

    def get_weakest_subtopics(self, user_id: str, top_n: int = 5) -> list[dict]:
        """Get the top N weakest subtopics for a student."""
        profile = self.diagnose(user_id)
        if "error" in profile:
            return []
        return profile["weak_subtopics"][:top_n]

    def get_cohort_weak_topics(self) -> pd.DataFrame:
        """
        Identify which topics trip up the most students — product insight for founders.
        Returns a DataFrame of (section, subtopic, pct_students_weak, avg_accuracy).
        """
        valid = self.topic_summary[
            self.topic_summary["total_attempts"] >= self.min_attempts
        ].copy()

        cohort = valid.groupby(["section", "subtopic"]).agg(
            num_students=("user_id", "nunique"),
            avg_accuracy=("weighted_accuracy", "mean"),
            median_accuracy=("weighted_accuracy", "median"),
            num_weak=("weighted_accuracy", lambda x: (x < self.weak_threshold).sum()),
        ).reset_index()

        total_students = valid["user_id"].nunique()
        cohort["pct_students_weak"] = (cohort["num_weak"] / total_students * 100).round(1)
        cohort = cohort.sort_values("pct_students_weak", ascending=False)
        return cohort


# ===================================================================
# TIER 2: IRT-lite Diagnostic (Logistic Regression)
# ===================================================================


class IRTLiteDiagnostic:
    """
    IRT-lite model using logistic regression.

    Models P(correct) = sigmoid(student_ability - question_difficulty)
    by encoding students and questions as features in a logistic regression.

    This separates "the student is weak" from "the question is hard" —
    the key insight that simple accuracy conflates.
    """

    def __init__(self, regularization: float = 1.0):
        self.regularization = regularization
        self.model = None
        self.user_encoder = LabelEncoder()
        self.question_encoder = LabelEncoder()
        self.user_abilities = {}
        self.question_difficulties = {}

    def fit(self, attempts: pd.DataFrame):
        """
        Fit the IRT-lite model on attempt data.

        Uses logistic regression with one-hot encoded student and question IDs.
        Student coefficients ≈ ability, question coefficients ≈ -difficulty.
        """
        logger.info("Fitting IRT-lite Diagnostic (logistic regression)...")

        df = attempts[["user_id", "question_id", "is_correct"]].dropna().copy()

        # Encode IDs
        df["user_idx"] = self.user_encoder.fit_transform(df["user_id"])
        df["question_idx"] = self.question_encoder.fit_transform(df["question_id"])

        n_users = len(self.user_encoder.classes_)
        n_questions = len(self.question_encoder.classes_)
        logger.info(f"  {n_users} users × {n_questions} questions = {len(df):,} attempts")

        # Build sparse feature matrix: [user_one_hot | question_one_hot]
        from scipy.sparse import lil_matrix

        X = lil_matrix((len(df), n_users + n_questions), dtype=np.float32)
        for i, (uid, qid) in enumerate(zip(df["user_idx"], df["question_idx"])):
            X[i, uid] = 1.0          # student ability (positive = stronger)
            X[i, n_users + qid] = -1.0  # question difficulty (negative = harder)

        y = df["is_correct"].astype(int).values

        # Fit logistic regression
        self.model = LogisticRegression(
            C=self.regularization,
            max_iter=500,
            solver="lbfgs",
            warm_start=False,
        )
        self.model.fit(X.tocsr(), y)

        # Extract abilities and difficulties from coefficients
        coefs = self.model.coef_[0]
        for i, uid in enumerate(self.user_encoder.classes_):
            self.user_abilities[uid] = float(coefs[i])
        for i, qid in enumerate(self.question_encoder.classes_):
            self.question_difficulties[qid] = float(-coefs[n_users + i])  # negate

        logger.info(f"  Model fit complete. Intercept: {self.model.intercept_[0]:.3f}")
        logger.info(f"  Ability range: [{min(self.user_abilities.values()):.3f}, {max(self.user_abilities.values()):.3f}]")
        logger.info(f"  Difficulty range: [{min(self.question_difficulties.values()):.3f}, {max(self.question_difficulties.values()):.3f}]")

        return self

    def get_student_ability(self, user_id: str) -> float | None:
        """Get the estimated ability for a student."""
        return self.user_abilities.get(user_id)

    def get_question_difficulty(self, question_id: str) -> float | None:
        """Get the estimated difficulty for a question."""
        return self.question_difficulties.get(question_id)

    def get_abilities_df(self) -> pd.DataFrame:
        """Return all student abilities as a DataFrame."""
        return pd.DataFrame(
            list(self.user_abilities.items()),
            columns=["user_id", "irt_ability"]
        ).sort_values("irt_ability", ascending=False)

    def get_difficulties_df(self) -> pd.DataFrame:
        """Return all question difficulties as a DataFrame."""
        return pd.DataFrame(
            list(self.question_difficulties.items()),
            columns=["question_id", "irt_difficulty"]
        ).sort_values("irt_difficulty", ascending=False)


# ===================================================================
# VALIDATION AGAINST GROUND TRUTH
# ===================================================================


def validate_against_ground_truth(
    diagnostic: WeightedAccuracyDiagnostic,
    irt: IRTLiteDiagnostic,
    section_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the diagnostic model against ground-truth skill scores.

    Computes correlation between:
    - Weighted accuracy (per section) vs. true skill scores
    - IRT ability vs. true skill scores
    """
    logger.info("=" * 60)
    logger.info("VALIDATION — Comparing against ground truth")
    logger.info("=" * 60)

    # Load ground truth
    gt_users = pd.read_csv(DATASET_DIR / "users_with_ground_truth.csv")
    gt_questions = pd.read_csv(DATASET_DIR / "questions_with_ground_truth.csv")

    # --- User ability validation ---
    # Map section names to ground truth columns
    section_to_gt = {
        "Quantitative Aptitude": "_skill_quant",
        "Logical Reasoning": "_skill_lr",
        "Verbal Ability": "_skill_va",
        "Data Interpretation": "_skill_di",
    }

    results = []

    # Weighted accuracy vs ground truth (per section)
    for section, gt_col in section_to_gt.items():
        sec_data = section_summary[section_summary["section"] == section][
            ["user_id", "avg_weighted_accuracy"]
        ].copy()

        merged = sec_data.merge(gt_users[["user_id", gt_col]], on="user_id", how="inner")
        if len(merged) < 10:
            continue

        corr, pval = stats.pearsonr(merged["avg_weighted_accuracy"], merged[gt_col])
        spearman, sp_pval = stats.spearmanr(merged["avg_weighted_accuracy"], merged[gt_col])

        results.append({
            "metric": f"Weighted Accuracy ({section})",
            "comparison": f"vs {gt_col}",
            "pearson_r": round(corr, 4),
            "pearson_p": round(pval, 6),
            "spearman_r": round(spearman, 4),
            "spearman_p": round(sp_pval, 6),
            "n_students": len(merged),
        })
        logger.info(
            f"  {section}: Pearson r={corr:.3f} (p={pval:.4f}), "
            f"Spearman rho={spearman:.3f} (p={sp_pval:.4f}), n={len(merged)}"
        )

    # IRT ability vs ground truth (overall)
    if irt.user_abilities:
        irt_df = irt.get_abilities_df()
        # Compare IRT ability against average ground truth skill
        gt_users["_avg_skill"] = gt_users[
            ["_skill_quant", "_skill_lr", "_skill_va", "_skill_di"]
        ].mean(axis=1)

        merged_irt = irt_df.merge(gt_users[["user_id", "_avg_skill"]], on="user_id", how="inner")
        if len(merged_irt) >= 10:
            corr, pval = stats.pearsonr(merged_irt["irt_ability"], merged_irt["_avg_skill"])
            spearman, sp_pval = stats.spearmanr(merged_irt["irt_ability"], merged_irt["_avg_skill"])
            results.append({
                "metric": "IRT Ability (overall)",
                "comparison": "vs avg ground truth skill",
                "pearson_r": round(corr, 4),
                "pearson_p": round(pval, 6),
                "spearman_r": round(spearman, 4),
                "spearman_p": round(sp_pval, 6),
                "n_students": len(merged_irt),
            })
            logger.info(
                f"  IRT overall: Pearson r={corr:.3f} (p={pval:.4f}), "
                f"Spearman rho={spearman:.3f} (p={sp_pval:.4f}), n={len(merged_irt)}"
            )

    # --- Question difficulty validation ---
    if irt.question_difficulties:
        diff_df = irt.get_difficulties_df()
        merged_q = diff_df.merge(
            gt_questions[["question_id", "_true_correct_prob"]],
            on="question_id", how="inner"
        )
        if len(merged_q) >= 10:
            # Higher true_correct_prob = easier, so we expect negative correlation with IRT difficulty
            corr, pval = stats.pearsonr(merged_q["irt_difficulty"], merged_q["_true_correct_prob"])
            results.append({
                "metric": "IRT Question Difficulty",
                "comparison": "vs true_correct_prob",
                "pearson_r": round(corr, 4),
                "pearson_p": round(pval, 6),
                "spearman_r": round(stats.spearmanr(merged_q["irt_difficulty"], merged_q["_true_correct_prob"])[0], 4),
                "spearman_p": round(stats.spearmanr(merged_q["irt_difficulty"], merged_q["_true_correct_prob"])[1], 6),
                "n_questions": len(merged_q),
            })
            logger.info(
                f"  Question difficulty: Pearson r={corr:.3f} (p={pval:.4f}), n={len(merged_q)}"
            )

    validation_df = pd.DataFrame(results)
    save_processed_csv(validation_df, "validation_results.csv")
    logger.info(f"\nValidation Results:\n{validation_df.to_string(index=False)}")
    return validation_df


# ===================================================================
# WEAK AREA PREDICTION (held-out validation)
# ===================================================================


def validate_weak_area_prediction(attempts: pd.DataFrame) -> dict:
    """
    Validate: does weak-area diagnosis predict future performance?

    Split attempts chronologically: use first 70% to diagnose weak areas,
    then check if accuracy in those areas is indeed lower in the last 30%.
    """
    logger.info("Validating weak-area prediction on held-out data...")

    df = attempts.sort_values("timestamp").copy()

    # Chronological split
    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Diagnose weak areas from training period
    train_summary = train.groupby(["user_id", "section", "subtopic"]).agg(
        accuracy=("is_correct", "mean"),
        n_attempts=("is_correct", "count"),
    ).reset_index()

    # Only consider subtopics with ≥3 training attempts
    train_summary = train_summary[train_summary["n_attempts"] >= 3]

    # Label: weak if accuracy < 0.5 in training period
    train_summary["is_weak"] = train_summary["accuracy"] < 0.5

    # Check test-period accuracy for these same (user, subtopic) pairs
    test_summary = test.groupby(["user_id", "section", "subtopic"]).agg(
        test_accuracy=("is_correct", "mean"),
        test_n=("is_correct", "count"),
    ).reset_index()

    merged = train_summary.merge(test_summary, on=["user_id", "section", "subtopic"], how="inner")
    merged = merged[merged["test_n"] >= 2]  # Need at least 2 test attempts

    if len(merged) == 0:
        logger.warning("  Not enough overlapping data for validation")
        return {}

    # Weak areas should have lower test accuracy than strong areas
    weak_test_acc = merged[merged["is_weak"]]["test_accuracy"].mean()
    strong_test_acc = merged[~merged["is_weak"]]["test_accuracy"].mean()

    # Classification: predict is_weak based on test accuracy
    merged["test_is_weak"] = merged["test_accuracy"] < 0.5

    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(merged["is_weak"], merged["test_is_weak"])
    precision = precision_score(merged["is_weak"], merged["test_is_weak"], zero_division=0)
    recall = recall_score(merged["is_weak"], merged["test_is_weak"], zero_division=0)
    f1 = f1_score(merged["is_weak"], merged["test_is_weak"], zero_division=0)

    results = {
        "n_pairs_evaluated": len(merged),
        "weak_test_accuracy": round(weak_test_acc, 4),
        "strong_test_accuracy": round(strong_test_acc, 4),
        "accuracy_gap": round(strong_test_acc - weak_test_acc, 4),
        "prediction_accuracy": round(acc, 4),
        "prediction_precision": round(precision, 4),
        "prediction_recall": round(recall, 4),
        "prediction_f1": round(f1, 4),
    }

    logger.info(f"  Pairs evaluated: {results['n_pairs_evaluated']}")
    logger.info(f"  Weak areas -> test accuracy: {results['weak_test_accuracy']:.3f}")
    logger.info(f"  Strong areas -> test accuracy: {results['strong_test_accuracy']:.3f}")
    logger.info(f"  Gap (strong - weak): {results['accuracy_gap']:.3f}")
    logger.info(f"  Prediction accuracy: {results['prediction_accuracy']:.3f}")
    logger.info(f"  Prediction F1: {results['prediction_f1']:.3f}")

    return results


# ===================================================================
# MAIN
# ===================================================================


def run_diagnostics():
    """Run the full diagnostic pipeline."""
    logger.info("=" * 60)
    logger.info("AptiDude Diagnostic Model — Starting")
    logger.info("=" * 60)

    # Load processed data
    attempts = load_processed_csv("attempts.csv")
    attempts["timestamp"] = pd.to_datetime(attempts["timestamp"])
    topic_summary = load_processed_csv("student_topic_summary.csv")
    topic_summary["last_attempt"] = pd.to_datetime(topic_summary["last_attempt"])
    section_summary = load_processed_csv("student_section_summary.csv")

    # --- Tier 1: Weighted Accuracy ---
    diagnostic = WeightedAccuracyDiagnostic(weak_threshold=0.5, min_attempts=3)
    diagnostic.fit(topic_summary, section_summary)

    # Example diagnosis
    sample_user = attempts["user_id"].iloc[0]
    profile = diagnostic.diagnose(sample_user)
    logger.info(f"\nSample diagnosis for {sample_user}:")
    logger.info(f"  Section scores: {profile['section_scores']}")
    logger.info(f"  Weak subtopics: {profile['num_weak']}")
    logger.info(f"  Strong subtopics: {profile['num_strong']}")

    # Cohort analysis
    cohort_weak = diagnostic.get_cohort_weak_topics()
    save_processed_csv(cohort_weak, "cohort_weak_topics.csv")
    logger.info(f"\nTop 10 hardest subtopics (by % students weak):")
    logger.info(cohort_weak.head(10).to_string(index=False))

    # --- Tier 2: IRT-lite ---
    irt = IRTLiteDiagnostic(regularization=1.0)
    irt.fit(attempts)

    # Save ability and difficulty estimates
    abilities_df = irt.get_abilities_df()
    difficulties_df = irt.get_difficulties_df()
    save_processed_csv(abilities_df, "irt_abilities.csv")
    save_processed_csv(difficulties_df, "irt_difficulties.csv")

    # --- Validation ---
    validation = validate_against_ground_truth(diagnostic, irt, section_summary)
    held_out = validate_weak_area_prediction(attempts)

    # Save held-out results
    held_out_df = pd.DataFrame([held_out])
    save_processed_csv(held_out_df, "held_out_validation.csv")

    logger.info("=" * 60)
    logger.info("Diagnostic Model — Complete")
    logger.info("=" * 60)

    return diagnostic, irt


if __name__ == "__main__":
    run_diagnostics()
