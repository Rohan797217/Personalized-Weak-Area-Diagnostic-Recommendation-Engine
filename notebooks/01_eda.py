# %% [markdown]
# # AptiDude — Exploratory Data Analysis (EDA)
# **Week 2 Deliverable**
#
# This notebook explores the AptiDude dataset to uncover patterns in student behavior,
# topic difficulty, time distributions, guessing patterns, and data quality issues.

# %% Setup
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Styling
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.family"] = "sans-serif"

from src.utils.helpers import DATA_PROCESSED_DIR, DATASET_DIR

print("Setup complete.")

# %% [markdown]
# ## 1. Load Data

# %% Load all tables
users = pd.read_csv(DATA_PROCESSED_DIR / "users.csv", parse_dates=["signup_date"])
questions = pd.read_csv(DATA_PROCESSED_DIR / "questions.csv")
attempts = pd.read_csv(DATA_PROCESSED_DIR / "attempts.csv", parse_dates=["timestamp"])
mock_tests = pd.read_csv(DATA_PROCESSED_DIR / "mock_tests.csv", parse_dates=["date"])
topic_summary = pd.read_csv(DATA_PROCESSED_DIR / "student_topic_summary.csv")
section_summary = pd.read_csv(DATA_PROCESSED_DIR / "student_section_summary.csv")

# Ground truth
gt_users = pd.read_csv(DATASET_DIR / "users_with_ground_truth.csv")
gt_questions = pd.read_csv(DATASET_DIR / "questions_with_ground_truth.csv")

print(f"Users:     {len(users):>8,}")
print(f"Questions: {len(questions):>8,}")
print(f"Attempts:  {len(attempts):>8,}")
print(f"Mocks:     {len(mock_tests):>8,}")
print(f"\nDate range: {attempts['timestamp'].min()} to {attempts['timestamp'].max()}")
print(f"Span: {(attempts['timestamp'].max() - attempts['timestamp'].min()).days} days")

# %% [markdown]
# ## 2. User Demographics

# %% Users by target exam
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Target exam distribution
exam_counts = users["target_exam"].value_counts()
axes[0].barh(exam_counts.index, exam_counts.values, color=sns.color_palette("viridis", len(exam_counts)))
axes[0].set_title("Students by Target Exam", fontweight="bold")
axes[0].set_xlabel("Count")

# Engagement level distribution
eng_counts = users["engagement_level"].value_counts()
colors = {"high": "#4ECDC4", "medium": "#FFE66D", "low": "#FF6B6B"}
axes[1].pie(eng_counts.values, labels=eng_counts.index, autopct="%1.1f%%",
            colors=[colors.get(e, "#999") for e in eng_counts.index], startangle=90)
axes[1].set_title("Engagement Level Distribution", fontweight="bold")

# Signup timeline
users["signup_month"] = users["signup_date"].dt.to_period("M").astype(str)
signup_monthly = users.groupby("signup_month").size()
axes[2].plot(range(len(signup_monthly)), signup_monthly.values, marker="o", linewidth=2, color="#667eea")
axes[2].set_xticks(range(len(signup_monthly)))
axes[2].set_xticklabels(signup_monthly.index, rotation=45, ha="right")
axes[2].set_title("Monthly Signups", fontweight="bold")
axes[2].set_ylabel("New Users")

plt.tight_layout()
plt.savefig("data/processed/eda_user_demographics.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nExam distribution:\n{exam_counts.to_string()}")
print(f"\nEngagement distribution:\n{eng_counts.to_string()}")

# %% [markdown]
# ## 3. Question Catalog Analysis

# %% Question distribution by section and subtopic
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# By section
section_counts = questions["section"].value_counts()
axes[0].barh(section_counts.index, section_counts.values, color=["#667eea", "#764ba2", "#4ECDC4", "#FFE66D"])
axes[0].set_title("Questions by Section", fontweight="bold")
axes[0].set_xlabel("Number of Questions")

# By difficulty
diff_counts = questions["difficulty_tag"].value_counts()
diff_colors = {"Easy": "#4ECDC4", "Medium": "#FFE66D", "Hard": "#FF6B6B"}
axes[1].bar(diff_counts.index, diff_counts.values,
            color=[diff_colors.get(d, "#999") for d in diff_counts.index])
axes[1].set_title("Questions by Difficulty", fontweight="bold")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("data/processed/eda_question_catalog.png", dpi=150, bbox_inches="tight")
plt.show()

# Top/bottom subtopics by question count
subtopic_counts = questions.groupby(["section", "subtopic"]).size().reset_index(name="count")
subtopic_counts = subtopic_counts.sort_values("count", ascending=False)
print(f"\nTotal unique sections: {questions['section'].nunique()}")
print(f"Total unique subtopics: {questions['subtopic'].nunique()}")
print(f"\nTop 10 subtopics by question count:")
print(subtopic_counts.head(10).to_string(index=False))
print(f"\nBottom 5 subtopics (potential gaps in question bank):")
print(subtopic_counts.tail(5).to_string(index=False))

# %% Difficulty vs true correct probability (ground truth)
fig, ax = plt.subplots(figsize=(10, 6))
for diff, color in diff_colors.items():
    subset = gt_questions[gt_questions["difficulty_tag"] == diff]
    ax.hist(subset["_true_correct_prob"], bins=30, alpha=0.6, label=diff, color=color)
ax.set_xlabel("True Correct Probability")
ax.set_ylabel("Number of Questions")
ax.set_title("True Difficulty Distribution by Tagged Difficulty", fontweight="bold")
ax.legend()
ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
plt.tight_layout()
plt.savefig("data/processed/eda_difficulty_validation.png", dpi=150, bbox_inches="tight")
plt.show()

# Check how well tags match reality
for diff in ["Easy", "Medium", "Hard"]:
    subset = gt_questions[gt_questions["difficulty_tag"] == diff]
    print(f"\n{diff}: mean true_correct_prob = {subset['_true_correct_prob'].mean():.3f} "
          f"(std={subset['_true_correct_prob'].std():.3f})")

# %% [markdown]
# ## 4. Attempt Distribution Analysis

# %% Attempts per user
attempts_per_user = attempts.groupby("user_id").size().reset_index(name="n_attempts")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram of attempts per user
axes[0].hist(attempts_per_user["n_attempts"], bins=40, color="#667eea", alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Number of Attempts")
axes[0].set_ylabel("Number of Students")
axes[0].set_title("Distribution of Attempts per Student", fontweight="bold")
axes[0].axvline(x=attempts_per_user["n_attempts"].median(), color="red", linestyle="--",
                label=f"Median: {attempts_per_user['n_attempts'].median():.0f}")
axes[0].legend()

# Attempts by section
section_attempts = attempts.groupby("section").size().reset_index(name="count").sort_values("count", ascending=False)
axes[1].barh(section_attempts["section"], section_attempts["count"],
             color=["#667eea", "#764ba2", "#4ECDC4", "#FFE66D"])
axes[1].set_title("Attempts by Section", fontweight="bold")
axes[1].set_xlabel("Total Attempts")

# Attempts by source
source_counts = attempts["source"].value_counts()
axes[2].pie(source_counts.values, labels=source_counts.index, autopct="%1.1f%%",
            colors=["#667eea", "#764ba2", "#4ECDC4"], startangle=90)
axes[2].set_title("Attempts by Source", fontweight="bold")

plt.tight_layout()
plt.savefig("data/processed/eda_attempt_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nAttempts per user stats:")
print(attempts_per_user["n_attempts"].describe().to_string())

# %% Attempts per subtopic (which subtopics are over/under-practiced?)
subtopic_attempts = attempts.groupby(["section", "subtopic"]).agg(
    n_attempts=("attempt_id", "count"),
    n_students=("user_id", "nunique"),
    accuracy=("is_correct", "mean"),
).reset_index().sort_values("n_attempts", ascending=False)

fig = px.scatter(
    subtopic_attempts,
    x="n_attempts",
    y="accuracy",
    size="n_students",
    color="section",
    hover_data=["subtopic"],
    title="Subtopic: Attempts vs Accuracy (bubble = # students)",
    labels={"n_attempts": "Total Attempts", "accuracy": "Average Accuracy"},
    color_discrete_sequence=["#667eea", "#764ba2", "#4ECDC4", "#FFE66D"],
)
fig.update_layout(template="plotly_dark", height=500)
fig.show()

# %% [markdown]
# ## 5. Accuracy Analysis by Topic & Difficulty

# %% Accuracy by section and difficulty
accuracy_matrix = attempts.groupby(["section", "difficulty_tag"])["is_correct"].mean().unstack()
accuracy_matrix = accuracy_matrix.reindex(columns=["Easy", "Medium", "Hard"])

fig, ax = plt.subplots(figsize=(10, 6))
accuracy_matrix.plot(kind="bar", ax=ax, color=["#4ECDC4", "#FFE66D", "#FF6B6B"], edgecolor="white")
ax.set_title("Accuracy by Section × Difficulty", fontweight="bold", fontsize=14)
ax.set_ylabel("Accuracy Rate")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Difficulty")
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("data/processed/eda_accuracy_by_section_difficulty.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAccuracy matrix (section × difficulty):")
print(accuracy_matrix.round(3).to_string())

# %% Top 10 hardest and easiest subtopics
subtopic_accuracy = attempts.groupby(["section", "subtopic"]).agg(
    accuracy=("is_correct", "mean"),
    n_attempts=("attempt_id", "count"),
).reset_index()
subtopic_accuracy = subtopic_accuracy[subtopic_accuracy["n_attempts"] >= 50]  # min threshold

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Hardest
hardest = subtopic_accuracy.nsmallest(15, "accuracy")
axes[0].barh(
    [f"{r['subtopic']}\n({r['section'][:5]})" for _, r in hardest.iterrows()],
    hardest["accuracy"],
    color="#FF6B6B", edgecolor="white"
)
axes[0].set_title("15 Hardest Subtopics", fontweight="bold")
axes[0].set_xlabel("Accuracy")
axes[0].set_xlim(0, 1)
axes[0].invert_yaxis()

# Easiest
easiest = subtopic_accuracy.nlargest(15, "accuracy")
axes[1].barh(
    [f"{r['subtopic']}\n({r['section'][:5]})" for _, r in easiest.iterrows()],
    easiest["accuracy"],
    color="#4ECDC4", edgecolor="white"
)
axes[1].set_title("15 Easiest Subtopics", fontweight="bold")
axes[1].set_xlabel("Accuracy")
axes[1].set_xlim(0, 1)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("data/processed/eda_hardest_easiest_subtopics.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Time-Taken Analysis & Guessing Patterns

# %% Time distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overall time distribution (capped at 300s for visibility)
capped_time = attempts["time_taken_sec"].clip(upper=300)
axes[0].hist(capped_time, bins=60, color="#667eea", alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Time Taken (seconds)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Time Distribution (capped at 300s)", fontweight="bold")
axes[0].axvline(x=10, color="red", linestyle="--", label="Guessing threshold (10s)")
axes[0].legend()

# Time by correctness
correct_times = attempts[attempts["is_correct"]]["time_taken_sec"].clip(upper=300)
wrong_times = attempts[~attempts["is_correct"]]["time_taken_sec"].clip(upper=300)
axes[1].hist(correct_times, bins=50, alpha=0.6, color="#4ECDC4", label="Correct")
axes[1].hist(wrong_times, bins=50, alpha=0.6, color="#FF6B6B", label="Wrong")
axes[1].set_xlabel("Time Taken (seconds)")
axes[1].set_title("Time by Correctness", fontweight="bold")
axes[1].legend()

# Time by difficulty
for diff, color in [("Easy", "#4ECDC4"), ("Medium", "#FFE66D"), ("Hard", "#FF6B6B")]:
    subset = attempts[attempts["difficulty_tag"] == diff]["time_taken_sec"].clip(upper=300)
    axes[2].hist(subset, bins=50, alpha=0.5, color=color, label=diff)
axes[2].set_xlabel("Time Taken (seconds)")
axes[2].set_title("Time by Difficulty", fontweight="bold")
axes[2].legend()

plt.tight_layout()
plt.savefig("data/processed/eda_time_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# Stats
print(f"\nTime taken stats (seconds):")
print(f"  Mean:   {attempts['time_taken_sec'].mean():.1f}")
print(f"  Median: {attempts['time_taken_sec'].median():.1f}")
print(f"  Std:    {attempts['time_taken_sec'].std():.1f}")
print(f"  Min:    {attempts['time_taken_sec'].min()}")
print(f"  Max:    {attempts['time_taken_sec'].max()}")

# %% Guessing pattern analysis
guessing = attempts[attempts["is_guessing"]]
print(f"\nGuessing Analysis:")
print(f"  Total guessing attempts: {len(guessing):,} ({len(guessing)/len(attempts)*100:.1f}%)")
print(f"  Avg time (guessing): {guessing['time_taken_sec'].mean():.1f}s")
print(f"  Guessing by section:")
print(attempts.groupby("section")["is_guessing"].mean().mul(100).round(1).to_string())
print(f"\n  Guessing by difficulty:")
print(attempts.groupby("difficulty_tag")["is_guessing"].mean().mul(100).round(1).to_string())

# Guessing rate per user
user_guessing = attempts.groupby("user_id").agg(
    guessing_rate=("is_guessing", "mean"),
    total=("is_guessing", "count"),
).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(user_guessing["guessing_rate"] * 100, bins=30, color="#FF6B6B", alpha=0.8, edgecolor="white")
ax.set_xlabel("Guessing Rate (%)")
ax.set_ylabel("Number of Students")
ax.set_title("Distribution of Guessing Rates Across Students", fontweight="bold")
ax.axvline(x=user_guessing["guessing_rate"].mean() * 100, color="red", linestyle="--",
           label=f"Mean: {user_guessing['guessing_rate'].mean()*100:.1f}%")
ax.legend()
plt.tight_layout()
plt.savefig("data/processed/eda_guessing_rates.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Temporal Patterns

# %% Practice patterns by hour and day
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# By hour of day
hourly = attempts.groupby("hour_of_day").agg(
    count=("attempt_id", "count"),
    accuracy=("is_correct", "mean"),
).reset_index()

ax1 = axes[0]
ax2 = ax1.twinx()
ax1.bar(hourly["hour_of_day"], hourly["count"], color="#667eea", alpha=0.6, label="Attempts")
ax2.plot(hourly["hour_of_day"], hourly["accuracy"], color="#FF6B6B", linewidth=2, marker="o", label="Accuracy")
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Number of Attempts", color="#667eea")
ax2.set_ylabel("Accuracy", color="#FF6B6B")
ax1.set_title("Activity by Hour of Day", fontweight="bold")

# By day of week
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily = attempts.groupby("day_of_week").agg(
    count=("attempt_id", "count"),
    accuracy=("is_correct", "mean"),
).reset_index()

ax3 = axes[1]
ax4 = ax3.twinx()
ax3.bar(daily["day_of_week"], daily["count"], color="#764ba2", alpha=0.6, label="Attempts")
ax4.plot(daily["day_of_week"], daily["accuracy"], color="#4ECDC4", linewidth=2, marker="o", label="Accuracy")
ax3.set_xlabel("Day of Week")
ax3.set_xticks(range(7))
ax3.set_xticklabels(day_names)
ax3.set_ylabel("Number of Attempts", color="#764ba2")
ax4.set_ylabel("Accuracy", color="#4ECDC4")
ax3.set_title("Activity by Day of Week", fontweight="bold")

plt.tight_layout()
plt.savefig("data/processed/eda_temporal_patterns.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Mock Test Analysis

# %% Mock test trends
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Score distribution
axes[0].hist(mock_tests["overall_score"], bins=30, color="#667eea", alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Overall Score")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Mock Test Score Distribution", fontweight="bold")
axes[0].axvline(x=mock_tests["overall_score"].mean(), color="red", linestyle="--",
                label=f"Mean: {mock_tests['overall_score'].mean():.1f}")
axes[0].legend()

# Score improvement over mock sequence
mock_trend = mock_tests.groupby("mock_seq").agg(
    avg_score=("overall_score", "mean"),
    avg_percentile=("percentile_est", "mean"),
    n_students=("user_id", "nunique"),
).reset_index()
mock_trend = mock_trend[mock_trend["mock_seq"] <= 15]  # First 15 mocks

axes[1].plot(mock_trend["mock_seq"], mock_trend["avg_score"], marker="o", color="#667eea",
             linewidth=2, label="Avg Score")
ax_r = axes[1].twinx()
ax_r.plot(mock_trend["mock_seq"], mock_trend["avg_percentile"], marker="s", color="#4ECDC4",
          linewidth=2, label="Avg Percentile")
axes[1].set_xlabel("Mock Test Number")
axes[1].set_ylabel("Average Score", color="#667eea")
ax_r.set_ylabel("Average Percentile", color="#4ECDC4")
axes[1].set_title("Score Progression Over Mocks", fontweight="bold")
axes[1].legend(loc="upper left")
ax_r.legend(loc="upper right")

plt.tight_layout()
plt.savefig("data/processed/eda_mock_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# Section score correlations
section_cols = ["varc_score", "dilr_score", "qa_score"]
corr = mock_tests[section_cols + ["overall_score"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, ax=ax, fmt=".2f",
            xticklabels=["VARC", "DILR", "QA", "Overall"],
            yticklabels=["VARC", "DILR", "QA", "Overall"])
ax.set_title("Mock Section Score Correlations", fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/eda_mock_correlations.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Student Ability Distribution (Ground Truth)

# %% Ground truth skill distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
skill_cols = {"_skill_quant": "Quantitative Aptitude", "_skill_lr": "Logical Reasoning",
              "_skill_va": "Verbal Ability", "_skill_di": "Data Interpretation"}

for ax, (col, name) in zip(axes.flatten(), skill_cols.items()):
    ax.hist(gt_users[col], bins=30, color="#667eea", alpha=0.8, edgecolor="white")
    ax.set_title(f"{name} (Ground Truth)", fontweight="bold")
    ax.set_xlabel("Skill Level (0-1)")
    ax.set_ylabel("Students")
    ax.axvline(x=gt_users[col].mean(), color="red", linestyle="--",
               label=f"Mean: {gt_users[col].mean():.3f}")
    ax.legend()

plt.tight_layout()
plt.savefig("data/processed/eda_ground_truth_skills.png", dpi=150, bbox_inches="tight")
plt.show()

# Skill correlations
skill_corr = gt_users[list(skill_cols.keys())].corr()
print("Ground truth skill correlations:")
print(skill_corr.round(3).to_string())

# %% [markdown]
# ## 10. Data Quality Assessment

# %% Data quality checks
print("=" * 60)
print("DATA QUALITY ASSESSMENT")
print("=" * 60)

# Missing values
print("\n--- Missing Values ---")
for name, df in [("users", users), ("questions", questions), ("attempts", attempts), ("mock_tests", mock_tests)]:
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n{name}:")
        print(missing[missing > 0].to_string())
    else:
        print(f"{name}: No missing values")

# Duplicate checks
print("\n--- Duplicate Checks ---")
print(f"Duplicate attempt_ids: {attempts['attempt_id'].duplicated().sum()}")
print(f"Duplicate user_ids: {users['user_id'].duplicated().sum()}")
print(f"Duplicate question_ids: {questions['question_id'].duplicated().sum()}")

# FK integrity (already validated in ETL, confirming)
print("\n--- Foreign Key Integrity ---")
orphan_users_in_attempts = ~attempts["user_id"].isin(users["user_id"])
orphan_questions_in_attempts = ~attempts["question_id"].isin(questions["question_id"])
print(f"Orphan user_ids in attempts: {orphan_users_in_attempts.sum()}")
print(f"Orphan question_ids in attempts: {orphan_questions_in_attempts.sum()}")

# Time outliers
print("\n--- Time Outliers ---")
print(f"Attempts < 3 seconds: {(attempts['time_taken_sec'] < 3).sum()}")
print(f"Attempts > 300 seconds: {(attempts['time_taken_sec'] > 300).sum()}")
print(f"Attempts > 600 seconds: {(attempts['time_taken_sec'] > 600).sum()}")

# Engagement by exam
print("\n--- Engagement by Exam ---")
user_attempts_count = attempts.groupby("user_id").size().reset_index(name="n_attempts")
merged = users.merge(user_attempts_count, on="user_id", how="left")
print(merged.groupby("target_exam")["n_attempts"].describe().round(0).to_string())

# %% [markdown]
# ## Summary of Key Findings
#
# See `reports/week2_eda_summary.md` for the formatted 1-page summary.
#
# **Key insights:**
# 1. 600 students, 10,500 questions, ~100K attempts across 4 sections and 61 subtopics
# 2. Guessing rate is ~1.6% (fast + wrong) — mostly in harder questions
# 3. Difficulty tags are roughly correct but ~33% of questions are mistagged
# 4. Weak areas cluster in specific subtopics — strong signal for the diagnostic model
# 5. Mock scores show moderate improvement over attempts (scores trend upward)
# 6. No significant data quality issues — clean dataset
# 7. Ground truth correlations confirm the diagnostic model has valid signal (r=0.4-0.55)
