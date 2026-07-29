# AptiDude — Model Card

## Diagnostic Model

### Model 1: Weighted Accuracy Diagnostic

**Purpose:** Identify each student's weak subtopics based on their attempt history.

**Method:** Computes per-subtopic accuracy weighted by recency (exponential decay with 30-day half-life). More recent attempts have higher influence than older ones, reflecting current skill level rather than historical.

**Inputs:** Student attempt logs (attempts.csv)

**Outputs:** Per-student profile with:
- Section-level scores (0–1)
- Ranked list of weak subtopics (accuracy < 0.5)
- Ranked list of strong subtopics

**Assumptions:**
- Students with fewer than 3 attempts in a subtopic are excluded from that subtopic's diagnosis
- A 50% accuracy threshold distinguishes "weak" from "not weak" — this is configurable
- Recency half-life of 30 days is appropriate for exam prep context

**Limitations:**
- Does not separate question difficulty from student ability (a student may appear weak simply because they attempted hard questions)
- Guessing detection is heuristic (< 10 seconds + wrong)
- No confidence intervals on the accuracy estimates

---

### Model 2: IRT-lite (Logistic Regression)

**Purpose:** Separate student ability from question difficulty using a 1-parameter Item Response Theory (Rasch-like) model.

**Method:** Logistic regression with one-hot encoded student IDs and question IDs as features. Student coefficients represent ability; negated question coefficients represent difficulty.

**Formula:** `P(correct) = σ(ability_student - difficulty_question + intercept)`

**Inputs:** Attempt-level data (user_id, question_id, is_correct)

**Outputs:**
- Student ability scores (continuous, higher = stronger)
- Question difficulty scores (continuous, higher = harder)

**Assumptions:**
- All items are equally discriminating (1-parameter model)
- No guessing parameter
- Student ability is a single dimension (unidimensional)

**Limitations:**
- Does not model section-specific abilities (a single ability score per student)
- Logistic regression regularization (C=1.0) may shrink estimates
- Large feature space (n_students + n_questions) can be memory-intensive

---

## Question Quality Audit

**Purpose:** Flag questions with poor discrimination — questions that don't differentiate between strong and weak students.

**Method:** Point-biserial correlation between item score (correct/incorrect) and student total accuracy.

**Thresholds:**
- `< 0`: Negative discrimination (🔴 review immediately)
- `0 – 0.1`: Low discrimination (🟡 likely problematic)
- `0.1 – 0.2`: Moderate discrimination (🟠 acceptable)
- `≥ 0.2`: Good discrimination (🟢 effective)

**Minimum data:** Questions with < 10 attempts are marked as "insufficient_data"

---

## Recommendation Engine

**Purpose:** Recommend next-best questions and topics for each student.

**Method:** Rule-based with three components:
1. **Weak-area targeting:** Select questions from the student's weakest 5 subtopics
2. **Difficulty progression:** Within weak areas, suggest Easy → Medium → Hard (confidence building)
3. **Spaced repetition:** Resurface previously-wrong questions after 7+ day gap

**Quality filter:** Questions flagged with "negative_discrimination" are excluded from the recommendation pool.

**Output:** Ranked list of up to 20 questions with reasons.

---

## Validation

### Ground Truth Validation
- Pearson/Spearman correlation between weighted accuracy (per section) and true skill scores
- Pearson/Spearman correlation between IRT ability and average true skill
- Pearson correlation between IRT difficulty and true correct probability (expected negative)

### Held-out Validation
- Chronological 70/30 split
- Check if weak areas diagnosed from first 70% of attempts predict lower accuracy in remaining 30%
- Metrics: accuracy, precision, recall, F1 of weak-area predictions

---

## How to Retrain

```bash
# 1. Run ETL pipeline with fresh data
python -m src.data.etl_pipeline

# 2. Run diagnostic model
python -m src.models.diagnostic_model

# 3. Run question quality audit
python -m src.quality_audit.question_discrimination

# 4. Generate recommendations
python -m src.models.recommendation_engine
```

Recommended retraining cadence: weekly (as new attempt data accumulates).
