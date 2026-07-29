# AptiDude Data Dictionary

## Raw Tables (DATASET/)

### users.csv
| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Anonymized unique student identifier (e.g., U00001) |
| `signup_date` | date | Date the student signed up |
| `target_exam` | string | Target exam: CAT, Placements, TCS NQT, etc. |
| `engagement_level` | string | Categorized engagement: low, medium, high |

### questions.csv
| Column | Type | Description |
|--------|------|-------------|
| `question_id` | string | Unique question identifier (e.g., Q00001) |
| `section` | string | Section: Quantitative Aptitude, Logical Reasoning, Verbal Ability, Data Interpretation |
| `subtopic` | string | Specific subtopic (e.g., Time & Work, Puzzles, Para Jumbles) |
| `difficulty_tag` | string | Tagged difficulty: Easy, Medium, Hard |
| `exam_tags` | string | Comma-separated exam relevance tags |

### attempts.csv
| Column | Type | Description |
|--------|------|-------------|
| `attempt_id` | string | Unique attempt identifier |
| `user_id` | string | FK → users.user_id |
| `question_id` | string | FK → questions.question_id |
| `section` | string | Denormalized from questions |
| `subtopic` | string | Denormalized from questions |
| `difficulty_tag` | string | Denormalized from questions |
| `is_correct` | boolean | Whether the student answered correctly |
| `time_taken_sec` | integer | Time taken in seconds |
| `timestamp` | datetime | When the attempt was made |
| `source` | string | practice, mock_test, daily_drill |

### mock_tests.csv
| Column | Type | Description |
|--------|------|-------------|
| `mock_id` | string | Unique mock test identifier |
| `user_id` | string | FK → users.user_id |
| `date` | date | Date of the mock test |
| `varc_score` | float | VARC section score |
| `dilr_score` | float | DILR section score |
| `qa_score` | float | QA section score |
| `overall_score` | float | Total score |
| `percentile_est` | float | Estimated percentile |

---

## Ground Truth Tables

### users_with_ground_truth.csv
Same as users.csv plus:
| Column | Type | Description |
|--------|------|-------------|
| `_skill_quant` | float [0,1] | True Quantitative Aptitude ability |
| `_skill_lr` | float [0,1] | True Logical Reasoning ability |
| `_skill_va` | float [0,1] | True Verbal Ability ability |
| `_skill_di` | float [0,1] | True Data Interpretation ability |

### questions_with_ground_truth.csv
Same as questions.csv plus:
| Column | Type | Description |
|--------|------|-------------|
| `_true_correct_prob` | float [0,1] | True probability of a random student answering correctly |

---

## Processed Tables (data/processed/)

### student_topic_summary.csv
Per-student, per-subtopic performance summary with recency weighting.
| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Student identifier |
| `section` | string | Section name |
| `subtopic` | string | Subtopic name |
| `total_attempts` | int | Total attempts in this subtopic |
| `raw_accuracy` | float | Simple accuracy (correct / total) |
| `weighted_accuracy` | float | Recency-weighted accuracy (exponential decay, 30-day half-life) |
| `avg_time_sec` | float | Average time per attempt |
| `guessing_rate` | float | Fraction of attempts flagged as guessing |
| `days_since_last` | float | Days since most recent attempt |

### student_section_summary.csv
Rolled up to section level for radar charts.

### daily_engagement.csv
Daily platform engagement metrics.

### irt_abilities.csv
IRT-estimated student ability scores.

### irt_difficulties.csv
IRT-estimated question difficulty scores.

### question_quality_audit.csv
Per-question discrimination index and quality flags.

### all_recommendations.csv
Pre-computed recommendations for all students.
