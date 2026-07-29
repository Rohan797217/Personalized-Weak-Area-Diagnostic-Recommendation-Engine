# AptiDude — Engineering Handoff Guide

## Architecture Overview

```
Raw CSVs → ETL Pipeline → Processed Tables → Models → Dashboards
                                  ↓
                         Neon PostgreSQL (optional)
```

## How to Integrate into Production

### 1. Recommendation Engine API

The recommendation engine can be wrapped as a simple API endpoint:

```python
from src.models.recommendation_engine import RecommendationEngine

# Initialize once at startup
engine = RecommendationEngine(n_weak_subtopics=5, spaced_rep_gap_days=7)
engine.fit(attempts_df, questions_df, topic_summary_df, quality_audit_df)

# Per-request
def get_recommendations(user_id: str) -> list[dict]:
    return engine.recommend(user_id)
```

**API Contract:**

```
GET /api/v1/recommendations/{user_id}

Response:
{
    "user_id": "U00001",
    "recommendations": [
        {
            "question_id": "Q01234",
            "section": "Quantitative Aptitude",
            "subtopic": "Time & Work",
            "difficulty_tag": "Easy",
            "reason": "Weak area: Time & Work (accuracy: 35%)",
            "priority": 8.5
        },
        ...
    ]
}
```

### 2. Diagnostic Endpoint

```
GET /api/v1/diagnosis/{user_id}

Response:
{
    "user_id": "U00001",
    "section_scores": {
        "Quantitative Aptitude": 0.45,
        "Logical Reasoning": 0.72,
        "Verbal Ability": 0.61,
        "Data Interpretation": 0.55
    },
    "weak_subtopics": [...],
    "strong_subtopics": [...],
    "overall_accuracy": 0.58
}
```

### 3. Retraining Pipeline

Run these commands on a schedule (recommended: weekly via cron or Prefect):

```bash
# Full pipeline refresh
python -m src.data.etl_pipeline --upload-db
python -m src.models.diagnostic_model
python -m src.quality_audit.question_discrimination
python -m src.models.recommendation_engine
```

### 4. Database (Neon PostgreSQL)

All processed tables are pushed to Neon PostgreSQL when `--upload-db` is used.

**Connection:** Set `DATABASE_URL` in `.env`:
```
DATABASE_URL=postgresql://user:pass@ep-xxxx.us-east-2.aws.neon.tech/aptidude?sslmode=require
```

**Tables in Neon:**
- `users`, `questions`, `attempts`, `mock_tests`
- `student_topic_summary`, `student_section_summary`
- `daily_engagement`
- `irt_abilities`, `irt_difficulties`
- `question_quality_audit`
- `all_recommendations`

### 5. Streamlit Deployment

Both dashboards are ready for Streamlit Community Cloud:

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo
4. Set `dashboards/founder_dashboard.py` or `dashboards/student_dashboard.py` as the entrypoint
5. Add `DATABASE_URL` to Streamlit secrets if using Neon

### 6. Key Files to Know

| File | Purpose |
|------|---------|
| `src/data/etl_pipeline.py` | Data cleaning and feature engineering |
| `src/models/diagnostic_model.py` | Weak-area diagnosis (weighted accuracy + IRT) |
| `src/quality_audit/question_discrimination.py` | Question quality flags |
| `src/models/recommendation_engine.py` | Next-best-question recommendations |
| `dashboards/founder_dashboard.py` | Founder analytics (Streamlit) |
| `dashboards/student_dashboard.py` | Student diagnostic report (Streamlit) |
| `src/utils/helpers.py` | Shared utilities (paths, DB connection, I/O) |

### 7. Dependencies

All listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

No paid services required. Neon PostgreSQL has a free tier sufficient for this project.
