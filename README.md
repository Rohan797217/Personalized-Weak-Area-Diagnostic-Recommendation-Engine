# AptiDude — Personalized Weak-Area Diagnostic & Recommendation Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Neon PostgreSQL](https://img.shields.io/badge/database-Neon_PostgreSQL-00E599.svg)](https://neon.tech)

## Overview

AptiDude's Data Science engine diagnoses each student's weak topics and recommends personalized practice — while giving founders cohort-level analytics and a question-quality audit.

### What this project does

1. **ETL Pipeline** — Cleans and transforms raw attempt logs into analytics-ready tables
2. **Diagnostic Model** — Identifies per-student weak areas using weighted accuracy + IRT-lite
3. **Question Quality Audit** — Flags ambiguous/low-discrimination questions
4. **Recommendation Engine** — Suggests next-best questions/topics per student
5. **Founder Dashboard** — Cohort insights, engagement trends, question flags
6. **Student Dashboard** — Individual diagnostic report with recommendations

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure database (Neon PostgreSQL)

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://user:password@ep-xxxx.us-east-2.aws.neon.tech/aptidude?sslmode=require
```

### 3. Run the ETL pipeline

```bash
python -m src.data.etl_pipeline
```

### 4. Launch dashboards

```bash
# Founder dashboard
streamlit run dashboards/founder_dashboard.py

# Student dashboard
streamlit run dashboards/student_dashboard.py
```

## Project Structure

```
AptiDude_intern/
├── README.md
├── requirements.txt
├── .gitignore
├── .env                    (create this — never commit)
├── DATASET/                (raw CSV data)
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/
├── src/
│   ├── data/
│   │   ├── etl_pipeline.py
│   │   └── generate_synthetic_data.py
│   ├── models/
│   │   ├── diagnostic_model.py
│   │   └── recommendation_engine.py
│   ├── quality_audit/
│   │   └── question_discrimination.py
│   └── utils/
│       └── helpers.py
├── dashboards/
│   ├── founder_dashboard.py
│   └── student_dashboard.py
├── tests/
├── docs/
└── reports/
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Data | pandas, numpy |
| Database | Neon PostgreSQL (via psycopg2 + SQLAlchemy) |
| Modeling | scikit-learn, scipy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |

## Evaluation Metrics

| Component | Metric |
|-----------|--------|
| Diagnostic model | Correlation with ground-truth skill scores |
| Question quality | Discrimination index vs. true correct probability |
| Recommendations | Offline: predicted accuracy improvement on held-out data |
| Dashboard | Adoption and usability feedback |

## License

Internal project — AptiDude confidential.
