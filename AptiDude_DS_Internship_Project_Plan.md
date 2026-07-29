# AptiDude — Personalized Weak-Area Diagnostic & Recommendation Engine
### Data Science Internship Project Plan (8 Weeks)

---

## 1. Project Overview

**Problem statement:** Students on AptiDude practice thousands of questions but get no personalized signal on *where* they're actually weak or *what* to practice next. This project builds a data pipeline + model + dashboard that diagnoses each student's weak topics and recommends what to practice next — while also giving the founders a cohort-level analytics view and a question-quality audit.

**Deliverables at the end of 8 weeks:**
1. A clean, documented data pipeline (raw attempt logs → analytics-ready tables)
2. A weak-area diagnostic model per student
3. A recommendation engine (next-best-questions/topics)
4. A founder-facing dashboard (cohort insights + question quality flags)
5. A student-facing prototype (individual diagnostic report)
6. Full documentation + handoff so an engineer can productionize it

**Success metrics:**
- Diagnostic accuracy: weak-area predictions match actual subsequent performance (validated on held-out data)
- Dashboard adopted/used by founders weekly
- Recommendation engine shows measurable engagement lift in a pilot (even a small A/B test with ~50–100 users)

---

## 2. Tech Stack (100% Free / Open Source)

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11 | Standard for DS work |
| Data manipulation | pandas, numpy | Core wrangling |
| Database (dev) | SQLite / PostgreSQL (free tier on [Supabase](https://supabase.com) or [Neon](https://neon.tech)) | Store attempt logs, no cost |
| Data pipeline/orchestration | Prefect (free/open-source) or plain Python + cron | Lightweight scheduling |
| Modeling | scikit-learn, statsmodels | IRT-lite, classification, clustering |
| IRT (optional advanced) | `py-irt` or `girth` (open source Python libs) | Proper difficulty/ability estimation |
| Experiment tracking | MLflow (free, self-hosted) | Track model versions/metrics |
| Visualization | Plotly, Matplotlib, Seaborn | Charts for dashboard & EDA |
| Dashboard | **Streamlit** (free, fastest to ship) or **Metabase** (free, self-hosted BI tool) | Founder + student-facing views |
| Version control | GitHub (free) | Code + project management (Issues/Projects board) |
| Notebooks | Jupyter / Google Colab (free GPU/CPU) | Exploration |
| Deployment (optional) | Streamlit Community Cloud (free) or Render free tier | Host the dashboard live |
| Docs | Notion (free tier) or Markdown in repo | Project documentation |
| Project management | GitHub Projects or Trello (free) | Sprint tracking |

No paid API keys, no paid compute — everything above has a genuinely free tier suitable for a project this size.

---

## 3. Data Requirements

### 3.1 Data needed from AptiDude's systems
Work with the engineering team to get (or generate anonymized/synthetic versions of) these tables:

**`users`**
| column | type | notes |
|---|---|---|
| user_id | string | anonymized |
| signup_date | date | |
| target_exam | string | CAT / SSC / Banking etc. |

**`questions`**
| column | type | notes |
|---|---|---|
| question_id | string | |
| topic | string | e.g. Quant |
| subtopic | string | e.g. Time & Work |
| difficulty_tag | string | as tagged manually (may be unreliable — this is something we'll validate) |
| exam_tags | string/list | which exams this question is relevant for |

**`attempts`** (the core table)
| column | type | notes |
|---|---|---|
| attempt_id | string | |
| user_id | string | FK |
| question_id | string | FK |
| is_correct | boolean | |
| time_taken_sec | int | |
| timestamp | datetime | |
| source | string | practice / daily drill / mock |

**`mock_tests`**
| column | type | notes |
|---|---|---|
| mock_id | string | |
| user_id | string | |
| section_scores | json | VARC/DILR/QA breakdown |
| overall_score | float | |
| percentile_est | float | if available |
| date | date | |

### 3.2 If real data isn't available yet
Build a **synthetic data generator** (Week 1 task) that simulates realistic attempt logs — this de-risks the whole project and lets the intern start immediately without waiting on data access/privacy approvals. This is standard industry practice for early-stage prototyping.

### 3.3 Data privacy note
Anonymize user_ids before the intern touches anything. No PII (names, emails, phone numbers) should ever be in the working dataset — this should be a hard rule the founders enforce from day one.

---

## 4. System Architecture

```
                ┌─────────────────────┐
                │  AptiDude App DB     │
                │ (attempts, users,    │
                │  questions, mocks)   │
                └──────────┬───────────┘
                           │ (extract, anonymize)
                           ▼
                ┌─────────────────────┐
                │  Staging DB          │
                │  (SQLite/Postgres)   │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │  ETL Pipeline         │
                │  (pandas / Prefect)   │
                │  - clean              │
                │  - feature engineer   │
                └──────────┬───────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
 ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
 │ Diagnostic Model│ │ Question       │ │ Recommendation    │
 │ (per-student     │ │ Quality Audit  │ │ Engine             │
 │  weak areas)     │ │ (discrimination│ │ (next best         │
 │                  │ │  index)        │ │  questions/topics) │
 └────────┬─────────┘ └───────┬────────┘ └─────────┬──────────┘
          │                    │                     │
          └────────────┬───────┴──────────┬──────────┘
                        ▼                  ▼
              ┌───────────────────┐ ┌──────────────────┐
              │ Founder Dashboard  │ │ Student Diagnostic │
              │ (Streamlit)        │ │ Report (Streamlit) │
              └───────────────────┘ └──────────────────┘
```

---

## 5. Week-by-Week Workflow

### Week 1 — Setup & Data Foundation
- Set up GitHub repo with proper structure (see Section 7)
- Set up environment: `requirements.txt`, virtual env, Jupyter
- Get access to real data OR build synthetic data generator (aim for ~500 users, 10,000 questions, ~200,000 attempts to simulate realistic scale)
- Set up staging database (SQLite is fine to start)
- **Deliverable:** working data pipeline that loads raw data into clean tables

### Week 2 — Exploratory Data Analysis (EDA)
- Distribution of attempts per user, per topic, per difficulty
- Accuracy rates by topic/subtopic
- Time-taken distributions (identify guessing patterns — very fast + wrong)
- Mock score trends over time
- Identify data quality issues (missing values, mistagged questions, outliers)
- **Deliverable:** EDA notebook + a 1-page findings summary for the founders

### Week 3–4 — Diagnostic Model
- Build topic-wise performance scores per student:
  - Simple version: weighted accuracy (recent attempts weighted higher) per subtopic
  - Advanced version (stretch goal): 1-parameter IRT model (Rasch model) to separate "question difficulty" from "student ability" — using `py-irt` or `girth`
- Build a **question quality audit**: flag questions with abnormally low discrimination index (i.e., strong students get it wrong as often as weak students — signals a bad/ambiguous question)
- Validate: does the weak-area diagnosis actually predict future performance? Hold out recent attempts as a test set.
- **Deliverable:** diagnostic model + validation report + list of flagged low-quality questions

### Week 5–6 — Recommendation Engine
- Rule-based v1: for each student, recommend questions from their weakest 3 subtopics, balanced with difficulty progression (don't just repeat the same hard questions — build confidence with a mix)
- Add a **spaced-repetition style element**: resurface previously-wrong questions after a gap
- Optional ML upgrade: collaborative filtering (which questions do similar students find helpful) if time allows
- **Deliverable:** recommendation function that takes `user_id` → returns ranked list of next questions/topics

### Week 7 — Dashboards
- **Founder dashboard (Streamlit):**
  - Cohort-level weak topics (which topics trip up the most students — product insight)
  - Question quality flags (which questions to review/retire)
  - Engagement trends (daily active users, streaks, drop-off signals)
- **Student-facing prototype (Streamlit):**
  - Individual diagnostic: "Your weak areas are X, Y, Z"
  - Suggested next questions
- **Deliverable:** two working Streamlit apps, deployed free on Streamlit Community Cloud

### Week 8 — Polish, Documentation, Handoff
- Clean code, add docstrings, write README
- Model card: what the model does, assumptions, limitations, how to retrain
- Final presentation deck (can even be a Streamlit app itself, or a simple slide deck)
- Handoff doc for engineering team: how to integrate the recommendation engine into the live product (API contract, retraining cadence)
- **Deliverable:** full GitHub repo, live dashboard links, final presentation to founders

---

## 6. Evaluation Metrics

| Component | Metric |
|---|---|
| Diagnostic model | Precision/recall of weak-area predictions vs. actual next-attempt performance |
| Question quality audit | Manual spot-check agreement (does the team agree flagged questions are actually bad?) |
| Recommendation engine | Offline: does recommended-topic accuracy improve faster than random practice, on held-out data. Online (if piloted): engagement/completion rate lift |
| Dashboard | Founder usage/adoption after handoff |

---

## 7. Repository Structure (Industry Standard)

```
aptidude-ds-internship/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/              (gitignored — never commit real user data)
│   ├── processed/
│   └── synthetic/         (synthetic data generator output)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_diagnostic_model.ipynb
│   └── 03_recommendation_engine.ipynb
│
├── src/
│   ├── data/
│   │   ├── generate_synthetic_data.py
│   │   └── etl_pipeline.py
│   ├── models/
│   │   ├── diagnostic_model.py
│   │   └── recommendation_engine.py
│   ├── quality_audit/
│   │   └── question_discrimination.py
│   └── utils/
│
├── dashboards/
│   ├── founder_dashboard.py       (Streamlit)
│   └── student_dashboard.py       (Streamlit)
│
├── tests/
│   └── test_*.py                  (basic unit tests — industry hygiene)
│
├── docs/
│   ├── model_card.md
│   ├── data_dictionary.md
│   └── handoff_guide.md
│
└── reports/
    ├── week2_eda_summary.md
    └── final_presentation.md
```

---

## 8. Weekly Rituals (Industry Practice)

- **Monday:** 15-min sprint planning — what's the goal this week
- **Wednesday:** mid-week check-in / unblock issues
- **Friday:** demo (even informal) + commit summary + update GitHub Project board
- Use **GitHub Issues** for task tracking, close them via PRs — gives the intern a real workflow habit and gives you a clean paper trail of what was done

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Real data not available/approved in time | Start with synthetic data generator in Week 1 |
| Data too sparse (few attempts per student) | Fall back to topic-level rather than subtopic-level diagnostics |
| IRT model too complex for timeline | Treat as stretch goal; weighted-accuracy model is the safe baseline |
| Scope creep | Lock scope after Week 1; anything extra goes into a "future work" backlog, not into the 8-week plan |

---

## 10. What Success Looks Like at Handoff

By the end of week 8, you should be able to:
- Enter any `user_id` into the student dashboard and see a real, sensible weak-area diagnosis and next-question recommendations
- Open the founder dashboard and see which topics/questions need attention across the whole platform
- Hand the repo to an engineer and have them understand, within an hour, how to wire the recommendation engine into the live app

This is the bar for "industry-ready" — not perfect models, but a **working, documented, reproducible system** that someone else can pick up and extend.
