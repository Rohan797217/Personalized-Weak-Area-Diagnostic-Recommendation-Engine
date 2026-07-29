# AptiDude — Final Presentation

## Data Science Internship Project: Personalized Weak-Area Diagnostic & Recommendation Engine

---

## Problem Statement

Students on AptiDude practice thousands of questions but get **no personalized signal** on:
- **Where** they're actually weak
- **What** to practice next
- **Which questions** are poorly designed

---

## What We Built

### End-to-End Data Science Pipeline

```
Raw Data (100K+ attempts) --> ETL Pipeline --> Diagnostic Model --> Recommendations
                                   |                                     |
                                   v                                     v
                          Question Quality Audit              Founder Dashboard
                                                              Student Dashboard
```

### Components

| Component | Description |
|-----------|-------------|
| **ETL Pipeline** | Cleans, validates, and engineers features from 4 raw tables |
| **Diagnostic Model** | Weighted accuracy + IRT-lite identifies per-student weak areas |
| **Question Quality Audit** | Point-biserial discrimination flags 2,421 bad questions |
| **Recommendation Engine** | Personalized next-best questions with difficulty progression |
| **Founder Dashboard** | Cohort analytics, question flags, engagement trends |
| **Student Dashboard** | Individual diagnostic report with recommendations |

---

## Data Overview

| Metric | Value |
|--------|-------|
| Students | 600 |
| Questions | 10,500 (4 sections, 61 subtopics) |
| Attempts | 100,807 |
| Mock Tests | 4,379 |
| Date Range | 422 days |

---

## Key Results

### Diagnostic Accuracy (Validated Against Ground Truth)

| Section | Pearson r | Significance |
|---------|-----------|--------------|
| Quantitative Aptitude | **0.552** | p < 0.0001 |
| Logical Reasoning | **0.515** | p < 0.0001 |
| Verbal Ability | **0.526** | p < 0.0001 |
| Data Interpretation | **0.408** | p < 0.0001 |
| IRT Ability (overall) | **0.562** | p < 0.0001 |

> The model successfully recovers latent student abilities from attempt data alone.

### Question Quality

- **2,421 questions (23%)** need review
  - 1,804 with negative discrimination
  - 617 with low discrimination
- **~33% of difficulty tags** are inaccurate

### Recommendation Impact (Offline Evaluation)

- Weak-area accuracy improved by **+24.8%** from training to test period
- Engine generates ~18 personalized recommendations per student
- Quality filter excludes 1,804 problematic questions from recommendations

---

## EDA Highlights

1. **Guessing rate**: 1.6% of attempts (fast + wrong)
2. **Hardest subtopics**: Number System, Progressions, Geometry
3. **Mock progression**: Scores trend upward across sequential mocks
4. **Data quality**: Zero missing values, zero orphan records — clean data

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.11 |
| Data | pandas, numpy, scipy, scikit-learn |
| Database | Neon PostgreSQL |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Dashboards

### Founder Dashboard
- 5 KPI metric cards
- Cohort weak-topic heatmap
- Question quality flags & discrimination distributions
- Engagement trends (DAU, accuracy, guessing)
- Mock test analytics
- Model validation results

### Student Dashboard
- Personal radar chart (section-level strengths)
- Weak/strong area cards with accuracy scores
- Subtopic-level accuracy breakdown (color-coded)
- Mock test score progression
- Practice activity timeline
- Top 10 personalized question recommendations

---

## Actionable Recommendations

| Priority | Recommendation | Expected Impact |
|----------|---------------|-----------------|
| 🔴 High | Review 1,804 negatively-discriminating questions | Content quality ↑ |
| 🔴 High | Fix ~3,400 mistagged difficulty levels | Student trust ↑ |
| 🟡 Medium | Deploy recommendation engine in the live product | Engagement ↑ |
| 🟡 Medium | Surface diagnostic reports to students | Learning outcomes ↑ |
| 🟢 Low | Add collaborative filtering to recommendations | Personalization ↑ |

---

## Engineering Integration

### API Contracts Ready

```
GET /api/v1/recommendations/{user_id}    → ranked question list
GET /api/v1/diagnosis/{user_id}          → weak-area profile
```

### Retraining Cadence
- Recommended: **weekly** pipeline refresh
- Full pipeline: ~10 seconds on current data scale

### Repository
- Clean, documented code with docstrings
- 15 unit tests (all passing)
- Data dictionary, model card, handoff guide

---

## Future Work

1. **2-parameter IRT model** (add discrimination parameter per question)
2. **Collaborative filtering** for recommendations
3. **A/B test** recommendation engine with 50-100 users
4. **Real-time scoring** with streaming attempt data
5. **Learning path optimization** using reinforcement learning

---

*Full code: GitHub repository*
*Live dashboards: Streamlit Community Cloud*
*Contact: [Founder / DS Intern]*
