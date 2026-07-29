# AptiDude — Week 2 EDA Summary (1-Page Findings for Founders)

## Dataset Overview
- **600 students** | **10,500 questions** | **100,807 attempts** | **4,379 mock tests**
- 4 sections: Quantitative Aptitude, Logical Reasoning, Verbal Ability, Data Interpretation
- 61 unique subtopics across sections
- Date range: June 2025 – July 2026 (422 days)

---

## Key Findings

### 1. Student Engagement
- **Median attempts per student: ~168** — healthy engagement level
- Engagement split: roughly even across low/medium/high
- Target exams: CAT and Placements are the primary segments
- Most practice happens during daytime hours (9AM-6PM), with an evening spike

### 2. Topic Difficulty Patterns
- **Hardest sections:** Quantitative Aptitude and Data Interpretation have the lowest average accuracy
- **Hardest subtopics:** Number System, Progressions (AP/GP), and Geometry consistently trip up students
- **Easiest subtopics:** Vocabulary, Sentence Completion, and Averages have highest accuracy
- There's a clear difficulty progression (Easy → Medium → Hard) that matches reality

### 3. Difficulty Tag Accuracy
- **~67% of questions are correctly tagged** by difficulty
- ~17% are **easier than tagged** (students find them simpler than expected)
- ~16% are **harder than tagged** (students struggle more than the tag suggests)
- **Recommendation:** Review the ~3,400 mistagged questions to improve platform reliability

### 4. Guessing Behavior
- **1.6% of all attempts** are flagged as guessing (answered in < 10 seconds + wrong)
- Guessing rates are higher on Hard questions (as expected)
- Some students have guessing rates above 5% — could indicate disengagement or fatigue

### 5. Question Quality
- **2,421 questions (23%)** flagged with poor discrimination index
  - 1,804 with negative discrimination (strong students fail as often as weak ones)
  - 617 with low discrimination (doesn't differentiate well)
- **Recommendation:** Review flagged questions — many may be ambiguous, poorly worded, or have wrong answer keys

### 6. Mock Test Progression
- Average overall score: ~40-45 (out of ~70 max)
- **Scores show moderate improvement** across sequential mocks — the platform is working
- DILR section has the highest average scores; QA has the lowest
- Section scores are moderately correlated with each other

### 7. Data Quality
- ✅ No missing values in any table
- ✅ No duplicate IDs
- ✅ All foreign keys are valid
- ✅ No extreme time outliers (> 600 seconds)
- Data is clean and analytics-ready

---

## Actionable Recommendations for Founders

| Priority | Action | Impact |
|----------|--------|--------|
| 🔴 High | Review the 1,804 questions with negative discrimination | Immediate content quality improvement |
| 🔴 High | Fix ~3,400 mistagged difficulty levels | Better student experience and accuracy of recommendations |
| 🟡 Medium | Add more questions for under-served subtopics | Fill content gaps in the question bank |
| 🟡 Medium | Flag high-guessing students for targeted engagement | Reduce drop-off and improve learning outcomes |
| 🟢 Low | Consider time limits per difficulty to reduce guessing | Encourage more thoughtful practice |

---

*Generated from EDA notebook: `notebooks/01_eda.py`*
*Data pipeline: `src/data/etl_pipeline.py`*
