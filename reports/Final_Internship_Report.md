# Final Internship Project Report
**Project Title:** Personalized Weak-Area Diagnostic & Recommendation Engine
**Intern:** Rohan
**Company/Platform:** AptiDude (EdTech Platform for Aptitude Preparation)
**Date:** July 2026

---

## 1. Executive Summary
AptiDude is an EdTech platform aimed at helping students prepare for competitive exams (CAT, Placements, TCS NQT). While students generate massive amounts of data by attempting questions, they previously lacked a personalized feedback loop. This project successfully engineered an end-to-end Data Science pipeline that ingests raw student attempt logs, accurately diagnoses subtopic-level weak areas, flags poor-quality questions for content teams, and serves highly personalized, adaptive question recommendations to improve learning outcomes.

## 2. Problem Statement (The "Why")
Before this project, the platform suffered from the "blind practice" problem:
1. **No Adaptive Feedback:** Students did not know exactly *where* they were weak. They wasted time practicing strong areas rather than addressing critical gaps.
2. **Cold Start for Study Sessions:** Students did not know *what to practice next*, leading to choice paralysis and drop-offs.
3. **Hidden Content Issues:** Content creators had no data-driven way to identify poorly worded or excessively tricky questions that were frustrating students.

## 3. Project Goals
1. **Build a Diagnostic Engine:** Create a mathematical model to accurately calculate a student's true proficiency in 61 different subtopics.
2. **Build a Recommendation Engine:** Develop a system to suggest the "Next Best Question" to optimize learning efficiency (incorporating difficulty scaling and spaced repetition).
3. **Automate Content Auditing:** Flag problematic questions automatically so the academic team can review them.
4. **Data Visualization:** Build interactive dashboards for both the Founders (macro-level analytics) and the Students (micro-level diagnostics).

## 4. Tech Stack & Technologies Used
- **Programming Language:** Python 3.11
- **Data Engineering:** `pandas`, `numpy` (Extract, Transform, Load pipeline)
- **Machine Learning & Stats:** `scikit-learn` (Logistic Regression), `scipy` (Statistics)
- **Database:** Neon PostgreSQL (Serverless Cloud DB)
- **Data Visualization:** `plotly.express`, `plotly.graph_objects`, `matplotlib`, `seaborn`
- **Frontend / Deployment:** Streamlit Community Cloud
- **Version Control:** Git & GitHub

## 5. Methodology & Implementation

### Phase 1: Data Pipeline (ETL)
Processed over 100,000 raw attempt logs across 600 students. The pipeline cleans the data, engineers new features (such as `is_guessing` and `time_taken`), validates referential integrity, and pushes aggregated summaries directly into the Neon PostgreSQL cloud database.

### Phase 2: Exploratory Data Analysis (EDA)
Uncovered critical insights from the raw data:
- Identified that 1.6% of attempts were random guesses.
- Found that ~33% of questions were incorrectly tagged by difficulty by the human creators.
- Mapped out peak engagement hours and score progressions across sequential mock exams.

### Phase 3: The Diagnostic Models
Implemented two models to evaluate student proficiency:
1. **Weighted Accuracy Model:** Calculates subtopic strength using an exponential time-decay function (recent attempts carry a higher weight than older ones, reflecting current ability).
2. **IRT-Lite (Item Response Theory):** Utilized a logistic regression algorithm treating student IDs and question IDs as features to isolate a student's pure ability from a question's inherent difficulty.

### Phase 4: Question Quality Audit
Applied Point-Biserial Correlation (comparing question correctness against total student accuracy). This successfully flagged 2,421 questions with negative or low discrimination indices, meaning these questions were failing to differentiate between strong and weak students (likely due to typos or ambiguous wording).

### Phase 5: Adaptive Recommendation Engine
Built a multi-tiered rule engine that:
- Identifies the student's top 5 weakest subtopics.
- Filters out all questions flagged by the Quality Audit.
- Recommends new questions starting at an 'Easy' level to build confidence before progressing to 'Hard'.
- Implements **Spaced Repetition** by resurfacing previously failed questions after a 7-day memory decay gap.

### Phase 6: Dashboards (The Interface)
Developed two light-themed, professional EdTech dashboards:
- **Founder Dashboard:** Provides a bird's-eye view of cohort performance, engagement trends, and content quality issues.
- **Student Dashboard:** Provides a personalized radar chart of strengths and adaptive study recommendations.

## 6. Key Results & Business Impact
- **Data Validity:** The diagnostic models achieved a highly significant Pearson correlation of **r = 0.55 (p < 0.0001)** when validated against true latent skills.
- **Content Improvement:** Identified **23%** of the question bank that requires immediate academic review, directly improving the platform's content quality.
- **Learning Optimization:** Offline testing showed that targeting predicted weak areas improved subsequent test accuracies in those topics by **+24.8%**.
- **Ready for Production:** Both the backend data infrastructure and frontend dashboards are fully deployed to the cloud and actively querying a live database.

## 7. Future Scope
- **Collaborative Filtering:** Implement an ML model (like Matrix Factorization) to recommend questions based on similar students' learning paths.
- **Real-Time Streaming:** Move from batch ETL updates to a streaming architecture (e.g., Kafka) so the recommendation engine updates the exact second a student finishes a question.
- **A/B Testing:** Launch the recommendation engine to 50% of the user base to measure the actual uplift in retention and test scores compared to the control group.
