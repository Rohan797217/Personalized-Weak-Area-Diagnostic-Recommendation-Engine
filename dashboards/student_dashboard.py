"""
AptiDude — Student Dashboard (Streamlit)

Individual student diagnostic report:
- Personal diagnostic radar chart (section-level strengths)
- Subtopic-level accuracy breakdown with weak areas highlighted
- Mock test score progression over time
- Personalized recommendations (next-best questions)
- Spaced repetition reminders

Run: streamlit run dashboards/student_dashboard.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils.helpers import DATA_PROCESSED_DIR, DATASET_DIR
from src.models.diagnostic_model import WeightedAccuracyDiagnostic
from src.models.recommendation_engine import RecommendationEngine

# =====================================================================
# PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="AptiDude — Student Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CUSTOM CSS
# =====================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(79, 70, 229, 0.25);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
        color: #e0e7ff;
    }

    /* Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .rec-card {
        background: linear-gradient(145deg, #1e2640 0%, #1a2035 100%);
        border: 1px solid rgba(102, 126, 234, 0.15);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.7rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .rec-card:hover {
        transform: translateX(4px);
        border-color: rgba(102, 126, 234, 0.4);
    }
    .rec-title {
        font-weight: 600;
        color: #ccd6f6;
        font-size: 0.95rem;
    }
    .rec-reason {
        font-size: 0.8rem;
        color: #8892b0;
        margin-top: 0.2rem;
    }
    .rec-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-easy { background: rgba(78,205,196,0.15); color: #4ECDC4; }
    .badge-medium { background: rgba(255,230,77,0.15); color: #FFE66D; }
    .badge-hard { background: rgba(255,107,107,0.15); color: #FF6B6B; }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ccd6f6;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(168, 237, 234, 0.2);
    }

    .weak-area {
        background: #fef2f2;
        border-left: 3px solid #ef4444;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        color: #7f1d1d;
        font-size: 0.9rem;
    }
    .strong-area {
        background: #f0fdf4;
        border-left: 3px solid #22c55e;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        color: #14532d;
        font-size: 0.9rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# DATA LOADING
# =====================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load processed data."""
    data = {}
    files = {
        "users": "users.csv",
        "questions": "questions.csv",
        "attempts": "attempts.csv",
        "mock_tests": "mock_tests.csv",
        "topic_summary": "student_topic_summary.csv",
        "section_summary": "student_section_summary.csv",
        "irt_abilities": "irt_abilities.csv",
        "recommendations": "all_recommendations.csv",
    }

    for key, filename in files.items():
        try:
            data[key] = pd.read_csv(DATA_PROCESSED_DIR / filename)
        except FileNotFoundError:
            data[key] = pd.DataFrame()

    if "timestamp" in data.get("attempts", pd.DataFrame()).columns:
        data["attempts"]["timestamp"] = pd.to_datetime(data["attempts"]["timestamp"])
    if "date" in data.get("mock_tests", pd.DataFrame()).columns:
        data["mock_tests"]["date"] = pd.to_datetime(data["mock_tests"]["date"])

    return data


@st.cache_resource
def build_diagnostic(data):
    """Build the diagnostic model from loaded data."""
    ts = data.get("topic_summary", pd.DataFrame())
    ss = data.get("section_summary", pd.DataFrame())
    if len(ts) == 0 or len(ss) == 0:
        return None
    diag = WeightedAccuracyDiagnostic(weak_threshold=0.5, min_attempts=3)
    diag.fit(ts, ss)
    return diag


@st.cache_resource
def build_recommender(data):
    """Build the recommendation engine."""
    attempts = data.get("attempts", pd.DataFrame())
    questions = data.get("questions", pd.DataFrame())
    topic_summary = data.get("topic_summary", pd.DataFrame())

    if len(attempts) == 0 or len(questions) == 0 or len(topic_summary) == 0:
        return None

    attempts_copy = attempts.copy()
    attempts_copy["timestamp"] = pd.to_datetime(attempts_copy["timestamp"])

    try:
        qa = pd.read_csv(DATA_PROCESSED_DIR / "question_quality_audit.csv")
    except FileNotFoundError:
        qa = None

    engine = RecommendationEngine(n_weak_subtopics=5, spaced_rep_gap_days=7, max_recommendations=20)
    engine.fit(attempts_copy, questions, topic_summary, qa)
    return engine


# =====================================================================
# RADAR CHART
# =====================================================================

def render_radar_chart(profile: dict, user_id: str):
    """Render the section-level diagnostic radar chart."""
    scores = profile.get("section_scores", {})
    if not scores:
        st.warning("No section scores available.")
        return

    categories = list(scores.keys())
    values = [scores[c] for c in categories]
    # Close the radar polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(102,126,234,0.15)",
        line=dict(color="#667eea", width=3),
        name="Your Accuracy",
        marker=dict(size=8, color="#667eea"),
    ))

    # Add a "target" benchmark at 0.7
    target = [0.7] * len(categories_closed)
    fig.add_trace(go.Scatterpolar(
        r=target,
        theta=categories_closed,
        line=dict(color="rgba(255,107,107,0.4)", width=2, dash="dash"),
        name="Target (70%)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(size=10, color="#8892b0"),
                gridcolor="rgba(255,255,255,0.05)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#ccd6f6", family="Inter"),
                gridcolor="rgba(255,255,255,0.05)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=f"Section-Level Diagnostic — {user_id}",
            font=dict(size=16, family="Inter"),
        ),
        font=dict(family="Inter"),
        showlegend=True,
        height=450,
    )

    st.plotly_chart(fig, width="stretch")


# =====================================================================
# SUBTOPIC BREAKDOWN
# =====================================================================

def render_subtopic_breakdown(data, user_id):
    """Render detailed subtopic-level accuracy breakdown."""
    ts = data.get("topic_summary", pd.DataFrame())
    user_ts = ts[ts["user_id"] == user_id].copy()

    if len(user_ts) == 0:
        st.info("No subtopic data for this student.")
        return

    user_ts = user_ts.sort_values("weighted_accuracy")

    # Color by performance
    user_ts["performance"] = pd.cut(
        user_ts["weighted_accuracy"],
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=["Weak 🔴", "Developing 🟡", "Good 🟢", "Strong 💪"],
    )

    color_map = {
        "Weak 🔴": "#FF6B6B",
        "Developing 🟡": "#FFE66D",
        "Good 🟢": "#4ECDC4",
        "Strong 💪": "#667eea",
    }

    fig = px.bar(
        user_ts,
        x="weighted_accuracy",
        y="subtopic",
        color="performance",
        color_discrete_map=color_map,
        orientation="h",
        title="Subtopic-Level Accuracy",
        hover_data=["section", "total_attempts", "guessing_rate"],
    )

    fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(255,107,107,0.5)",
                  annotation_text="Weak threshold", annotation_position="top right")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(400, len(user_ts) * 28),
        font=dict(family="Inter"),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Weighted Accuracy", range=[0, 1]),
    )

    st.plotly_chart(fig, width="stretch")


# =====================================================================
# MOCK TEST PROGRESSION
# =====================================================================

def render_mock_progression(data, user_id):
    """Render mock test score progression chart."""
    mocks = data.get("mock_tests", pd.DataFrame())
    user_mocks = mocks[mocks["user_id"] == user_id].sort_values("date")

    if len(user_mocks) == 0:
        st.info("No mock test data for this student.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=user_mocks["date"], y=user_mocks["overall_score"],
        mode="lines+markers",
        name="Overall Score",
        line=dict(color="#667eea", width=3),
        marker=dict(size=10, symbol="circle"),
    ))

    section_colors = {"varc_score": "#4ECDC4", "dilr_score": "#FFE66D", "qa_score": "#FF6B6B"}
    section_names = {"varc_score": "VARC", "dilr_score": "DILR", "qa_score": "QA"}

    for col, color in section_colors.items():
        if col in user_mocks.columns:
            fig.add_trace(go.Scatter(
                x=user_mocks["date"], y=user_mocks[col],
                mode="lines+markers",
                name=section_names[col],
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=6),
                opacity=0.7,
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Mock Test Score Progression",
        xaxis_title="Date",
        yaxis_title="Score",
        font=dict(family="Inter"),
        height=400,
    )

    st.plotly_chart(fig, width="stretch")


# =====================================================================
# RECOMMENDATIONS
# =====================================================================

def render_recommendations(recommender, user_id):
    """Render personalized question recommendations."""
    if recommender is None:
        st.info("Recommendation engine not available.")
        return

    recs = recommender.recommend(user_id)
    if not recs:
        st.info("No recommendations available for this student.")
        return

    for i, rec in enumerate(recs[:10], 1):
        diff = rec["difficulty_tag"]
        badge_class = f"badge-{diff.lower()}" if diff.lower() in ["easy", "medium", "hard"] else "badge-medium"

        st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">
                    <span class="rec-badge {badge_class}">{diff}</span>
                    {rec['subtopic']}
                    <span style="float:right; color:#8892b0; font-size:0.8rem;">#{rec['question_id']}</span>
                </div>
                <div class="rec-reason">💡 {rec['reason']}</div>
            </div>
        """, unsafe_allow_html=True)


# =====================================================================
# ACTIVITY TIMELINE
# =====================================================================

def render_activity_timeline(data, user_id):
    """Render the student's practice activity over time."""
    attempts = data.get("attempts", pd.DataFrame())
    user_attempts = attempts[attempts["user_id"] == user_id].copy()

    if len(user_attempts) == 0:
        return

    user_attempts["date"] = user_attempts["timestamp"].dt.date
    daily = user_attempts.groupby("date").agg(
        attempts=("attempt_id", "count"),
        accuracy=("is_correct", "mean"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["attempts"],
        name="Attempts",
        marker_color="rgba(102,126,234,0.5)",
    ))

    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["accuracy"] * 100,
        mode="lines+markers",
        name="Accuracy %",
        line=dict(color="#4ECDC4", width=2),
        marker=dict(size=4),
        yaxis="y2",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Practice Activity Timeline",
        yaxis=dict(title="Attempts"),
        yaxis2=dict(title="Accuracy %", overlaying="y", side="right", range=[0, 100]),
        font=dict(family="Inter"),
        height=350,
    )

    st.plotly_chart(fig, width="stretch")


# =====================================================================
# MAIN APP
# =====================================================================

def main():
    # Header
    st.markdown("""
        <div class="student-header">
            <h1>🧠 AptiDude — Student Diagnostic Report</h1>
            <p>Your personalized weak-area analysis and study recommendations</p>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    data = load_data()

    if all(len(df) == 0 for df in data.values() if isinstance(df, pd.DataFrame)):
        st.error("⚠️ No processed data found. Please run the ETL pipeline first:\n\n```bash\npython -m src.data.etl_pipeline\npython -m src.models.diagnostic_model\npython -m src.models.recommendation_engine\n```")
        return

    # Sidebar — Student selector
    st.sidebar.markdown("## 🧠 AptiDude")
    st.sidebar.markdown("### Student Diagnostic")
    st.sidebar.markdown("---")

    users = data.get("users", pd.DataFrame())
    attempts = data.get("attempts", pd.DataFrame())

    if len(users) > 0:
        user_ids = sorted(users["user_id"].unique().tolist())
    elif len(attempts) > 0:
        user_ids = sorted(attempts["user_id"].unique().tolist())
    else:
        st.error("No user data found.")
        return

    selected_user = st.sidebar.selectbox("👤 Select Student", user_ids, index=0)

    # User info card
    if len(users) > 0:
        user_info = users[users["user_id"] == selected_user]
        if len(user_info) > 0:
            info = user_info.iloc[0]
            st.sidebar.markdown(f"""
                **Exam Target:** {info.get('target_exam', 'N/A')}
                **Engagement:** {info.get('engagement_level', 'N/A')}
                **Signup:** {info.get('signup_date', 'N/A')}
            """)

    st.sidebar.markdown("---")

    # IRT ability
    irt = data.get("irt_abilities", pd.DataFrame())
    if len(irt) > 0:
        user_irt = irt[irt["user_id"] == selected_user]
        if len(user_irt) > 0:
            ability = user_irt.iloc[0]["irt_ability"]
            percentile = (irt["irt_ability"] < ability).mean() * 100
            st.sidebar.metric("IRT Ability Percentile", f"{percentile:.0f}th")

    st.sidebar.markdown("""
    <p style='color: #8892b0; font-size: 0.75rem; text-align: center; margin-top: 2rem;'>
    AptiDude Student Analytics v1.0
    </p>
    """, unsafe_allow_html=True)

    # ============ MAIN CONTENT ============

    # Build diagnostic
    diagnostic = build_diagnostic(data)
    recommender = build_recommender(data)

    if diagnostic is None:
        st.error("Diagnostic model could not be built. Ensure processed data is available.")
        return

    profile = diagnostic.diagnose(selected_user)

    if "error" in profile:
        st.warning(f"No diagnostic data for {selected_user}. Student may have too few attempts.")
        return

    # --- KPI row ---
    user_attempts = attempts[attempts["user_id"] == selected_user] if len(attempts) > 0 else pd.DataFrame()

    cols = st.columns(5)
    total_att = len(user_attempts)
    accuracy = user_attempts["is_correct"].mean() * 100 if total_att > 0 else 0
    num_weak = profile["num_weak"]
    num_strong = profile["num_strong"]
    overall = profile["overall_accuracy"] * 100

    card_data = [
        ("📝", str(total_att), "Total Attempts", ""),
        ("🎯", f"{accuracy:.1f}%", "Raw Accuracy", "good" if accuracy >= 60 else "warn" if accuracy >= 40 else "bad"),
        ("📊", f"{overall:.1f}%", "Weighted Accuracy", "good" if overall >= 60 else "warn" if overall >= 40 else "bad"),
        ("🔴", str(num_weak), "Weak Subtopics", "bad" if num_weak > 5 else "warn" if num_weak > 2 else "good"),
        ("💪", str(num_strong), "Strong Subtopics", "good"),
    ]

    for col, (icon, val, label, cls) in zip(cols, card_data):
        with col:
            st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 1.3rem;">{icon}</div>
                    <div class="stat-value {cls}">{val}</div>
                    <div class="stat-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Two-column layout: Radar + Weak/Strong areas ---
    col_left, col_right = st.columns([3, 2])

    with col_left:
        render_radar_chart(profile, selected_user)

    with col_right:
        st.markdown('<div class="section-title">🔴 Your Weak Areas</div>', unsafe_allow_html=True)
        weak_subs = profile.get("weak_subtopics", [])[:7]
        if weak_subs:
            for w in weak_subs:
                st.markdown(
                    f'<div class="weak-area">'
                    f'<strong>{w["subtopic"]}</strong> '
                    f'<span style="float:right;">{w["weighted_accuracy"]:.0%}</span>'
                    f'<br><span style="color:#8892b0; font-size:0.8rem;">{w["section"]} · {w["total_attempts"]} attempts</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No weak areas detected! Keep it up!")

        st.markdown('<div class="section-title">💪 Your Strengths</div>', unsafe_allow_html=True)
        strong_subs = profile.get("strong_subtopics", [])[:5]
        if strong_subs:
            for s in strong_subs:
                st.markdown(
                    f'<div class="strong-area">'
                    f'<strong>{s["subtopic"]}</strong> '
                    f'<span style="float:right;">{s["weighted_accuracy"]:.0%}</span>'
                    f'<br><span style="color:#8892b0; font-size:0.8rem;">{s["section"]} · {s["total_attempts"]} attempts</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # --- Subtopic breakdown ---
    st.markdown('<div class="section-title">📊 Subtopic-Level Breakdown</div>', unsafe_allow_html=True)
    render_subtopic_breakdown(data, selected_user)

    # --- Two columns: Mock progression + Recommendations ---
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-title">📋 Mock Test Progression</div>', unsafe_allow_html=True)
        render_mock_progression(data, selected_user)

        st.markdown('<div class="section-title">📈 Activity Timeline</div>', unsafe_allow_html=True)
        render_activity_timeline(data, selected_user)

    with col_right:
        st.markdown('<div class="section-title">🎯 Recommended Next Questions</div>', unsafe_allow_html=True)
        render_recommendations(recommender, selected_user)


if __name__ == "__main__":
    main()
