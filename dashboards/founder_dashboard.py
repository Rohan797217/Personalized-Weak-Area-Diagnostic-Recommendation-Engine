"""
AptiDude — Founder Dashboard (Streamlit)

Cohort-level analytics for AptiDude founders:
- Cohort weak-topic heatmap
- Question quality flags & discrimination scores
- Engagement trends (DAU, accuracy, streaks)
- Mock test performance trends
- Filterable by exam type, date range, engagement level

Run: streamlit run dashboards/founder_dashboard.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.helpers import DATA_PROCESSED_DIR

# =====================================================================
# PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="AptiDude — Founder Dashboard",
    page_icon="🎯",
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

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.3rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e7ff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        color: #4f46e5 !important;
        background-color: #e0e7ff;
    }
    
    /* Text colors for standard markdown */
    .stMarkdown, .stText {
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# DATA LOADING
# =====================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load all processed data files."""
    data = {}
    files = {
        "users": "users.csv",
        "questions": "questions.csv",
        "attempts": "attempts.csv",
        "mock_tests": "mock_tests.csv",
        "topic_summary": "student_topic_summary.csv",
        "section_summary": "student_section_summary.csv",
        "daily_engagement": "daily_engagement.csv",
        "cohort_weak": "cohort_weak_topics.csv",
        "question_audit": "question_quality_audit.csv",
        "irt_abilities": "irt_abilities.csv",
        "irt_difficulties": "irt_difficulties.csv",
        "validation": "validation_results.csv",
    }

    for key, filename in files.items():
        try:
            data[key] = pd.read_csv(DATA_PROCESSED_DIR / filename)
        except FileNotFoundError:
            data[key] = pd.DataFrame()

    # Parse dates
    if "timestamp" in data.get("attempts", pd.DataFrame()).columns:
        data["attempts"]["timestamp"] = pd.to_datetime(data["attempts"]["timestamp"])
    if "date" in data.get("daily_engagement", pd.DataFrame()).columns:
        data["daily_engagement"]["date"] = pd.to_datetime(data["daily_engagement"]["date"])
    if "date" in data.get("mock_tests", pd.DataFrame()).columns:
        data["mock_tests"]["date"] = pd.to_datetime(data["mock_tests"]["date"])

    return data


# =====================================================================
# SIDEBAR FILTERS
# =====================================================================

def render_sidebar(data):
    """Render sidebar with filters."""
    st.sidebar.markdown("## 🎯 AptiDude")
    st.sidebar.markdown("### Founder Analytics")
    st.sidebar.markdown("---")

    filters = {}

    # Exam filter
    if "users" in data and len(data["users"]) > 0:
        exams = ["All"] + sorted(data["users"]["target_exam"].unique().tolist())
        filters["exam"] = st.sidebar.selectbox("🎓 Target Exam", exams, index=0)

    # Engagement filter
    if "users" in data and len(data["users"]) > 0:
        engagements = ["All"] + sorted(data["users"]["engagement_level"].unique().tolist())
        filters["engagement"] = st.sidebar.selectbox("📊 Engagement Level", engagements, index=0)

    # Date range
    if "attempts" in data and len(data["attempts"]) > 0:
        min_date = data["attempts"]["timestamp"].min().date()
        max_date = data["attempts"]["timestamp"].max().date()
        filters["date_range"] = st.sidebar.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    # Section filter
    if "questions" in data and len(data["questions"]) > 0:
        sections = ["All"] + sorted(data["questions"]["section"].unique().tolist())
        filters["section"] = st.sidebar.selectbox("📚 Section", sections, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='color: #8892b0; font-size: 0.75rem; text-align: center;'>"
        "AptiDude DS Analytics v1.0<br>Powered by Streamlit</p>",
        unsafe_allow_html=True,
    )

    return filters


def apply_filters(data, filters):
    """Apply sidebar filters to the data."""
    filtered = {}
    for key, df in data.items():
        filtered[key] = df.copy() if isinstance(df, pd.DataFrame) else df

    # Filter users by exam and engagement
    if filters.get("exam") and filters["exam"] != "All" and len(filtered.get("users", pd.DataFrame())) > 0:
        user_ids = set(filtered["users"][filtered["users"]["target_exam"] == filters["exam"]]["user_id"])
        for key in ["attempts", "topic_summary", "section_summary", "mock_tests"]:
            if key in filtered and "user_id" in filtered[key].columns:
                filtered[key] = filtered[key][filtered[key]["user_id"].isin(user_ids)]

    if filters.get("engagement") and filters["engagement"] != "All" and len(filtered.get("users", pd.DataFrame())) > 0:
        user_ids = set(filtered["users"][filtered["users"]["engagement_level"] == filters["engagement"]]["user_id"])
        for key in ["attempts", "topic_summary", "section_summary", "mock_tests"]:
            if key in filtered and "user_id" in filtered[key].columns:
                filtered[key] = filtered[key][filtered[key]["user_id"].isin(user_ids)]

    # Filter by section
    if filters.get("section") and filters["section"] != "All":
        for key in ["topic_summary", "section_summary", "cohort_weak", "question_audit"]:
            if key in filtered and "section" in filtered[key].columns:
                filtered[key] = filtered[key][filtered[key]["section"] == filters["section"]]

    return filtered


# =====================================================================
# KPI METRICS
# =====================================================================

def render_kpi_metrics(data):
    """Render top-level KPI cards."""
    cols = st.columns(5)

    n_users = len(data.get("users", pd.DataFrame()))
    n_attempts = len(data.get("attempts", pd.DataFrame()))
    n_questions = len(data.get("questions", pd.DataFrame()))

    avg_accuracy = data["attempts"]["is_correct"].mean() * 100 if len(data.get("attempts", pd.DataFrame())) > 0 else 0

    flagged = 0
    if len(data.get("question_audit", pd.DataFrame())) > 0:
        flagged = len(data["question_audit"][
            data["question_audit"]["flag"].isin(["negative_discrimination", "low_discrimination"])
        ])

    metrics = [
        ("👥", f"{n_users:,}", "Total Students"),
        ("📝", f"{n_attempts:,}", "Total Attempts"),
        ("❓", f"{n_questions:,}", "Questions"),
        ("🎯", f"{avg_accuracy:.1f}%", "Avg Accuracy"),
        ("⚠️", f"{flagged}", "Flagged Questions"),
    ]

    for col, (icon, value, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)


# =====================================================================
# TAB 1: COHORT ANALYTICS
# =====================================================================

def render_cohort_analytics(data):
    """Render cohort-level weak topic analysis."""
    st.markdown('<div class="section-header">📊 Cohort Weak-Area Heatmap</div>', unsafe_allow_html=True)

    topic_summary = data.get("topic_summary", pd.DataFrame())
    if len(topic_summary) == 0:
        st.info("No topic summary data available. Run the ETL pipeline first.")
        return

    # Build heatmap: section × subtopic, colored by avg accuracy
    heatmap_data = topic_summary.groupby(["section", "subtopic"]).agg(
        avg_accuracy=("weighted_accuracy", "mean"),
        num_students=("user_id", "nunique"),
        total_attempts=("total_attempts", "sum"),
    ).reset_index()

    # Pivot for heatmap
    pivot = heatmap_data.pivot_table(
        index="subtopic", columns="section", values="avg_accuracy", aggfunc="mean"
    )

    if len(pivot) > 0:
        fig = px.imshow(
            pivot,
            color_continuous_scale=["#FF6B6B", "#FFE66D", "#4ECDC4"],
            aspect="auto",
            title="Average Accuracy by Subtopic × Section",
            labels={"color": "Accuracy"},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=max(400, len(pivot) * 22),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")

    # Top 15 hardest subtopics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔴 Hardest Subtopics (lowest accuracy)")
        hardest = heatmap_data.nsmallest(15, "avg_accuracy")
        fig = px.bar(
            hardest,
            x="avg_accuracy",
            y="subtopic",
            color="section",
            orientation="h",
            title="Bottom 15 Subtopics by Accuracy",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
            height=500,
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### 🟢 Easiest Subtopics (highest accuracy)")
        easiest = heatmap_data.nlargest(15, "avg_accuracy")
        fig = px.bar(
            easiest,
            x="avg_accuracy",
            y="subtopic",
            color="section",
            orientation="h",
            title="Top 15 Subtopics by Accuracy",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
            height=500,
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")


# =====================================================================
# TAB 2: QUESTION QUALITY
# =====================================================================

def render_question_quality(data):
    """Render question quality flags and discrimination analysis."""
    st.markdown('<div class="section-header">🔍 Question Quality Audit</div>', unsafe_allow_html=True)

    audit = data.get("question_audit", pd.DataFrame())
    if len(audit) == 0:
        st.info("No question quality data. Run the quality audit first.")
        return

    # Flag distribution
    col1, col2 = st.columns([1, 2])

    with col1:
        flag_counts = audit["flag"].value_counts().reset_index()
        flag_counts.columns = ["Flag", "Count"]

        colors = {
            "good_discrimination": "#4ECDC4",
            "moderate_discrimination": "#FFE66D",
            "low_discrimination": "#FF9F43",
            "negative_discrimination": "#FF6B6B",
            "insufficient_data": "#8892b0",
        }
        flag_counts["color"] = flag_counts["Flag"].map(colors)

        fig = px.pie(
            flag_counts,
            values="Count",
            names="Flag",
            color="Flag",
            color_discrete_map=colors,
            title="Question Quality Distribution",
            hole=0.45,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Discrimination index distribution
        valid_audit = audit.dropna(subset=["discrimination_index"])
        fig = px.histogram(
            valid_audit,
            x="discrimination_index",
            color="flag",
            nbins=50,
            title="Discrimination Index Distribution",
            color_discrete_map=colors,
            barmode="overlay",
            opacity=0.7,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")

    # Flagged questions table
    st.markdown("#### ⚠️ Flagged Questions (Need Review)")
    flagged = audit[audit["flag"].isin(["negative_discrimination", "low_discrimination"])].copy()
    if len(flagged) > 0:
        display_cols = ["question_id", "section", "subtopic", "difficulty_tag",
                        "discrimination_index", "correct_rate", "n_attempts", "flag"]
        available_cols = [c for c in display_cols if c in flagged.columns]
        st.dataframe(
            flagged[available_cols].head(50),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("No questions flagged — all questions have acceptable discrimination.")

    # Discrimination by section
    if "section" in valid_audit.columns:
        fig = px.box(
            valid_audit,
            x="section",
            y="discrimination_index",
            color="section",
            title="Discrimination Index by Section",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")


# =====================================================================
# TAB 3: ENGAGEMENT TRENDS
# =====================================================================

def render_engagement_trends(data):
    """Render engagement analytics."""
    st.markdown('<div class="section-header">📈 Engagement Trends</div>', unsafe_allow_html=True)

    daily = data.get("daily_engagement", pd.DataFrame())
    if len(daily) == 0:
        st.info("No engagement data available.")
        return

    # DAU + attempts trend
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Daily Active Users", "Daily Accuracy & Attempts"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["active_users"],
            mode="lines+markers",
            name="Active Users",
            line=dict(color="#667eea", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(102,126,234,0.1)",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=daily["date"], y=daily["total_attempts"],
            name="Total Attempts",
            marker_color="rgba(118,75,162,0.4)",
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["accuracy"],
            mode="lines",
            name="Accuracy",
            line=dict(color="#4ECDC4", width=2),
            yaxis="y4",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
        font=dict(family="Inter"),
        showlegend=True,
    )
    st.plotly_chart(fig, width="stretch")

    # Source breakdown
    attempts = data.get("attempts", pd.DataFrame())
    if len(attempts) > 0 and "source" in attempts.columns:
        col1, col2 = st.columns(2)

        with col1:
            source_counts = attempts["source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Count"]
            fig = px.pie(
                source_counts,
                values="Count",
                names="Source",
                title="Attempts by Source",
                hole=0.45,
                color_discrete_sequence=["#667eea", "#764ba2", "#4ECDC4", "#FFE66D"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            if "hour_of_day" in attempts.columns:
                hourly = attempts.groupby("hour_of_day").size().reset_index(name="count")
                fig = px.bar(
                    hourly,
                    x="hour_of_day",
                    y="count",
                    title="Attempts by Hour of Day",
                    color="count",
                    color_continuous_scale=["#1a1a2e", "#667eea", "#764ba2"],
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    xaxis_title="Hour of Day",
                    yaxis_title="Number of Attempts",
                )
                st.plotly_chart(fig, width="stretch")

    # Guessing analysis
    if len(attempts) > 0 and "is_guessing" in attempts.columns:
        st.markdown("#### 🎲 Guessing Patterns")
        col1, col2, col3 = st.columns(3)

        guessing_rate = attempts["is_guessing"].mean() * 100
        col1.metric("Overall Guessing Rate", f"{guessing_rate:.1f}%")

        if "section" in attempts.columns:
            section_guess = attempts.groupby("section")["is_guessing"].mean() * 100
            col2.metric("Worst Section (Guessing)", f"{section_guess.idxmax()}", f"{section_guess.max():.1f}%")
            col3.metric("Best Section (Guessing)", f"{section_guess.idxmin()}", f"{section_guess.min():.1f}%")


# =====================================================================
# TAB 4: MOCK TEST ANALYTICS
# =====================================================================

def render_mock_analytics(data):
    """Render mock test performance trends."""
    st.markdown('<div class="section-header">📋 Mock Test Analytics</div>', unsafe_allow_html=True)

    mocks = data.get("mock_tests", pd.DataFrame())
    if len(mocks) == 0:
        st.info("No mock test data available.")
        return

    # Overall score distribution
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            mocks,
            x="overall_score",
            nbins=30,
            title="Overall Score Distribution",
            color_discrete_sequence=["#667eea"],
            opacity=0.8,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.histogram(
            mocks,
            x="percentile_est",
            nbins=30,
            title="Percentile Distribution",
            color_discrete_sequence=["#4ECDC4"],
            opacity=0.8,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")

    # Section score distributions
    section_cols = ["varc_score", "dilr_score", "qa_score"]
    section_labels = ["VARC", "DILR", "QA"]
    available_sections = [c for c in section_cols if c in mocks.columns]

    if available_sections:
        melted = mocks.melt(
            value_vars=available_sections,
            var_name="Section",
            value_name="Score",
        )
        melted["Section"] = melted["Section"].map(dict(zip(section_cols, section_labels)))

        fig = px.box(
            melted,
            x="Section",
            y="Score",
            color="Section",
            title="Section-wise Score Distribution",
            color_discrete_sequence=["#667eea", "#764ba2", "#4ECDC4"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")

    # Score trend over time (cohort average)
    if "date" in mocks.columns:
        monthly = mocks.set_index("date").resample("W").agg(
            avg_score=("overall_score", "mean"),
            avg_percentile=("percentile_est", "mean"),
            n_tests=("mock_id", "count"),
        ).reset_index()

        if len(monthly) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["date"], y=monthly["avg_score"],
                mode="lines+markers",
                name="Avg Overall Score",
                line=dict(color="#667eea", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=monthly["date"], y=monthly["avg_percentile"],
                mode="lines+markers",
                name="Avg Percentile",
                line=dict(color="#4ECDC4", width=3),
                yaxis="y2",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Weekly Mock Performance Trend",
                yaxis=dict(title="Average Score"),
                yaxis2=dict(title="Average Percentile", overlaying="y", side="right"),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, width="stretch")


# =====================================================================
# TAB 5: MODEL VALIDATION
# =====================================================================

def render_validation(data):
    """Render model validation results."""
    st.markdown('<div class="section-header">✅ Model Validation Results</div>', unsafe_allow_html=True)

    validation = data.get("validation", pd.DataFrame())
    if len(validation) == 0:
        st.info("No validation results available. Run the diagnostic model first.")
        return

    st.dataframe(validation, width="stretch", hide_index=True)

    # IRT ability distribution
    col1, col2 = st.columns(2)

    abilities = data.get("irt_abilities", pd.DataFrame())
    if len(abilities) > 0:
        with col1:
            fig = px.histogram(
                abilities,
                x="irt_ability",
                nbins=40,
                title="IRT Student Ability Distribution",
                color_discrete_sequence=["#667eea"],
                opacity=0.8,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, width="stretch")

    difficulties = data.get("irt_difficulties", pd.DataFrame())
    if len(difficulties) > 0:
        with col2:
            fig = px.histogram(
                difficulties,
                x="irt_difficulty",
                nbins=40,
                title="IRT Question Difficulty Distribution",
                color_discrete_sequence=["#764ba2"],
                opacity=0.8,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, width="stretch")


# =====================================================================
# MAIN APP
# =====================================================================

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎯 AptiDude — Founder Dashboard</h1>
            <p>Cohort-level analytics, question quality audit, and engagement insights</p>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    data = load_data()

    if all(len(df) == 0 for df in data.values() if isinstance(df, pd.DataFrame)):
        st.error("⚠️ No processed data found. Please run the ETL pipeline first:\n\n```bash\npython -m src.data.etl_pipeline\n```")
        return

    # Sidebar filters
    filters = render_sidebar(data)
    data = apply_filters(data, filters)

    # KPI metrics
    render_kpi_metrics(data)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Cohort Analytics",
        "🔍 Question Quality",
        "📈 Engagement Trends",
        "📋 Mock Tests",
        "✅ Validation",
    ])

    with tab1:
        render_cohort_analytics(data)
    with tab2:
        render_question_quality(data)
    with tab3:
        render_engagement_trends(data)
    with tab4:
        render_mock_analytics(data)
    with tab5:
        render_validation(data)


if __name__ == "__main__":
    main()
