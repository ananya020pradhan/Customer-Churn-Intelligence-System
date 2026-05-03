import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, confusion_matrix

from src.preprocessing import add_features
from src.predict import predict_customer


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
    display: none !important;
}

.block-container {
    padding-top: 0.8rem !important;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(34,211,238,0.35), transparent 28%),
        radial-gradient(circle at top right, rgba(236,72,153,0.30), transparent 28%),
        linear-gradient(135deg, #020617 0%, #0f172a 35%, #1e1b4b 70%, #581c87 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #111827, #312e81);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3, h4, h5, h6, p, label, span {
    color: white !important;
}

.hero {
    background: linear-gradient(135deg, #06b6d4, #6366f1, #ec4899);
    padding: 34px;
    border-radius: 28px;
    margin-bottom: 24px;
    box-shadow: 0 22px 55px rgba(0,0,0,0.42);
}

.hero-title {
    font-size: 44px;
    font-weight: 950;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: white;
    margin-top: 10px;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.22);
    padding: 8px 15px;
    border-radius: 999px;
    margin-right: 8px;
    margin-top: 16px;
    font-weight: 800;
}

.text-card {
    background: rgba(255,255,255,0.12);
    padding: 24px;
    border-radius: 24px;
    margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.20);
    box-shadow: 0 14px 32px rgba(0,0,0,0.28);
}

.kpi {
    padding: 24px;
    border-radius: 24px;
    text-align: center;
    min-height: 130px;
    box-shadow: 0 14px 32px rgba(0,0,0,0.28);
}

.kpi h3 {
    font-size: 15px;
    margin-bottom: 10px;
}

.kpi p {
    font-size: 30px;
    font-weight: 950;
    margin: 0;
}

.cyan {background: linear-gradient(135deg, #0891b2, #22d3ee);}
.indigo {background: linear-gradient(135deg, #4f46e5, #8b5cf6);}
.pink {background: linear-gradient(135deg, #db2777, #f43f5e);}
.amber {background: linear-gradient(135deg, #f97316, #facc15);}
.green {background: linear-gradient(135deg, #16a34a, #22c55e);}
.red {background: linear-gradient(135deg, #dc2626, #fb7185);}

input {
    background: white !important;
    color: #111827 !important;
    font-weight: 800 !important;
}

div[data-baseweb="select"] > div {
    background: white !important;
}

div[data-baseweb="select"] * {
    color: #111827 !important;
    font-weight: 800 !important;
}

ul[role="listbox"], ul[role="listbox"] li, ul[role="listbox"] li * {
    background: white !important;
    color: #111827 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #06b6d4, #6366f1, #ec4899);
    color: white !important;
    font-size: 18px;
    font-weight: 950;
    border-radius: 16px;
    border: none;
    width: 100%;
    padding: 0.9rem 1.4rem;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(6,182,212,0.62), rgba(99,102,241,0.70));
    padding: 18px;
    border-radius: 20px;
}

div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {
    color: white !important;
    font-weight: 900 !important;
}

[data-testid="stDataFrame"] * {
    color: #111827 !important;
}

canvas {
    background: white !important;
    border-radius: 14px;
}

div[data-testid="stAlert"] * {
    color: #111827 !important;
    font-weight: 750 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    if os.path.exists("data/churn_data.csv"):
        return pd.read_csv("data/churn_data.csv")
    return None


@st.cache_resource
def load_model():
    if os.path.exists("models/churn_model.pkl"):
        return joblib.load("models/churn_model.pkl")
    return None


def risk_label(probability):
    if probability >= 0.75:
        return "Critical"
    if probability >= 0.55:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def retention_action(probability):
    if probability >= 0.75:
        return "Urgent retention call + personalized discount"
    if probability >= 0.55:
        return "Priority support callback + loyalty offer"
    if probability >= 0.30:
        return "Engagement campaign + product education"
    return "Maintain normal engagement"


def prepare_scored_data(dataframe, trained_model):
    feature_df = add_features(dataframe.copy())
    X = feature_df.drop(columns=["customer_id", "churn"], errors="ignore")
    probabilities = trained_model.predict_proba(X)[:, 1]

    scored = dataframe.copy()
    scored["churn_probability"] = probabilities
    scored["risk_segment"] = scored["churn_probability"].apply(risk_label)
    scored["recommended_action"] = scored["churn_probability"].apply(retention_action)

    return scored.sort_values("churn_probability", ascending=False)


def show_countplot(data, x, title, palette, hue=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=data, x=x, hue=hue, palette=palette, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    plt.xticks(rotation=15)
    st.pyplot(fig)


def show_barplot(data, x, y, title, palette):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=data, x=x, y=y, palette=palette, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=15)
    st.pyplot(fig)


df = load_data()
model = load_model()

if df is None:
    st.error("Dataset not found. First run: python main.py")
    st.stop()

if model is None:
    st.error("Model not found. First run: python main.py")
    st.stop()

scored_df = prepare_scored_data(df, model)


with st.sidebar:
    st.markdown("""
    <div class="hero" style="padding:20px; border-radius:22px;">
        <div style="font-size:24px; font-weight:950;">📉 ChurnIQ Pro</div>
        <div style="font-size:13px;">Customer Retention Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Dashboard Sections",
        [
            "🔍 Synthetic Churn Dashboard",
            "📊 IBM Telco Dataset Analysis",
            "🔮 Single Customer Prediction",
            "🚦 Risk Segmentation",
            "📈 Model Performance Charts",
            "🏆 Top 50 Churn Watchlist",
            "🧠 Business Insights",
        ],
    )

    st.markdown("""
    <div class="text-card">
        <b>🎯 Focus</b><br>
        Churn behavior<br>
        Risk scoring<br>
        Retention strategy
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="hero-title">📉 Customer Churn Intelligence Dashboard</div>
    <div class="hero-subtitle">
        A professional machine learning dashboard for churn behavior analysis, churn probability prediction,
        risk segmentation, customer retention strategy, model performance tracking, and high-risk customer watchlist.
    </div>
    <span class="badge">Customer Churn Behavior</span>
    <span class="badge">Risk Segmentation</span>
    <span class="badge">Top 50 Watchlist</span>
    <span class="badge">Model Performance</span>
    <span class="badge">Single Prediction</span>
</div>
""", unsafe_allow_html=True)


if page == "🔍 Synthetic Churn Dashboard":
    total_customers = len(scored_df)
    churn_rate = df["churn"].mean() * 100
    avg_probability = scored_df["churn_probability"].mean() * 100
    critical_customers = (scored_df["risk_segment"] == "Critical").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi cyan"><h3>Total Customers</h3><p>{total_customers}</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi pink"><h3>Actual Churn Rate</h3><p>{churn_rate:.1f}%</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi indigo"><h3>Avg Churn Probability</h3><p>{avg_probability:.1f}%</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi red"><h3>Critical Risk</h3><p>{critical_customers}</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="text-card">
        <h2>🔍 This Project Focuses On</h2>
        <ul>
            <li>Customer churn behavior</li>
            <li>Churn probability prediction</li>
            <li>Low, Medium, High, and Critical risk customers</li>
            <li>Customer retention strategy</li>
            <li>Contract, tenure, usage, billing, and service-related churn factors</li>
            <li>Top 50 high-risk customer watchlist</li>
            <li>Model performance using Accuracy, ROC-AUC, PR-AUC, and Recall</li>
            <li>Single customer churn prediction</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual Churn Distribution")
        show_countplot(df, "churn", "Actual Churn Distribution", ["#22c55e", "#ef4444"])

    with col2:
        st.subheader("Risk Segment Distribution")
        risk_counts = scored_df["risk_segment"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).reset_index()
        risk_counts.columns = ["Risk Segment", "Count"]
        show_barplot(risk_counts, "Risk Segment", "Count", "Risk Segment Distribution", ["#22c55e", "#facc15", "#f97316", "#ef4444"])


elif page == "📊 IBM Telco Dataset Analysis":
    st.markdown("""
    <div class="text-card">
        <h2>📊 IBM Telco Dataset Analysis Style</h2>
        <p>This section follows telecom churn analysis style using contract, service, autopay, and region-based churn factors.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Contract Type vs Churn")
        show_countplot(df, "contract_type", "Contract Type vs Churn", ["#06b6d4", "#ec4899"], hue="churn")

    with col2:
        st.subheader("Internet Service vs Churn")
        show_countplot(df, "internet_service", "Internet Service vs Churn", ["#8b5cf6", "#f97316"], hue="churn")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Autopay vs Churn")
        show_countplot(df, "is_autopay", "Autopay vs Churn", ["#22c55e", "#ef4444"], hue="churn")

    with col4:
        st.subheader("Region vs Churn")
        show_countplot(df, "region", "Region vs Churn", ["#3b82f6", "#f43f5e"], hue="churn")


elif page == "🔮 Single Customer Prediction":
    st.subheader("🔮 Single Customer Churn Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Customer Age", 18, 80, 30)
        tenure_months = st.number_input("Tenure Months", 1, 100, 12)
        monthly_charges = st.number_input("Monthly Charges", 100, 10000, 1200)
        total_charges = st.number_input("Total Charges", 100, 500000, 15000)

    with col2:
        usage_hours = st.number_input("Usage Hours", 1, 200, 40)
        login_count = st.number_input("Login Count", 1, 200, 25)
        support_tickets = st.number_input("Support Tickets", 0, 20, 2)
        payment_delay_days = st.number_input("Payment Delay Days", 0, 60, 5)

    with col3:
        contract_type = st.selectbox("Contract Type", ["Monthly", "Quarterly", "Yearly"])
        internet_service = st.selectbox("Internet Service", ["Fiber", "DSL", "No"])
        is_autopay = st.selectbox("Autopay Enabled", ["Yes", "No"])
        region = st.selectbox("Region", ["East", "West", "North", "South"])

    if st.button("🚀 Predict Churn Probability"):
        result = predict_customer({
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "usage_hours": usage_hours,
            "login_count": login_count,
            "support_tickets": support_tickets,
            "payment_delay_days": payment_delay_days,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "is_autopay": is_autopay,
            "region": region,
        })

        probability = result["churn_probability"]
        risk = risk_label(probability)

        st.success(f"Prediction: {result['prediction']}")
        st.metric("Churn Probability", f"{probability * 100:.2f}%")
        st.metric("Risk Segment", risk)
        st.info(f"Recommended Action: {retention_action(probability)}")


elif page == "🚦 Risk Segmentation":
    st.subheader("🚦 Customer Risk Segmentation")

    segment_summary = scored_df.groupby("risk_segment").agg(
        customers=("customer_id", "count"),
        avg_probability=("churn_probability", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_support_tickets=("support_tickets", "mean"),
    ).reindex(["Low", "Medium", "High", "Critical"]).reset_index()

    segment_summary["avg_probability"] = (segment_summary["avg_probability"] * 100).round(2)
    segment_summary["avg_monthly_charges"] = segment_summary["avg_monthly_charges"].round(2)
    segment_summary["avg_support_tickets"] = segment_summary["avg_support_tickets"].round(2)

    st.dataframe(segment_summary, use_container_width=True)
    show_barplot(segment_summary, "risk_segment", "customers", "Customers by Risk Segment", ["#22c55e", "#facc15", "#f97316", "#ef4444"])


elif page == "📈 Model Performance Charts":
    import json

    st.markdown("""
    <div class="text-card">
        <h2>📈 Model Performance Dashboard</h2>
        <p>This section shows recruiter-friendly evaluation outputs including Accuracy, ROC-AUC, PR-AUC, Recall, Confusion Matrix, ROC Curve, Precision-Recall Curve, and Feature Importance.</p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists("outputs/metrics.json"):
        with open("outputs/metrics.json", "r") as f:
            metrics = json.load(f)

        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(
            f'<div class="kpi cyan"><h3>Accuracy</h3><p>{metrics["accuracy"]:.2f}</p></div>',
            unsafe_allow_html=True
        )
        c2.markdown(
            f'<div class="kpi indigo"><h3>ROC-AUC</h3><p>{metrics["roc_auc"]:.2f}</p></div>',
            unsafe_allow_html=True
        )
        c3.markdown(
            f'<div class="kpi pink"><h3>PR-AUC</h3><p>{metrics["pr_auc"]:.2f}</p></div>',
            unsafe_allow_html=True
        )
        c4.markdown(
            f'<div class="kpi amber"><h3>Recall</h3><p>{metrics["recall"]:.2f}</p></div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            if os.path.exists("outputs/confusion_matrix.png"):
                st.image("outputs/confusion_matrix.png", use_container_width=True)

        with col2:
            st.subheader("ROC Curve")
            if os.path.exists("outputs/roc_curve.png"):
                st.image("outputs/roc_curve.png", use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Precision-Recall Curve")
            if os.path.exists("outputs/pr_curve.png"):
                st.image("outputs/pr_curve.png", use_container_width=True)

        with col4:
            st.subheader("Feature Importance")
            if os.path.exists("outputs/feature_importance.png"):
                st.image("outputs/feature_importance.png", use_container_width=True)

        if os.path.exists("outputs/feature_importance.csv"):
            st.subheader("Top Important Features")
            importance_df = pd.read_csv("outputs/feature_importance.csv")
            st.dataframe(importance_df, use_container_width=True)

    else:
        st.warning("Model performance files not found. First run: python main.py")


elif page == "🏆 Top 50 Churn Watchlist":
    st.subheader("🏆 Top 50 High-Risk Customer Watchlist")

    watchlist = scored_df[
        [
            "customer_id",
            "churn_probability",
            "risk_segment",
            "contract_type",
            "tenure_months",
            "usage_hours",
            "monthly_charges",
            "support_tickets",
            "payment_delay_days",
            "is_autopay",
            "recommended_action",
        ]
    ].head(50).copy()

    watchlist["churn_probability"] = (watchlist["churn_probability"] * 100).round(2).astype(str) + "%"

    st.dataframe(watchlist, use_container_width=True)


elif page == "🧠 Business Insights":
    st.markdown("""
    <div class="text-card">
        <h2>🧠 Business Insights</h2>
        <p><b>Customer churn behavior:</b> Customers with low usage, delayed payments, and more support tickets are more likely to churn.</p>
        <p><b>Contract-related churn factor:</b> Monthly contract customers usually have higher churn risk.</p>
        <p><b>Billing-related factor:</b> High monthly charges with low usage may indicate poor perceived value.</p>
        <p><b>Service factor:</b> High support tickets suggest dissatisfaction or unresolved issues.</p>
        <p><b>Retention strategy:</b> Focus retention actions on High and Critical risk customers instead of giving discounts to everyone.</p>
    </div>
    """, unsafe_allow_html=True)

    playbook = pd.DataFrame({
        "Risk Segment": ["Low", "Medium", "High", "Critical"],
        "Customer Meaning": [
            "Healthy customers with low churn probability",
            "Customers showing early churn signals",
            "Customers likely to leave soon",
            "Urgent customers with very high churn probability",
        ],
        "Recommended Action": [
            "Maintain normal engagement",
            "Send engagement campaign",
            "Offer loyalty benefit",
            "Immediate retention call and discount",
        ],
        "Business Goal": [
            "Maintain loyalty",
            "Prevent risk growth",
            "Reduce near-term churn",
            "Protect revenue immediately",
        ],
    })

    st.dataframe(playbook, use_container_width=True)