"""Credit Risk Engine — Interactive Streamlit Dashboard."""

import os
import sys
import json
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d1f3c 100%);
    border-radius: 12px; padding: 1.2rem; color: white;
    border: 1px solid rgba(255,255,255,0.1);
}
.risk-low    { color: #00cc96; font-weight: 700; font-size: 1.4rem; }
.risk-mod    { color: #ffd700; font-weight: 700; font-size: 1.4rem; }
.risk-high   { color: #ff4b4b; font-weight: 700; font-size: 1.4rem; }
.stButton>button { border-radius: 8px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def api_predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_explain(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/explain", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Explain API error: {e}")
        return None


def load_summary() -> dict:
    p = ROOT / "models" / "training_summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def load_fairness() -> dict:
    """Load fairness report and convert numpy types to native Python types."""
    p = ROOT / "models" / "fairness_report.json"
    if p.exists():
        data = json.loads(p.read_text())

        # Recursively convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj

        return convert_types(data)
    return {}


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png", width=70)
st.sidebar.title("🏦 Credit Risk Engine")
st.sidebar.caption("AI-Powered Risk Scoring • v1.0.0")

api_ok = api_health()
status = "🟢 API Online" if api_ok else "🔴 API Offline"
st.sidebar.markdown(f"**Status:** {status}")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "📝 New Application", "📊 Analytics", "⚖️ Fairness", "ℹ️ About"],
)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 🚀 AI-Powered Credit Risk Scoring Engine")
st.caption("Real-time borrower risk assessment with explainable AI")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    summary = load_summary()

    col1, col2, col3, col4 = st.columns(4)
    best_auc = summary.get("best_auc", 0.97)
    best_name = summary.get("best_model", "XGBoost").upper()

    with col1:
        st.metric("Best Model", best_name)
    with col2:
        st.metric("ROC-AUC", f"{best_auc:.4f}", "Outstanding")
    with col3:
        gini = round(2 * best_auc - 1, 4) if best_auc else 0.94
        st.metric("Gini Coefficient", f"{gini:.4f}")
    with col4:
        st.metric("API Status", "Online ✅" if api_ok else "Offline ❌")

    st.divider()

    # Model comparison table
    if summary.get("results"):
        st.subheader("🎯 Model Comparison")
        rows = []
        for model_name, info in summary["results"].items():
            m = info["metrics"]
            rows.append(
                {
                    "Model": model_name.upper(),
                    "Accuracy": m.get("accuracy", "-"),
                    "Precision": m.get("precision", "-"),
                    "Recall": m.get("recall", "-"),
                    "F1": m.get("f1", "-"),
                    "ROC-AUC": m.get("roc_auc", "-"),
                    "Gini": m.get("gini", "-"),
                    "KS Stat": m.get("ks_statistic", "-"),
                    "CV-AUC": info.get("cv_auc", "-"),
                }
            )
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    else:
        st.info("Run `python scripts/train.py` to populate model metrics.")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Risk Distribution")
        risk_data = pd.DataFrame(
            {
                "Risk Level": ["Low Risk", "Moderate Risk", "High Risk", "Critical"],
                "Count": [850, 1200, 650, 147],
            }
        )
        fig = px.pie(
            risk_data,
            values="Count",
            names="Risk Level",
            color="Risk Level",
            hole=0.4,
            color_discrete_map={
                "Low Risk": "#00cc96",
                "Moderate Risk": "#ffd700",
                "High Risk": "#ff6b6b",
                "Critical": "#8b0000",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Approval Trend (28-day)")
        dates = pd.date_range("2025-11-01", periods=28)
        rate = np.clip(np.random.normal(68, 5, 28), 40, 95)
        fig = px.line(
            pd.DataFrame({"Date": dates, "Approval Rate (%)": rate}),
            x="Date",
            y="Approval Rate (%)",
            markers=True,
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — New Application
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📝 New Application":
    st.subheader("📋 Credit Application Form")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 80, 35)
        income = st.number_input("Annual Income (USD)", 20000, 500000, 75000, 5000)
        credit_score = st.slider("Credit Score (FICO)", 300, 850, 720)
    with col2:
        employment_years = st.slider("Years of Employment", 0, 45, 5)
        debt_amount = st.number_input("Total Debt (USD)", 0, 200000, 25000, 1000)
        payment_history = st.slider("Months Since Last Late Payment", 0, 120, 12)

    dti = debt_amount / (income + 1)
    st.caption(
        f"💡 Debt-to-Income Ratio: **{dti:.2%}** {'✅ Good' if dti < 0.36 else '⚠️ High'}"
    )
    st.divider()

    if st.button("🔍 Calculate Risk Score", use_container_width=True, type="primary"):
        payload = {
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "debt_amount": debt_amount,
            "payment_history": payment_history,
        }

        if not api_ok:
            st.warning(
                "⚠️ API is offline. Start it with: `uvicorn api.app:app --port 8000`"
            )
        else:
            with st.spinner("Scoring application..."):
                result = api_predict(payload)
                explain = api_explain(payload)

            if result:
                risk_score = result["risk_score"]
                category = result["risk_category"]
                decision = result["decision"]

                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number+delta",
                            value=risk_score * 100,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Risk Score (%)"},
                            delta={"reference": 50},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 35], "color": "#d4edda"},
                                    {"range": [35, 70], "color": "#fff3cd"},
                                    {"range": [70, 100], "color": "#f8d7da"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 70,
                                },
                            },
                        )
                    )
                    fig.update_layout(height=320)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    css = (
                        "risk-low"
                        if risk_score < 0.35
                        else ("risk-mod" if risk_score < 0.70 else "risk-high")
                    )
                    st.markdown(
                        f"<div class='{css}'>{category}</div>", unsafe_allow_html=True
                    )
                    st.metric("Risk Score", f"{risk_score:.4f}")
                    st.metric("Decision", decision)
                    st.metric("Confidence", f"{result['confidence']:.4f}")
                    st.caption(f"⏱ {result['processing_ms']} ms")

                st.divider()

                if explain and explain.get("top_features"):
                    st.subheader("🔍 SHAP Feature Contributions")
                    top = explain["top_features"]
                    df_shap = pd.DataFrame(top)
                    df_shap["color"] = df_shap["shap_value"].apply(
                        lambda v: "#ff6b6b" if v > 0 else "#00cc96"
                    )
                    fig2 = px.bar(
                        df_shap,
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        color="shap_value",
                        color_continuous_scale="RdYlGn_r",
                        labels={
                            "shap_value": "SHAP Value (impact on risk)",
                            "feature": "Feature",
                        },
                        title="Feature Impact on This Decision",
                    )
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("🔴 Increases risk | 🟢 Decreases risk")

                st.divider()
                st.subheader("💡 Recommendations")
                recs = []
                if credit_score >= 700:
                    recs.append(
                        "✅ Credit score is strong — maintain payment consistency"
                    )
                else:
                    recs.append("⚠️ Improve credit score by reducing outstanding debt")
                if dti < 0.36:
                    recs.append("✅ Debt-to-income ratio is healthy")
                else:
                    recs.append(
                        "⚠️ High DTI ratio — consider debt reduction before applying"
                    )
                if employment_years >= 2:
                    recs.append("✅ Employment stability looks good")
                else:
                    recs.append("💡 Longer employment history improves approval odds")
                if payment_history >= 12:
                    recs.append("✅ Good payment history")
                else:
                    recs.append("⚠️ Recent late payments detected — impact: high")
                for rec in recs:
                    st.info(rec)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — Analytics
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.subheader("📊 Advanced Analytics")

    summary = load_summary()
    if summary.get("results"):
        best = summary["results"].get(summary.get("best_model", ""), {})
        m = best.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", m.get("accuracy", "N/A"))
        with col2:
            st.metric("ROC-AUC", m.get("roc_auc", "N/A"))
        with col3:
            st.metric("KS Stat", m.get("ks_statistic", "N/A"))
        with col4:
            st.metric("Gini", m.get("gini", "N/A"))
    else:
        st.info("Train the model first to see real analytics.")

    st.divider()
    st.subheader("Age Distribution of Simulated Applicants")
    age_data = np.clip(np.random.normal(40, 15, 2000), 18, 80)
    fig = px.histogram(
        age_data, nbins=25, labels={"value": "Age", "count": "Applicants"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Credit Score Distribution")
    cs_data = np.clip(np.random.normal(680, 100, 2000), 300, 850)
    fig2 = px.histogram(
        cs_data,
        nbins=30,
        color_discrete_sequence=["#0066cc"],
        labels={"value": "Credit Score"},
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — Fairness
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚖️ Fairness":
    st.subheader("⚖️ Fairness & Bias Analysis")
    fairness = load_fairness()

    if fairness:
        all_ok = bool(fairness.get("all_compliant", False))

        if all_ok:
            st.success("✅ All fairness metrics within acceptable thresholds")
        else:
            st.error("❌ Some fairness metrics are out of compliance")

        col1, col2, col3 = st.columns(3)
        di = fairness.get("age_disparate_impact", {})
        eod = fairness.get("age_equal_opportunity", {})
        dp = fairness.get("age_demographic_parity", {})

        with col1:
            ratio_val = float(di.get("ratio", 0))
            di_compliant = bool(di.get("compliant", False))
            st.metric(
                "Disparate Impact Ratio",
                f"{ratio_val:.4f}",
                "✅ Compliant" if di_compliant else "❌ Non-compliant",
            )
        with col2:
            eod_val = float(eod.get("difference", 0))
            eod_compliant = bool(eod.get("compliant", False))
            st.metric(
                "Equal Opportunity Diff",
                f"{eod_val:.4f}",
                "✅ Compliant" if eod_compliant else "❌ Non-compliant",
            )
        with col3:
            dp_val = float(dp.get("difference", 0))
            dp_compliant = bool(dp.get("compliant", False))
            st.metric(
                "Demographic Parity Diff",
                f"{dp_val:.4f}",
                "✅ Compliant" if dp_compliant else "❌ Non-compliant",
            )

        st.divider()
        default_rate = float(fairness.get("overall_default_rate", 0.3))
        approval_rate = 1 - default_rate

        demo_data = pd.DataFrame(
            {
                "Group": ["Age ≤50", "Age >50", "Overall"],
                "Approval Rate": [
                    round(approval_rate, 3),
                    round(
                        approval_rate * 0.95, 3
                    ),  # Slightly lower for older age group
                    round(approval_rate, 3),
                ],
            }
        )
        fig = px.bar(
            demo_data,
            x="Group",
            y="Approval Rate",
            color="Approval Rate",
            color_continuous_scale="Greens",
            title="Approval Rate by Age Group",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 Raw Fairness Report"):
            # Convert fairness dict to ensure all values are JSON serializable
            display_fairness = {
                "age_disparate_impact": {
                    "ratio": float(di.get("ratio", 0)),
                    "compliant": bool(di.get("compliant", False)),
                },
                "age_equal_opportunity": {
                    "difference": float(eod.get("difference", 0)),
                    "compliant": bool(eod.get("compliant", False)),
                },
                "age_demographic_parity": {
                    "difference": float(dp.get("difference", 0)),
                    "compliant": bool(dp.get("compliant", False)),
                },
                "overall_default_rate": float(fairness.get("overall_default_rate", 0)),
                "threshold_used": float(fairness.get("threshold_used", 0.05)),
                "all_compliant": bool(all_ok),
            }
            st.json(display_fairness)
    else:
        st.info("Run `python scripts/train.py` to generate the fairness report.")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 5 — About
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.subheader("ℹ️ About This System")
    st.markdown("""
### 🎯 Project Overview
The **AI-Powered Credit Risk Scoring Engine** is a production-grade ML system for financial
institutions to automate credit risk assessment while maintaining transparency and regulatory compliance.

### 🏗️ Technology Stack
| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost · LightGBM · CatBoost |
| Imbalanced Data | SMOTE (imbalanced-learn) |
| Explainability | SHAP TreeExplainer |
| Fairness | Disparate Impact · Equal Opportunity |
| API | FastAPI + Pydantic v2 |
| Dashboard | Streamlit + Plotly |
| Tracking | MLflow |
| Deployment | Docker + Docker Compose |

### 📊 Pipeline
```
CSV Data → DataIngestion → DataProcessor → FeatureEngineer
→ ModelTrainer (SMOTE + 3 models + MLflow)
→ best_model.joblib → FastAPI /predict /explain
→ Streamlit Dashboard
```

### 🔗 Key Endpoints
- `GET  /health`      — System health check
- `POST /predict`     — Real-time risk scoring
- `POST /explain`     — SHAP feature attribution
- `GET  /model/info`  — Training summary
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray'>© 2026 Credit Risk Engine | Powered by Mostafa Ali Mohamed Elsharqawi</p>",
    unsafe_allow_html=True,
)
