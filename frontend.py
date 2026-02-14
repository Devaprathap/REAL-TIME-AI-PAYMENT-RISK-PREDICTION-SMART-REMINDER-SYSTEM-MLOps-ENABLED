import streamlit as st
import requests
import plotly.graph_objects as go

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(page_title="CollectIQ AI", layout="wide")
API_URL = "https://real-time-ai-payment-risk-prediction.onrender.com"

# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
.main-title {
    font-size:42px;
    font-weight:700;
}
.subtitle {
    font-size:18px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💰 CollectIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Invoice Risk Intelligence Platform</div>', unsafe_allow_html=True)

# ==========================================
# HEALTH CHECK
# ==========================================

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    if health.get("status") == "healthy":
        st.success("🟢 API Status: Healthy")
    else:
        st.error("🔴 API Issue Detected")
except:
    st.error("⚠ Cannot reach backend service")

st.divider()

# ==========================================
# MODEL SELECTION
# ==========================================

model_choice = st.selectbox(
    "Select Model Version",
    ["v1", "v2"]
)

# ==========================================
# INPUT SECTION
# ==========================================

st.header("📥 Enter Invoice Details")

col1, col2 = st.columns(2)

with col1:
    invoice_amount = st.number_input("Invoice Amount", min_value=0.0, value=5000.0)
    avg_delay_days = st.number_input("Average Delay (Days)", min_value=0.0, value=5.0)
    num_past_invoices = st.number_input("Number of Past Invoices", min_value=0.0, value=10.0)

with col2:
    invoice_gap_days = st.number_input("Invoice Gap (Days)", min_value=0.0, value=30.0)
    industry_category = st.number_input("Industry Category (Encoded)", min_value=0.0, value=1.0)
    reliability_score = st.number_input("Reliability Score (0–1)", min_value=0.0, max_value=1.0, value=0.8)

# ==========================================
# PREDICTION
# ==========================================

if st.button("🔮 Predict Risk"):

    payload = {
        "invoice_amount": invoice_amount,
        "avg_delay_days": avg_delay_days,
        "num_past_invoices": num_past_invoices,
        "invoice_gap_days": invoice_gap_days,
        "industry_category": industry_category,
        "reliability_score": reliability_score,
    }

    try:
        with st.spinner("Analyzing invoice risk..."):

            response = requests.post(
                f"{API_URL}/predict?model_version={model_choice}",
                json=payload,
                timeout=60
            )

        if response.status_code == 200:

            result = response.json()

            prob = result.get("late_payment_probability", 0.0)
            action = result.get("recommended_action", "N/A")
            tone = result.get("tone", "N/A")
            model_version = result.get("model_version", "N/A")

            percentage = round(prob * 100, 2)

            # Risk Label
            if prob < 0.3:
                risk_label = "🟢 Low Risk"
            elif prob < 0.7:
                risk_label = "🟡 Medium Risk"
            else:
                risk_label = "🔴 High Risk"

            # Metrics
            st.subheader("📊 Risk Score")
            st.metric("Late Payment Risk", f"{percentage}%")

            st.subheader("📋 Prediction Details")
            st.success(f"Risk Level: {risk_label}")
            st.write(f"**Recommended Action:** {action}")
            st.write(f"**Tone:** {tone}")
            st.write(f"**Model Version Used:** {model_version}")

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percentage,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                },
                title={'text': "Late Payment Risk"}
            ))

            st.plotly_chart(fig, width="stretch")

        else:
            st.error("❌ API Error")
            st.write(response.text)

    except Exception as e:
        st.error("🚨 Connection Error")
        st.write(str(e))

# ==========================================
# LIVE ANALYTICS
# ==========================================

st.divider()
st.header("📈 Live System Analytics")

try:
    stats_response = requests.get(f"{API_URL}/stats", timeout=5)

    if stats_response.status_code == 200:
        stats = stats_response.json()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", stats.get("total_predictions", 0))
        col2.metric("Average Risk", f"{stats.get('average_risk', 0):.2%}")
        col3.metric("High Risk Cases", stats.get("high_risk_predictions", 0))
    else:
        st.warning("Stats unavailable")

except:
    st.warning("Live stats unavailable")

# ==========================================
# EXPLANATION
# ==========================================

st.divider()
st.header("🧠 How CollectIQ Works")

st.markdown("""
1. Invoice data is submitted in real time.
2. AI model predicts late payment probability.
3. System recommends reminder tone.
4. All predictions are logged to database.
5. Dashboard monitors business risk trends.
""")

st.markdown("---")
st.caption("CollectIQ © 2026 | AI Risk Intelligence | Built by Saipriya Shet")

# ==============================
# FEATURE IMPORTANCE SECTION
# ==============================

st.divider()
st.header("📊 Model Feature Importance")

try:
    fi_response = requests.get(f"{API_URL}/feature-importance", timeout=10)

    if fi_response.status_code == 200:
        fi_data = fi_response.json()

        if "feature_importance" in fi_data:

            importance_dict = fi_data["feature_importance"]

            if importance_dict:

                features = list(importance_dict.keys())
                scores = list(importance_dict.values())

                fig_importance = go.Figure(
                    go.Bar(
                        x=scores,
                        y=features,
                        orientation='h'
                    )
                )

                fig_importance.update_layout(
                    title="Feature Importance (XGBoost)",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    height=400
                )

                st.plotly_chart(fig_importance, width="stretch")

            else:
                st.info("No feature importance available.")

        else:
            st.warning("Feature importance not returned by API.")

    else:
        st.warning("Feature importance endpoint unavailable.")

except:
    st.warning("Could not load feature importance.")
