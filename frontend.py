import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="CollectIQ AI", layout="wide")

st.title("💰 CollectIQ – AI Payment Risk Prediction System")
st.markdown("Real-Time Invoice Risk Intelligence Engine")

API_URL = "https://collectiq-api.onrender.com"

# ----------------------------
# INPUT SECTION
# ----------------------------

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

# ----------------------------
# PREDICTION BUTTON
# ----------------------------

if st.button("🔮 Predict Risk"):

    payload = {
        "invoice_amount": invoice_amount,
        "avg_delay_days": avg_delay_days,
        "num_past_invoices": num_past_invoices,
        "invoice_gap_days": invoice_gap_days,
        "industry_category": industry_category,
        "reliability_score": reliability_score
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            prob = result.get("late_payment_probability", 0)
            action = result.get("recommended_action", "N/A")
            tone = result.get("tone", "N/A")
            model_version = result.get("model_version", "N/A")

            # ----------------------------
            # RISK LEVEL LOGIC
            # ----------------------------
            if prob < 0.3:
                risk_label = "🟢 Low Risk"
            elif prob < 0.7:
                risk_label = "🟡 Medium Risk"
            else:
                risk_label = "🔴 High Risk"

            st.subheader("📊 Prediction Result")
            st.success(f"Risk Level: {risk_label}")
            st.write(f"**Probability:** {prob:.2%}")
            st.write(f"**Recommended Action:** {action}")
            st.write(f"**Tone:** {tone}")
            st.write(f"**Model Version Used:** {model_version}")

            # ----------------------------
            # GAUGE CHART
            # ----------------------------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
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

            st.plotly_chart(fig, width='stretch')

        else:
            st.error("❌ Prediction failed. Check API logs.")

    except Exception as e:
        st.error("🚨 Connection Error")
        st.write(str(e))

# ----------------------------
# SYSTEM STATS
# ----------------------------

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

# ----------------------------
# BUSINESS EXPLANATION PANEL
# ----------------------------

st.divider()
st.header("🧠 How CollectIQ Works")

st.markdown("""
1. Invoice data is submitted in real time.
2. AI model predicts late payment probability.
3. System recommends reminder tone.
4. All predictions are stored in database.
5. Live dashboard tracks system performance.
""")

st.markdown("---")
st.caption("CollectIQ – Real-Time AI Payment Risk Intelligence | Built with FastAPI + XGBoost + Streamlit")
