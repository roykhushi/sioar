import os
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")


def get_base_url():
    return st.session_state.get("api_base_url", API_BASE_URL)


def api_get(path):
    response = requests.get(f"{get_base_url()}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path, payload):
    response = requests.post(f"{get_base_url()}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Smart-Food Link", layout="wide", page_icon="🥗")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .sfl-card {
            padding: 1.25rem;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid #eef2f6;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }
        .sfl-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .sfl-muted { color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Smart-Food Link")
    st.caption("Configure and monitor")
    st.text_input("API Base URL", value=API_BASE_URL, key="api_base_url")
    st.markdown("---")
    st.write("Tip: start the API before running predictions.")

st.markdown('<div class="sfl-title">Smart-Food Link Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sfl-muted">AI-assisted risk prediction and donation matching</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="sfl-card">', unsafe_allow_html=True)
    st.subheader("Model Training")
    st.caption("Retrain the model with the latest data.")
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                result = api_post("/train", {})
                st.success(result.get("message", "Training completed."))
            except requests.RequestException as exc:
                st.error(f"Training failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sfl-card">', unsafe_allow_html=True)
    st.subheader("API Status")
    try:
        api_get("/")
        st.success("Connected to API")
    except requests.RequestException:
        st.warning("API not reachable. Start FastAPI at http://localhost:8000")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="sfl-card">', unsafe_allow_html=True)
    st.subheader("Quick Tips")
    st.write("Use Predict to score items.")
    st.write("Use Donation Match for NGO recommendation.")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

st.header("Predict Item Risk")
with st.container():
    st.markdown('<div class="sfl-card">', unsafe_allow_html=True)
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input("Price", min_value=0.0, step=0.1)
            avg_daily_sales = st.number_input("Avg Daily Sales", min_value=0.0, step=0.1)
        with c2:
            quantity = st.number_input("Quantity", min_value=0, step=1)
            days_until_expiry = st.number_input("Days Until Expiry", min_value=0, step=1)
        submitted = st.form_submit_button("Predict")
    st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    payload = {
        "price": float(price),
        "quantity": int(quantity),
        "avg_daily_sales": float(avg_daily_sales),
        "days_until_expiry": int(days_until_expiry),
    }
    try:
        prediction = api_post("/predict", payload)
        st.success("Prediction complete.")
        risk_level = prediction.get("Risk_Level", "Unknown")
        probability = prediction.get("Probability", 0)
        action = prediction.get("Action", "N/A")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Level", risk_level)
        m2.metric("Probability", f"{probability:.2f}")
        m3.metric("Recommended Action", action)
        with st.expander("Raw response"):
            st.json(prediction)
    except requests.RequestException as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()

st.header("NGO Directory")
try:
    ngos = api_get("/ngos")
    if ngos:
        st.markdown('<div class="sfl-card">', unsafe_allow_html=True)
        st.table(ngos)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No NGOs available.")
except requests.RequestException as exc:
    st.error(f"Unable to load NGOs: {exc}")

st.divider()

st.header("Donation Match")
with st.container():
    st.markdown('<div class="sfl-card">', unsafe_allow_html=True)
    category = st.text_input("Food Category (e.g., Dairy, Bakery, Produce)")
    if st.button("Find NGO"):
        if not category.strip():
            st.warning("Please enter a category.")
        else:
            try:
                match = api_post("/match", {"category": category.strip()})
                recommended = match.get("recommended_ngo")
                if recommended:
                    st.success("Recommended NGO found.")
                    st.json(recommended)
                else:
                    st.info(match.get("message", "No matching NGO found."))
            except requests.RequestException as exc:
                st.error(f"Match failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)
