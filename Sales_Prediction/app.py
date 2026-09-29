"""Streamlit interface for the saved advertising-to-sales model."""

import json
import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "sales_prediction_model.pkl"
METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"
HERO_IMAGE_PATH = PROJECT_DIR / "assets" / "advertising-analytics.jpg"
FEATURES = ["TV", "Radio", "Newspaper"]


@st.cache_resource
def load_model():
    """Load the trained model once for this Streamlit session."""
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Sales Prediction", page_icon="📊", layout="centered")

hero_image = base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode("ascii") if HERO_IMAGE_PATH.exists() else ""
st.markdown(
    f"""
    <style>
    .stApp {{ background: #f2f3ed; color: #1f332f; }}
    [data-testid="stMainBlockContainer"] {{ max-width: 1080px; padding-top: 1.5rem; padding-bottom: 3rem; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    h2, h3 {{ color: #203a34; }}
    .hero {{
        min-height: 270px;
        display: flex;
        align-items: flex-end;
        padding: 34px 38px;
        margin-bottom: 30px;
        border-radius: 10px;
        color: #fff;
        background-color: #183b35;
        background-image: linear-gradient(90deg, rgba(13, 38, 34, .92) 0%, rgba(13, 38, 34, .64) 55%, rgba(13, 38, 34, .14) 100%), url("data:image/jpeg;base64,{hero_image}");
        background-position: center 48%;
        background-size: cover;
        box-shadow: 0 14px 32px rgba(24, 47, 41, .15);
    }}
    .hero-copy {{ max-width: 630px; }}
    .hero-kicker {{ margin: 0 0 12px; color: #b7d6c9; font-size: 12px; font-weight: 700; letter-spacing: 1.4px; }}
    .hero-title {{ margin: 0 0 10px; color: #fff; font-size: 38px; font-weight: 700; line-height: 1.12; letter-spacing: 0; }}
    .hero-description {{ max-width: 540px; margin: 0; color: #e1ebe5; font-size: 15px; line-height: 1.55; }}
    div[data-testid="stNumberInput"] label {{ color: #334a44; font-weight: 650; }}
    div[data-testid="stNumberInput"] input {{ border-radius: 5px; background: #fff; }}
    div[data-testid="stButton"] button[kind="primary"] {{
        min-height: 46px; border: 0; border-radius: 5px; background: #176b58;
        color: #fff; font-weight: 700; transition: background-color .16s ease, transform .16s ease;
    }}
    div[data-testid="stButton"] button[kind="primary"]:hover {{ background: #105341; transform: translateY(-1px); }}
    div[data-testid="stMetric"] {{
        min-height: 94px; padding: 14px 16px; border: 1px solid #dce3dc;
        border-top: 3px solid #c87b4a; border-radius: 7px; background: #fff;
    }}
    div[data-testid="stMetricLabel"] p {{ color: #5d7169; font-weight: 650; }}
    .section-note {{ color: #60736c; font-size: 14px; }}
    @media (max-width: 640px) {{
        .hero {{ min-height: 235px; padding: 25px 22px; background-position: 58% center; }}
        .hero-title {{ font-size: 30px; }}
        [data-testid="stMainBlockContainer"] {{ padding: 1rem 1rem 2rem; }}
    }}
    </style>
    <section class="hero">
      <div class="hero-copy">
        <p class="hero-kicker">ADVERTISING MIX / SALES OUTLOOK</p>
        <h1 class="hero-title">📊 Sales Prediction Using Machine Learning</h1>
        <p class="hero-description">Estimate product sales from TV, radio, and newspaper advertising budgets with a trained Linear Regression model.</p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run `python src/train_model.py` first.")
    st.stop()
if not METRICS_PATH.exists():
    st.error("Model metrics are missing. Run `python src/train_model.py` to create them.")
    st.stop()

model = load_model()
with METRICS_PATH.open(encoding="utf-8") as metrics_file:
    metrics = json.load(metrics_file)

st.subheader("Set campaign budgets")
st.caption("Enter values in the same units used by the dataset. Budgets must be non-negative.")
budget_columns = st.columns(3)
tv_budget = budget_columns[0].number_input("TV Advertising Budget", min_value=0.0, value=150.0, step=1.0)
radio_budget = budget_columns[1].number_input("Radio Advertising Budget", min_value=0.0, value=25.0, step=1.0)
newspaper_budget = budget_columns[2].number_input("Newspaper Advertising Budget", min_value=0.0, value=20.0, step=1.0)

if st.button("Predict Sales", type="primary", use_container_width=True):
    input_data = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]], columns=FEATURES)
    predicted_sales = float(model.predict(input_data)[0])
    st.metric("Predicted Sales", f"{predicted_sales:.2f}")

st.divider()
st.subheader("Model and test performance")
st.write("**Algorithm:** Linear Regression, trained with an 80/20 split and `random_state=42`.")
st.write(
    "Each input is one feature the model uses to estimate Sales. A coefficient describes the "
    "estimated sales change for a one-unit budget increase while other budgets are held constant."
)
metric_columns = st.columns(4)
metric_labels = [("MAE", "mae"), ("MSE", "mse"), ("RMSE", "rmse"), ("R²", "r2")]
for column, (label, key) in zip(metric_columns, metric_labels):
    column.metric(label, f"{metrics[key]:.3f}")
st.caption(
    "MAE is average absolute error; MSE averages squared errors; RMSE is error in sales units; "
    "R² describes variance explained on this test split. These metrics do not imply perfect predictions."
)