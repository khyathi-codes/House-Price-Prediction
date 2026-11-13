"""House Price Prediction - Streamlit app.

Run locally:  streamlit run app.py
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

from train import DATA_PATH, OCEAN_TYPES, build_features

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

st.markdown(
    """
    <style>
      .stAppDeployButton, #MainMenu {display: none !important;}
      .block-container {padding-top: 2.2rem; max-width: 1150px;}
      h1 {font-weight: 800; letter-spacing: -0.5px;}
      .price-card {
        border: 1px solid rgba(168,85,247,.45); border-radius: 18px; padding: 26px 28px;
        background: linear-gradient(135deg, rgba(168,85,247,.16), rgba(59,130,246,.08));
      }
      .price-card .label {font-size: .8rem; letter-spacing: .18em; text-transform: uppercase; color: #c4b5fd;}
      .price-card .value {font-size: 3rem; font-weight: 800; line-height: 1.15; margin: 6px 0;}
      .price-card .range {color: #9ca3af; font-size: .92rem;}
      .mini-row {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 14px;}
      .mini {border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.03); border-radius: 14px; padding: 12px 14px;}
      .mini-name {font-size: .75rem; color: #9ca3af;}
      .mini-value {font-size: 1.15rem; font-weight: 700; margin-top: 2px; white-space: nowrap;}
      div[data-testid="stMetric"] {background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px; padding: 14px 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    """Load trained models; train them first if the files are missing."""
    if not (os.path.exists("models.joblib") and os.path.exists("metrics.json")):
        import train

        train.main()
    bundle = joblib.load("models.joblib")
    with open("metrics.json") as f:
        info = json.load(f)
    return bundle["models"], bundle["columns"], info


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


models, columns, info = load_artifacts()
data = load_data()
metrics = info["metrics"]

st.title("🏠 House Price Prediction")
st.caption(
    "Machine learning system that estimates house prices from property and location features. "
    "Trained on the California Housing dataset (20,640 census block groups, 1990)."
)

tab_predict, tab_models, tab_data = st.tabs(["🏠 Predict Price", "📊 Model Comparison", "🗺️ Data Explorer"])

# ----------------------------------------------------------------- Predict
with tab_predict:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.subheader("Property details")
        c1, c2 = st.columns(2)
        income = c1.slider("Median income ($10,000s)", 0.5, 15.0, 3.5, 0.1,
                           help="Median household income of the area, in units of $10,000.")
        age = c2.slider("Housing median age (years)", 1, 52, 28)

        c3, c4, c5 = st.columns(3)
        rooms = c3.number_input("Total rooms", 2, 40000, 2100, step=50)
        bedrooms = c4.number_input("Total bedrooms", 1, 7000, 430, step=10)
        households = c5.number_input("Households", 1, 6000, 500, step=10)

        c6, c7 = st.columns(2)
        population = c6.number_input("Population", 3, 36000, 1400, step=50)
        ocean = c7.selectbox("Ocean proximity", OCEAN_TYPES, index=1)

        c8, c9 = st.columns(2)
        lat = c8.number_input("Latitude", 32.5, 42.0, 34.2, step=0.1, format="%.2f")
        lon = c9.number_input("Longitude", -124.4, -114.3, -118.5, step=0.1, format="%.2f")

        model_name = st.selectbox("Prediction model", list(models), index=2)

    row = pd.DataFrame([{
        "longitude": lon, "latitude": lat, "housing_median_age": age, "total_rooms": rooms,
        "total_bedrooms": bedrooms, "population": population, "households": households,
        "median_income": income, "ocean_proximity": ocean,
    }])
    X = build_features(row).reindex(columns=columns, fill_value=0)
    predictions = {name: float(m.predict(X)[0]) for name, m in models.items()}
    price = predictions[model_name]
    mae = metrics[model_name]["mae"]

    with right:
        st.subheader("Estimated value")
        st.markdown(
            f"""
            <div class="price-card">
              <div class="label">{model_name}</div>
              <div class="value">${price:,.0f}</div>
              <div class="range">Typical error about ±${mae:,.0f} &nbsp;·&nbsp; likely range
              ${max(price - mae, 0):,.0f} – ${price + mae:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.caption("What every model predicts for these inputs")
        cards = "".join(
            f'<div class="mini"><div class="mini-name">{name}</div><div class="mini-value">${value:,.0f}</div></div>'
            for name, value in predictions.items()
        )
        st.markdown(f'<div class="mini-row">{cards}</div>', unsafe_allow_html=True)
        st.caption("Prices reflect 1990 California census data and are for demonstration only.")

# --------------------------------------------------------- Model comparison
with tab_models:
    st.subheader("How the three models perform")
    st.caption(
        f"Evaluated on {info['test_rows']:,} unseen houses "
        f"(80/20 split, {info['train_rows']:,} used for training)."
    )

    table = pd.DataFrame(metrics).T
    table.columns = ["R² score", "MAE ($)", "RMSE ($)"]
    best = table["R² score"].idxmax()

    m1, m2, m3 = st.columns(3)
    m1.metric("Best model", best)
    m2.metric("Best R² score", f"{table.loc[best, 'R² score']:.3f}")
    m3.metric("Average error (MAE)", f"${table.loc[best, 'MAE ($)']:,.0f}")

    st.dataframe(
        table.style.format({"R² score": "{:.3f}", "MAE ($)": "${:,.0f}", "RMSE ($)": "${:,.0f}"}),
        use_container_width=True,
    )

    g1, g2 = st.columns(2, gap="large")
    g1.caption("R² score (higher is better)")
    g1.bar_chart(table["R² score"], color="#a855f7")
    g2.caption("Average error, MAE in $ (lower is better)")
    g2.bar_chart(table["MAE ($)"], color="#3b82f6")

    st.subheader("What drives the price? (Random Forest)")
    imp = pd.Series(info["importances"]).head(8).sort_values(ascending=False)
    st.bar_chart(imp, color="#a855f7")

# ------------------------------------------------------------ Data explorer
with tab_data:
    st.subheader("California Housing dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Records", f"{len(data):,}")
    d2.metric("Features", f"{data.shape[1] - 1}")
    d3.metric("Median price", f"${data['median_house_value'].median():,.0f}")
    d4.metric("Median income", f"${data['median_income'].median() * 10000:,.0f}")

    st.caption("Map of 3,000 sampled neighbourhoods")
    st.map(data.sample(3000, random_state=1), latitude="latitude", longitude="longitude", size=250, color="#a855f7")

    st.caption("Price by ocean proximity (average, $)")
    st.bar_chart(data.groupby("ocean_proximity")["median_house_value"].mean().round(0), color="#3b82f6")

    with st.expander("Preview the raw data"):
        st.dataframe(data.head(50), use_container_width=True)
