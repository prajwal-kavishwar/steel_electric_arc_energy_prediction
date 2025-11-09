import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/merged_cleaned.csv"

st.title("EAF Energy Consumption Predictor")
st.caption("Trains a fresh model on startup using your processed dataset, then predicts MWh.")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", ".", regex=False)
                .replace({"": np.nan})
            )
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame):
    if "duration_hours" not in df.columns and "DURATION_SEC" in df.columns:
        df["duration_hours"] = df["DURATION_SEC"] / 3600.0

    if "Energy_MWh" not in df.columns:
        if {"MW_mean", "duration_hours"}.issubset(df.columns):
            df["Energy_MWh"] = df["MW_mean"] * df["duration_hours"]
        else:
            raise ValueError("Cannot create Energy_MWh: need MW_mean and duration_hours.")

    FEATURES = [
        "DURATION_SEC", "MW_mean",
        "TEMP_mean", "TEMP_p95",
        "VALO2_mean", "VALO2_p95",
        "O2_AMOUNT_sum", "GAS_AMOUNT_sum",
        "O2_FLOW_mean", "GAS_FLOW_mean",
        "duration_hours",
    ]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    X = df[FEATURES].copy()
    y = df["Energy_MWh"].copy()

    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    if len(X) < 50:
        raise ValueError(f"Too few rows after cleaning: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    try:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
    except Exception:
        model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

    return {
        "model": model,
        "FEATURES": FEATURES,
        "X_sample": X_test.head(5),
        "y_sample": y_test.head(5),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}")
    st.stop()

try:
    bundle = train_model(df)
except Exception as e:
    st.error(f"Training error: {e}")
    st.stop()

model = bundle["model"]
FEATURES = bundle["FEATURES"]

with st.expander("Dataset & training summary"):
    st.write(f"Rows: {len(df):,}")
    st.write(f"Train/Test: {bundle['train_size']:,} / {bundle['test_size']:,}")
    st.write("Feature columns used:", FEATURES)

st.subheader("Manual Prediction")

col1, col2 = st.columns(2)
with col1:
    DURATION_SEC = st.number_input("Process Duration (seconds)", min_value=1, value=120000, step=60)
    MW_mean = st.number_input("Average Power (MW)", min_value=0.0, value=6.5, step=0.1)
    TEMP_mean = st.number_input("Average Temperature (°C)", min_value=800.0, value=1635.0, step=1.0)
    TEMP_p95 = st.number_input("95th Percentile Temperature (°C)", min_value=800.0, value=1650.0, step=1.0)
    VALO2_mean = st.number_input("Avg Oxygen (ppm)", min_value=0.0, value=800.0, step=1.0)
with col2:
    VALO2_p95 = st.number_input("95th Percentile O2 (ppm)", min_value=0.0, value=1000.0, step=1.0)
    O2_AMOUNT_sum = st.number_input("Total Oxygen (Nm³)", min_value=0.0, value=500000.0, step=1000.0)
    GAS_AMOUNT_sum = st.number_input("Total Gas (Nm³)", min_value=0.0, value=150000.0, step=1000.0)
    O2_FLOW_mean = st.number_input("Avg O2 Flow (Nm³/hr)", min_value=0.0, value=2000.0, step=10.0)
    GAS_FLOW_mean = st.number_input("Avg Gas Flow (Nm³/hr)", min_value=0.0, value=800.0, step=10.0)

if st.button("Predict Energy (MWh)"):
    duration_hours = DURATION_SEC / 3600.0

    sample = pd.DataFrame([{
        "DURATION_SEC": DURATION_SEC,
        "MW_mean": MW_mean,
        "TEMP_mean": TEMP_mean,
        "TEMP_p95": TEMP_p95,
        "VALO2_mean": VALO2_mean,
        "VALO2_p95": VALO2_p95,
        "O2_AMOUNT_sum": O2_AMOUNT_sum,
        "GAS_AMOUNT_sum": GAS_AMOUNT_sum,
        "O2_FLOW_mean": O2_FLOW_mean,
        "GAS_FLOW_mean": GAS_FLOW_mean,
        "duration_hours": duration_hours
    }])[FEATURES]

    pred = model.predict(sample)[0]
    st.success(f"Predicted Energy Consumption: {pred:.2f} MWh")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(["Predicted MWh", "Duration (hrs)", "Avg Power (MW)"],
                   [pred, duration_hours, MW_mean])
    ax.set_xlabel("Value Scale")
    ax.set_title("Key Prediction Metrics")
    st.pyplot(fig)
