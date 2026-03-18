import streamlit as st
import pandas as pd

from src.config import (
    COL_FLIGHT,
    COL_GATE,
    COL_RUNWAY,
    MODEL_PATH,
    RAW_CSV,
    COL_LSV,
    COL_TERMINAL,
    COL_AIRCRAFT,
    COL_RUNWAY_CONFIG,
    COL_ORG_DES,
)
from src.predict import load_metrics, load_model, predict

st.header("Voorspellingsmodel")
st.caption("Voer vluchtgegevens in om vertragingen te voorspellen.")

if not MODEL_PATH.exists():
    st.warning("Geen getraind model gevonden. Voer eerst `python -m src.train` uit.")
    st.stop()


@st.cache_resource
def _load_model():
    return load_model()


@st.cache_data
def _load_metrics():
    return load_metrics()


@st.cache_data
def _load_unique_values():
    df = pd.read_csv(RAW_CSV, on_bad_lines="skip")
    return {
        COL_LSV: sorted(df[COL_LSV].dropna().unique()),
        COL_TERMINAL: sorted(df[COL_TERMINAL].dropna().unique()),
        "gate_zone": sorted(
            df[COL_GATE]
            .str.extract(r"^([A-Z])", expand=False)
            .dropna()
            .unique()
        ),
        COL_AIRCRAFT: sorted(df[COL_AIRCRAFT].dropna().unique()),
        COL_RUNWAY: sorted(df[COL_RUNWAY].dropna().unique()),
        COL_RUNWAY_CONFIG: sorted(df[COL_RUNWAY_CONFIG].dropna().unique()),
        COL_ORG_DES: sorted(df[COL_ORG_DES].dropna().unique()),
        "carrier": sorted(
            df[COL_FLIGHT]
            .str.extract(r"^([A-Z]{2,3})", expand=False)
            .dropna()
            .unique()
        ),
    }


model = _load_model()
metrics = _load_metrics()
uniques = _load_unique_values()

# -- Metrics ------------------------------------------------------------------

st.subheader("Modelprestaties (testset)")

overall = metrics["overall"]
c1, c2, c3 = st.columns(3)
c1.metric("MAE (min)", f"{overall['mae']:.1f}")
c2.metric("RMSE (min)", f"{overall['rmse']:.1f}")
c3.metric("R²", f"{overall['r2']:.3f}")

if "by_lsv" in metrics:
    st.markdown("**Per bewegingstype**")
    label_map = {"L": "Aankomsten", "S": "Vertrekken"}
    lsv_cols = st.columns(len(metrics["by_lsv"]))
    for col, (group, m) in zip(lsv_cols, metrics["by_lsv"].items()):
        col.metric(f"{label_map.get(group, group)} MAE", f"{m['mae']:.1f} min")

# -- Single-flight prediction -------------------------------------------------

st.subheader("Voorspel vertraging voor een enkele vlucht")

with st.form("predict_form"):
    r1c1, r1c2 = st.columns(2)
    lsv = r1c1.selectbox(
        "Aankomst / Vertrek",
        uniques[COL_LSV],
        format_func=lambda x: "Aankomst" if x == "L" else "Vertrek",
    )
    terminal = r1c2.selectbox("Terminal / Gebied", uniques[COL_TERMINAL])

    r2c1, r2c2 = st.columns(2)
    aircraft = r2c1.selectbox("Vliegtuigtype", uniques[COL_AIRCRAFT])
    gate_zone = r2c2.selectbox("Gate zone", uniques["gate_zone"])

    r3c1, r3c2 = st.columns(2)
    runway = r3c1.selectbox("Start/Landingsbaan", uniques[COL_RUNWAY])
    rw_config = r3c2.selectbox("Baanconfiguratie", uniques[COL_RUNWAY_CONFIG])

    r4c1, r4c2 = st.columns(2)
    org_des = r4c1.selectbox(
        "Oorsprong / Bestemming (ICAO)", uniques[COL_ORG_DES]
    )
    carrier = r4c2.selectbox("Luchtvaartmaatschappij", uniques["carrier"])

    r5c1, r5c2 = st.columns(2)
    sched_time = r5c1.time_input(
        "Geplande tijd", value=pd.Timestamp("08:00").time()
    )
    sched_date = r5c2.date_input("Geplande datum")

    congestion = st.slider("Geschatte bewegingen in ±30 min", 0, 80, 20)
    submitted = st.form_submit_button("Voorspellen")

if submitted:
    dow = pd.Timestamp(sched_date).dayofweek
    row = pd.DataFrame(
        [
            {
                COL_LSV: lsv,
                COL_TERMINAL: terminal,
                "gate_zone": gate_zone,
                COL_AIRCRAFT: aircraft,
                COL_RUNWAY: runway,
                COL_RUNWAY_CONFIG: rw_config,
                COL_ORG_DES: org_des,
                "carrier": carrier,
                "hour": sched_time.hour,
                "minute": sched_time.minute,
                "day_of_week": dow,
                "month": sched_date.month,
                "is_weekend": int(dow >= 5),
                "moves_30m": congestion,
                "moves_60m": int(congestion * 1.8),
                "arr_30m": congestion // 2,
                "dep_30m": congestion - congestion // 2,
            }
        ]
    )
    pred = predict(model, row)[0]
    st.success(f"Voorspelde vertraging: **{pred:.1f} minuten**")
