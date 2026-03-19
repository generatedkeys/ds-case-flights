import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

from src.config import PROCESSED_CSV, BEST_MODEL_PATH
from src.predict import (
    load_best_model,
    load_feature_meta,
    load_thresholds,
    build_feature_matrix,
    predict_delay_proba,
)

st.title("Dagsimulatie")
st.markdown(
    "Kies een datum en bekijk hoe goed het model de vertragingen voorspelt."
)


@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_CSV, parse_dates=["date", "scheduled_dt"])


@st.cache_resource
def _load_model():
    return load_best_model()


@st.cache_data
def _load_meta():
    return load_feature_meta()


df = load_data()

if not BEST_MODEL_PATH.exists():
    st.warning("Geen getraind model gevonden. Voer eerst het notebook uit.")
    st.stop()

model = _load_model()
meta = _load_meta()
_thresholds = load_thresholds()
_best_threshold = _thresholds.get(
    next((k for k in _thresholds if "gradient" in k.lower()), ""), 0.5
)

selected_date = st.date_input(
    "Kies een datum",
    value=pd.Timestamp("2019-07-15").date(),
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date(),
)

day_df = (
    df[df["date"].dt.date == selected_date].copy().sort_values("scheduled_dt")
)

if day_df.empty:
    st.warning("Geen vluchten gevonden op deze datum.")
    st.stop()

day_df["tijd"] = day_df["scheduled_dt"].dt.strftime("%H:%M")

X_day = build_feature_matrix(day_df, meta)
day_df["pred_proba"] = predict_delay_proba(model, X_day)
day_df["pred_delayed"] = (day_df["pred_proba"] >= _best_threshold).astype(int)
day_df["correct"] = day_df["pred_delayed"] == day_df["is_delayed"]

acc = accuracy_score(day_df["is_delayed"], day_df["pred_delayed"])

c1, c2, c3 = st.columns(3)
c1.metric("Vluchten", len(day_df))
c2.metric(
    "Werkelijk vertraagd",
    f"{day_df['is_delayed'].sum()} ({day_df['is_delayed'].mean():.0%})",
)
c3.metric("Model nauwkeurigheid", f"{acc:.0%}")

st.subheader("Vluchtverloop door de dag")

hour_range = st.slider("Tijdsvenster (uur)", 0, 23, (0, 23))
window_df = day_df[
    (day_df["hour"] >= hour_range[0]) & (day_df["hour"] <= hour_range[1])
]


def _delay_cat(m):
    if m <= 0:
        return "Op tijd / vroeg"
    if m <= 15:
        return "Licht (0–15 min)"
    if m <= 60:
        return "Vertraagd (15–60 min)"
    return "Fors (>60 min)"


color_map = {
    "Op tijd / vroeg": "#2ecc71",
    "Licht (0–15 min)": "#f39c12",
    "Vertraagd (15–60 min)": "#e74c3c",
    "Fors (>60 min)": "#8e44ad",
}
window_df["status"] = window_df["delay_minutes"].apply(_delay_cat)

fig = px.strip(
    window_df,
    x="hour",
    y="delay_minutes",
    color="status",
    color_discrete_map=color_map,
    category_orders={"status": list(color_map.keys())},
    hover_data=["FLT", "carrier", "Org/Des", "tijd", "pred_proba", "correct"],
    labels={
        "hour": "Uur",
        "delay_minutes": "Vertraging (min)",
        "status": "Status",
        "FLT": "Vlucht",
        "carrier": "Maatschappij",
        "Org/Des": "Bestemming",
        "tijd": "Gepland",
        "pred_proba": "Voorspelde kans",
        "correct": "Model correct",
    },
)
fig.update_layout(
    height=450,
    xaxis=dict(dtick=1),
    yaxis_range=[
        min(-10, window_df["delay_minutes"].min() - 5),
        max(60, window_df["delay_minutes"].quantile(0.98) + 10),
    ],
)
fig.add_hline(
    y=15,
    line_dash="dash",
    line_color="red",
    opacity=0.5,
    annotation_text="15 min grens",
)
fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
st.plotly_chart(fig, width='stretch')

st.subheader("Werkelijk vs. voorspeld per uur")
hourly = (
    window_df.groupby("hour")
    .agg(
        vertraagd=("is_delayed", "sum"),
        voorspeld=("pred_delayed", "sum"),
    )
    .reset_index()
)

fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=hourly["hour"],
        y=hourly["vertraagd"],
        name="Werkelijk",
        marker_color="#e74c3c",
    )
)
fig2.add_trace(
    go.Bar(
        x=hourly["hour"],
        y=hourly["voorspeld"],
        name="Voorspeld",
        marker_color="#3498db",
    )
)
fig2.update_layout(
    barmode="group",
    height=300,
    xaxis=dict(title="Uur", dtick=1),
    yaxis_title="Vertraagde vluchten",
)
st.plotly_chart(fig2, width='stretch')

st.subheader("Vluchten")
tbl = window_df[
    [
        "tijd",
        "FLT",
        "carrier",
        "LSV",
        "Org/Des",
        "delay_minutes",
        "pred_proba",
        "correct",
    ]
].copy()
tbl["pred_proba"] = (tbl["pred_proba"] * 100).round(1)
tbl["delay_minutes"] = tbl["delay_minutes"].round(1)
tbl.columns = [
    "Tijd",
    "Vlucht",
    "Maatschappij",
    "Type",
    "Bestemming",
    "Vertraging (min)",
    "Kans (%)",
    "Correct",
]
tbl["Type"] = tbl["Type"].map({"L": "Aankomst", "S": "Vertrek"})


def _color(val):
    if val is True:
        return "background-color: rgba(46,204,113,0.15)"
    if val is False:
        return "background-color: rgba(231,76,60,0.15)"
    return ""


st.dataframe(
    tbl.style.applymap(_color, subset=["Correct"]),
    width='stretch',
    height=400,
)
