import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PROCESSED_CSV

st.title("Data-analyse")


@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_CSV, parse_dates=["date", "scheduled_dt"])


df = load_data()

st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Datumbereik",
    value=[df["date"].min(), df["date"].max()],
    min_value=df["date"].min(),
    max_value=df["date"].max(),
)
if len(date_range) == 2:
    df = df[
        (df["date"] >= pd.Timestamp(date_range[0]))
        & (df["date"] <= pd.Timestamp(date_range[1]))
    ]

movement = st.sidebar.radio(
    "Type beweging", ["Alle", "Aankomsten", "Vertrekken"]
)
if movement == "Aankomsten":
    df = df[df["LSV"] == "L"]
elif movement == "Vertrekken":
    df = df[df["LSV"] == "S"]

st.metric("Gefilterde vluchten", f"{len(df):,}")

st.subheader("Vertragingspercentage per uur")
hourly = (
    df.groupby("hour")
    .agg(
        vertr_perc=("is_delayed", "mean"),
        vluchten=("FLT", "size"),
    )
    .reset_index()
)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=hourly["hour"],
        y=hourly["vluchten"],
        name="Vluchten",
        yaxis="y",
        opacity=0.3,
    )
)
fig.add_trace(
    go.Scatter(
        x=hourly["hour"],
        y=hourly["vertr_perc"],
        name="Vertragingspercentage",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="red", width=2),
    )
)
fig.update_layout(
    yaxis=dict(title="Aantal vluchten"),
    yaxis2=dict(
        title="Vertragingspercentage",
        overlaying="y",
        side="right",
        range=[0, hourly["vertr_perc"].max() * 1.2],
    ),
    xaxis=dict(title="Uur", dtick=1),
    legend=dict(x=0.01, y=0.99),
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Vertragingspercentage per maatschappij")
top_n = st.slider("Toon top N maatschappijen", 5, 30, 15)
carrier_stats = (
    df.groupby("carrier")
    .agg(
        vluchten=("FLT", "size"),
        vertr_perc=("is_delayed", "mean"),
        gem_vertraging=("delay_minutes", "mean"),
    )
    .sort_values("vluchten", ascending=False)
    .head(top_n)
    .sort_values("vertr_perc", ascending=True)
    .reset_index()
)
fig2 = px.bar(
    carrier_stats,
    x="vertr_perc",
    y="carrier",
    orientation="h",
    color="vertr_perc",
    color_continuous_scale="RdYlGn_r",
    hover_data=["vluchten", "gem_vertraging"],
    labels={
        "vertr_perc": "Vertragingspercentage",
        "carrier": "Maatschappij",
        "vluchten": "Vluchten",
        "gem_vertraging": "Gem. vertraging (min)",
    },
)
fig2.update_layout(height=max(300, top_n * 28), coloraxis_showscale=False)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Invloed van weer op vertragingen")
weather_labels = {
    "wind_speed_10m": "Windsnelheid (m/s)",
    "wind_gusts_10m": "Windstoten (m/s)",
    "precipitation": "Neerslag (mm)",
    "snowfall": "Sneeuwval (cm)",
    "temperature_2m": "Temperatuur (°C)",
    "cloud_cover": "Bewolking (%)",
}
weather_col = st.selectbox(
    "Weervariabele",
    list(weather_labels.keys()),
    format_func=lambda x: weather_labels[x],
)
bins = pd.qcut(df[weather_col], q=10, duplicates="drop")
weather_agg = (
    df.groupby(bins, observed=True)
    .agg(
        vertr_perc=("is_delayed", "mean"),
        vluchten=("FLT", "size"),
    )
    .reset_index()
)
weather_agg[weather_col] = weather_agg[weather_col].astype(str)

fig3 = px.bar(
    weather_agg,
    x=weather_col,
    y="vertr_perc",
    labels={
        "vertr_perc": "Vertragingspercentage",
        weather_col: weather_labels[weather_col],
    },
    hover_data=["vluchten"],
)
fig3.update_layout(height=400)
st.plotly_chart(fig3, use_container_width=True)
