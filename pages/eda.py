import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pydeck as pdk
from src.config import PROCESSED_CSV, AIRPORTS_CSV, ZRH_LAT, ZRH_LON

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_CSV, parse_dates=["date", "scheduled_dt"])

@st.cache_data
def load_airports() -> pd.DataFrame:
    a = pd.read_csv(AIRPORTS_CSV, sep=";")
    for col in ("Latitude", "Longitude"):
        a[col] = a[col].astype(str).str.replace(",", ".").astype(float)
    return a

df = load_data()
airports = load_airports()

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

# Sidebar filters
movement = st.sidebar.radio(
    "Type beweging", ["Alle", "Aankomsten", "Vertrekken"]
)
if movement == "Aankomsten":
    df = df[df["LSV"] == "L"]
elif movement == "Vertrekken":
    df = df[df["LSV"] == "S"]

# --- Page content ---
st.title("Data-analyse")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Vluchten", f"{len(df):,}")
with col2:
    avg_delay = df["delay_minutes"].mean()
    st.metric("Gem. vertraging", f"{avg_delay:.1f} min")
with col3:
    pct_delayed = (df["delay_minutes"] > 15).mean() * 100
    st.metric("Vertraagd > 15 min", f"{pct_delayed:.1f}%")
with col4:
    unique_destinations = df["Org/Des"].nunique()
    st.metric("Unieke bestemmingen", f"{unique_destinations:,}")
with col5:
    unique_carriers = df["carrier"].nunique()
    st.metric("Unieke maatschappijen", f"{unique_carriers:,}")

st.subheader("Routenetwerk vanuit ZRH")

route_stats = (
    df.groupby("Org/Des")
    .agg(
        vluchten=("FLT", "size"),
        gem_vertraging=("delay_minutes", "mean"),
        vertr_perc=("is_delayed", "mean"),
    )
    .reset_index()
    .merge(
        airports[["ICAO", "Latitude", "Longitude", "Name", "Country"]],
        left_on="Org/Des",
        right_on="ICAO",
        how="inner",
    )
)
route_stats["gem_vertraging_str"] = (
    route_stats["gem_vertraging"].round(1).astype(str) + " min"
)
route_stats["vertr_perc_str"] = (route_stats["vertr_perc"] * 100).round(
    1
).astype(str) + "%"

min_flights = st.slider("Minimum vluchten per route", 10, 500, 50, step=10)
arc_data = route_stats[route_stats["vluchten"] >= min_flights].copy()
arc_data["src_lat"], arc_data["src_lon"] = ZRH_LAT, ZRH_LON

vmin, vmax = arc_data["vertr_perc"].min(), arc_data["vertr_perc"].max()
norm = (arc_data["vertr_perc"] - vmin) / max(vmax - vmin, 1e-9)
arc_data["r"] = (norm * 255 + (1 - norm) * 0).astype(int)
arc_data["g"] = ((1 - norm) * 200 + norm * 30).astype(int)
arc_data["b"] = ((1 - norm) * 255 + norm * 30).astype(int)

st.pydeck_chart(
    pdk.Deck(
        layers=[
            pdk.Layer(
                "ArcLayer",
                data=arc_data,
                get_source_position=["src_lon", "src_lat"],
                get_target_position=["Longitude", "Latitude"],
                get_source_color=["r", "g", "b", 200],
                get_target_color=["r", "g", "b", 240],
                get_width=2,
                pickable=True,
                auto_highlight=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=arc_data,
                get_position=["Longitude", "Latitude"],
                get_radius="vluchten",
                radius_scale=20,
                get_fill_color=["r", "g", "b", 240],
                pickable=True,
            ),
        ],
        initial_view_state=pdk.ViewState(
            latitude=ZRH_LAT, longitude=ZRH_LON, zoom=2.5, pitch=25
        ),
        tooltip={
            "html": "<b>{Name}</b> ({Country})<br/>"
            "Vluchten: {vluchten}<br/>"
            "Gem. vertraging: {gem_vertraging_str}<br/>"
            "Vertragingspercentage: {vertr_perc_str}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        },
    )
)
low_pct = f"{vmin * 100:.0f}%"
high_pct = f"{vmax * 100:.0f}%"
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:8px; margin:4px 0 8px;">
        <span style="font-size:0.85em;">{low_pct}</span>
        <div style="height:12px; width:200px; border-radius:4px;
             background:linear-gradient(to right, rgb(0,200,255), rgb(255,30,30));"></div>
        <span style="font-size:0.85em;">{high_pct}</span>
        <span style="font-size:0.8em; color:gray; margin-left:4px;">vertragingspercentage</span>
    </div>
    <div style="font-size:0.8em; color:gray;">{len(arc_data)} bestemmingen</div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Maandelijks vluchtvolume")
monthly = (
    df.groupby(df["date"].dt.to_period("M"))
    .size()
    .reset_index(name="vluchten")
)
monthly["date"] = monthly["date"].dt.to_timestamp()
st.line_chart(monthly.set_index("date")["vluchten"], width='stretch')
st.caption("De impact van COVID-19 is duidelijk zichtbaar vanaf maart 2020.")

# ── 1. TIJDGERELATEERD ────────────────────────────────────────────────────────
st.subheader("Tijdgerelateerd")
tcol1, tcol2 = st.columns(2)
with tcol1:
    st.markdown("**Vertraging per uur**")
    hourly = (
        df.groupby("hour")
        .agg(vertr_perc=("is_delayed", "mean"), vluchten=("FLT", "size"))
        .reset_index()
    )
    fig_h = go.Figure()
    fig_h.add_trace(
        go.Bar(
            x=hourly["hour"],
            y=hourly["vluchten"],
            name="Vluchten",
            yaxis="y",
            opacity=0.3,
        )
    )
    fig_h.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["vertr_perc"],
            name="Vertraagd %",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="red", width=2),
        )
    )
    fig_h.update_layout(
        yaxis=dict(title="Vluchten"),
        yaxis2=dict(
            title="Vertraagd %",
            overlaying="y",
            side="right",
            range=[0, hourly["vertr_perc"].max() * 1.2],
        ),
        xaxis=dict(title="Uur", dtick=1),
        legend=dict(x=0.01, y=0.99),
        height=350,
    )
    st.plotly_chart(fig_h, width="stretch")
with tcol2:
    st.markdown("**Vertraging per dag v/d week**")
    day_labels = {
        0: "Ma",
        1: "Di",
        2: "Wo",
        3: "Do",
        4: "Vr",
        5: "Za",
        6: "Zo",
    }
    daily = (
        df.groupby("day_of_week")
        .agg(vertr_perc=("is_delayed", "mean"), vluchten=("FLT", "size"))
        .reset_index()
    )
    daily["dag"] = daily["day_of_week"].map(day_labels)
    fig_d = go.Figure()
    fig_d.add_trace(
        go.Bar(
            x=daily["dag"],
            y=daily["vluchten"],
            name="Vluchten",
            yaxis="y",
            opacity=0.3,
        )
    )
    fig_d.add_trace(
        go.Scatter(
            x=daily["dag"],
            y=daily["vertr_perc"],
            name="Vertraagd %",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="red", width=2),
        )
    )
    fig_d.update_layout(
        yaxis=dict(title="Vluchten"),
        yaxis2=dict(
            title="Vertraagd %",
            overlaying="y",
            side="right",
            range=[0, daily["vertr_perc"].max() * 1.2],
        ),
        xaxis=dict(title="Dag"),
        legend=dict(x=0.01, y=0.99),
        height=350,
    )
    st.plotly_chart(fig_d, width="stretch")

# ── 2. LUCHTHAVEN / INFRASTRUCTUUR ───────────────────────────────────────────
st.subheader("Luchthaven & Infrastructuur")

# Row 1: Vertraging per baanconfiguratie (full width)
st.markdown("**Vertraging per baanconfiguratie**")
rwc_stats = (
    df.groupby("RWC")
    .agg(vertr_perc=("is_delayed", "mean"), vluchten=("FLT", "size"))
    .sort_values("vertr_perc", ascending=True)
    .reset_index()
)
# Calculate dynamic height based on number of runway configs
rwc_height = max(350, len(rwc_stats) * 40)

fig_rwc = px.bar(
    rwc_stats,
    x="vertr_perc",
    y="RWC",
    orientation="h",
    color="vertr_perc",
    color_continuous_scale="RdYlGn_r",
    hover_data=["vluchten"],
    labels={
        "vertr_perc": "Vertraagd %",
        "RWC": "Baanconfiguratie",
        "vluchten": "Vluchten",
    },
)
fig_rwc.update_layout(height=rwc_height, coloraxis_showscale=False)
st.plotly_chart(fig_rwc, width="stretch")

# Row 2: Vertraagd % & Gem. vertraging per bewegingstype (2 columns)
lcol1, lcol2 = st.columns(2)
lsv_stats = (
    df.groupby("LSV")
    .agg(
        vertr_perc=("is_delayed", "mean"),
        gem_vertraging=("delay_minutes", "mean"),
        vluchten=("FLT", "size"),
    )
    .reset_index()
)
lsv_stats["type"] = lsv_stats["LSV"].map({"L": "Aankomst", "S": "Vertrek"})
colors = {"Aankomst": "steelblue", "Vertrek": "tomato"}

with lcol1:
    st.markdown("**Vertraagd % per bewegingstype**")
    fig_lsv1 = px.bar(
        lsv_stats,
        x="type",
        y="vertr_perc",
        color="type",
        color_discrete_map=colors,
        text_auto=".1%",
        labels={"vertr_perc": "Vertraagd %", "type": ""},
    )
    fig_lsv1.update_layout(
        height=350, showlegend=False, margin=dict(t=10, b=10)
    )
    fig_lsv1.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_lsv1, width="stretch")

with lcol2:
    st.markdown("**Gem. vertraging per bewegingstype**")
    fig_lsv2 = px.bar(
        lsv_stats,
        x="type",
        y="gem_vertraging",
        color="type",
        color_discrete_map=colors,
        text_auto=".1f",
        labels={"gem_vertraging": "Gem. vertraging (min)", "type": ""},
    )
    fig_lsv2.update_layout(
        height=350, showlegend=False, margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_lsv2, width="stretch")

# ── 3. WEER ───────────────────────────────────────────────────────────────────
st.subheader("Weerinvloeden")
weather_labels = {
    "temperature_2m": "Temperatuur (°C)",
    "wind_gusts_10m": "Windstoten (m/s)",
    "wind_speed_10m": "Windsnelheid (m/s)",
    "precipitation": "Neerslag (mm)",
    "snowfall": "Sneeuwval (cm)",
    "cloud_cover": "Bewolking (%)",
}
wcol1, wcol2 = st.columns(2)
for wcol, (key, label) in zip(
    [wcol1, wcol2], list(weather_labels.items())[:2]
):
    with wcol:
        st.markdown(f"**Vertraging {label.split(' (')[0].lower()}**")
        bins = pd.qcut(df[key], q=10, duplicates="drop")
        w_agg = (
            df.groupby(bins, observed=True)
            .agg(vertr_perc=("is_delayed", "mean"), vluchten=("FLT", "size"))
            .reset_index()
        )
        w_agg[key] = w_agg[key].apply(
            lambda iv: f"{iv.left:.1f} – {iv.right:.1f}"
        )
        fig_w = px.bar(
            w_agg,
            x=key,
            y="vertr_perc",
            hover_data=["vluchten"],
            labels={"vertr_perc": "Vertraagd %", key: label},
        )
        fig_w.update_layout(height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig_w, width="stretch")
