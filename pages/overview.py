import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from src.config import PROCESSED_CSV, AIRPORTS_CSV, ZRH_LAT, ZRH_LON

st.title("Luchthaven Zürich — Vertragingen Dashboard")
st.markdown(
    "Twee jaar vluchtoperaties op ZRH (2019–2020), "
    "gecombineerd met uurlijkse weerdata."
)


@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_CSV, parse_dates=["date", "scheduled_dt"])


@st.cache_data
def load_airports():
    a = pd.read_csv(AIRPORTS_CSV, sep=";")
    for col in ("Latitude", "Longitude"):
        a[col] = a[col].astype(str).str.replace(",", ".").astype(float)
    return a


df = load_data()
airports = load_airports()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Totaal vluchten", f"{len(df):,}")
c2.metric("Bestemmingen", df["Org/Des"].nunique())
c3.metric("Vertragingspercentage", f"{df['is_delayed'].mean():.1%}")
c4.metric("Mediaan vertraging", f"{df['delay_minutes'].median():.0f} min")

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

thresh = arc_data["vertr_perc"].quantile(0.75)
arc_data["r"] = np.where(arc_data["vertr_perc"] > thresh, 220, 50)
arc_data["g"] = np.where(arc_data["vertr_perc"] > thresh, 60, 130)
arc_data["b"] = np.where(arc_data["vertr_perc"] > thresh, 60, 200)

st.pydeck_chart(
    pdk.Deck(
        layers=[
            pdk.Layer(
                "ArcLayer",
                data=arc_data,
                get_source_position=["src_lon", "src_lat"],
                get_target_position=["Longitude", "Latitude"],
                get_source_color=[50, 130, 200, 160],
                get_target_color=["r", "g", "b", 180],
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
                get_fill_color=["r", "g", "b", 200],
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
st.caption(
    f"{len(arc_data)} bestemmingen. "
    "Rode bogen = top-25% vertragingspercentage."
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
