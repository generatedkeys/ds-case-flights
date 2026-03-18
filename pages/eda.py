from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Exploratory Data Analysis")
st.write("Verken de vluchtgegevens en bekijk vertragingstrends door de tijd heen.")

DATA_DIR = Path("data")
MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "schedule_airport.csv")


@st.cache_data
def load_airports():
    airports = pd.read_csv(DATA_DIR / "airports-extended-clean.csv", sep=";")
    return airports[["IATA", "Name"]].dropna().drop_duplicates(subset=["IATA"])


def build_delay_features(frame: pd.DataFrame) -> pd.DataFrame:
    scheduled_ts = pd.to_datetime(
        frame["STD"].astype(str) + " " + frame["STA_STD_ltc"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    actual_ts = pd.to_datetime(
        frame["STD"].astype(str) + " " + frame["ATA_ATD_ltc"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    flight_date = pd.to_datetime(frame["STD"], format="%d/%m/%Y", errors="coerce")
    delay_minutes = ((actual_ts - scheduled_ts).dt.total_seconds().div(60) + 720) % 1440 - 720

    return frame.assign(
        flight_date=flight_date,
        delay_minutes=delay_minutes,
        year=flight_date.dt.year,
        month=flight_date.dt.month,
        year_month=flight_date.dt.to_period("M").dt.to_timestamp(),
    )


def prepare_top_airports(frame: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    return (
        frame["Org/Des"]
        .value_counts()
        .head(5)
        .rename_axis("Airport Code")
        .reset_index(name="Number of Flights")
        .merge(airports, left_on="Airport Code", right_on="IATA", how="left")
        .assign(Airport=lambda df: df["Name"].fillna(df["Airport Code"]))
        [["Airport", "Number of Flights"]]
        .set_index("Airport")
    )


def prepare_monthly_delay_views(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    delay_data = frame.dropna(subset=["year_month", "delay_minutes"])
    monthly_delay = (
        delay_data.groupby("year_month", as_index=False)
        .agg(avg_delay_minutes=("delay_minutes", "mean"), flights=("delay_minutes", "size"))
        .sort_values("year_month")
    )
    monthly_pattern = (
        delay_data.groupby(["year", "month"])["delay_minutes"]
        .mean()
        .unstack("year")
        .rename(index=MONTH_LABELS)
    )
    delayed_share = (
        delay_data.assign(delayed=delay_data["delay_minutes"] > 0)
        .groupby("year_month", as_index=False)["delayed"]
        .mean()
        .rename(columns={"delayed": "share_delayed"})
    )
    delayed_share["share_delayed"] = delayed_share["share_delayed"] * 100
    return delay_data, monthly_delay, monthly_pattern, delayed_share

data = load_data()
airports = load_airports()
eda_data = build_delay_features(data)

# Display the data
st.subheader("Overzicht van de dataset")
st.dataframe(data)

# Basic EDA
st.subheader("Basisstatistieken")
st.write(data.describe(include="all"))

st.subheader("Ontbrekende waarden")
st.write(data.isnull().sum())

# Top five airports by number of flights
st.subheader("Top 5 luchthavens op aantal vluchten")
st.bar_chart(prepare_top_airports(data, airports))

st.subheader("Vertraging per maand over twee jaar")
delay_data, monthly_delay, monthly_pattern, delayed_share = prepare_monthly_delay_views(eda_data)

if delay_data.empty:
    st.warning("De vertraging kon niet worden berekend op basis van de beschikbare geplande en werkelijke tijden.")
else:
    st.caption("Deze tabel helpt om seizoenspatronen per maand tussen jaren te vergelijken.")
    st.dataframe(monthly_pattern.style.format("{:.1f}"))
    st.caption("Aandeel vluchten met meer dan 0 minuten vertraging per maand.")
    st.line_chart(delayed_share.set_index("year_month")["share_delayed"])

# ---------------------------------------------------------------------------
# Feature-engineering-driven EDA graphs
# ---------------------------------------------------------------------------
import plotly.express as px

st.header("Feature Engineering Verkenning")

fe_data = build_delay_features(data).copy()
scheduled_dt = pd.to_datetime(
    fe_data["STD"].astype(str) + " " + fe_data["STA_STD_ltc"].astype(str),
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce",
)
fe_data["hour"] = scheduled_dt.dt.hour
fe_data["day_of_week"] = scheduled_dt.dt.dayofweek
fe_data["carrier"] = fe_data["FLT"].str.extract(r"^([A-Z]{2,3})", expand=False)
fe_data = fe_data.dropna(subset=["delay_minutes"])

DAY_NAMES = {0: "Ma", 1: "Di", 2: "Wo", 3: "Do", 4: "Vr", 5: "Za", 6: "Zo"}

# --- 1. Delay distribution (histogram) ---
st.subheader("1. Verdeling van vertraging (minuten)")
st.caption("Laat de scheefheid zien → motiveert log-transformatie en delay-cap.")
fig1 = px.histogram(
    fe_data, x="delay_minutes", nbins=80,
    labels={"delay_minutes": "Vertraging (min)"},
)
fig1.update_layout(bargap=0.05)
st.plotly_chart(fig1, use_container_width=True)

# --- 2. Average delay by hour of day ---
st.subheader("2. Gemiddelde vertraging per uur van de dag")
st.caption("Toont of het uur van de dag een sterk signaal is voor het model.")
hourly = fe_data.groupby("hour", as_index=False)["delay_minutes"].mean()
fig2 = px.bar(
    hourly, x="hour", y="delay_minutes",
    labels={"hour": "Uur", "delay_minutes": "Gem. vertraging (min)"},
)
st.plotly_chart(fig2, use_container_width=True)

# --- 3. Delay by day of week ---
st.subheader("3. Vertraging per dag van de week")
st.caption("Vergelijkt weekdagen vs. weekend — relevant voor de is_weekend feature.")
daily = fe_data.groupby("day_of_week", as_index=False)["delay_minutes"].mean()
daily["day_name"] = daily["day_of_week"].map(DAY_NAMES)
fig3 = px.bar(
    daily, x="day_name", y="delay_minutes",
    labels={"day_name": "Dag", "delay_minutes": "Gem. vertraging (min)"},
)
st.plotly_chart(fig3, use_container_width=True)

# --- 4. Landing vs Departure (LSV) ---
st.subheader("4. Vertraging: Landing (L) vs. Vertrek (S)")
st.caption("Valideert LSV als categorische feature in het model.")
fig4 = px.box(
    fe_data, x="LSV", y="delay_minutes",
    labels={"LSV": "Landing / Start", "delay_minutes": "Vertraging (min)"},
)
fig4.update_layout(yaxis=dict(range=[-30, 60]))
st.plotly_chart(fig4, use_container_width=True)

# --- 5. Delay by top 10 carriers ---
st.subheader("5. Gemiddelde vertraging per luchtvaartmaatschappij (top 10)")
st.caption("Valideert de geëxtraheerde carrier-feature uit het vluchtnummer.")
top_carriers = fe_data["carrier"].value_counts().head(10).index
carrier_delay = (
    fe_data[fe_data["carrier"].isin(top_carriers)]
    .groupby("carrier", as_index=False)["delay_minutes"]
    .agg(["mean", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)
fig5 = px.bar(
    carrier_delay, x="carrier", y="mean", text="count",
    labels={"carrier": "Carrier", "mean": "Gem. vertraging (min)", "count": "Vluchten"},
)
fig5.update_traces(textposition="outside")
st.plotly_chart(fig5, use_container_width=True)

# --- 6. Delay by aircraft type (top 10) ---
st.subheader("6. Gemiddelde vertraging per vliegtuigtype (top 10)")
st.caption("Valideert ACT als categorische feature — sommige types zijn systematisch later.")
top_act = fe_data["ACT"].value_counts().head(10).index
act_delay = (
    fe_data[fe_data["ACT"].isin(top_act)]
    .groupby("ACT", as_index=False)["delay_minutes"]
    .agg(["mean", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)
fig6 = px.bar(
    act_delay, x="ACT", y="mean", text="count",
    labels={"ACT": "Vliegtuigtype", "mean": "Gem. vertraging (min)", "count": "Vluchten"},
)
fig6.update_traces(textposition="outside")
st.plotly_chart(fig6, use_container_width=True)
