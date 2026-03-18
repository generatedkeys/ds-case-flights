from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration & Constants ---
DATA_DIR = Path("data")
DAY_NAMES = {0: "Ma", 1: "Di", 2: "Wo", 3: "Do", 4: "Vr", 5: "Za", 6: "Zo"}

# --- Data Functions ---
@st.cache_data
def load_flight_data():
    """Load and perform initial cleaning of flight schedule data."""
    df = pd.read_csv(DATA_DIR / "schedule_airport.csv")
    df["STD"] = pd.to_datetime(df["STD"], format="%d/%m/%Y", errors="coerce")
    return df

@st.cache_data
def load_airport_metadata():
    """Load and clean airport names and IATA codes."""
    df = pd.read_csv(DATA_DIR / "airports-extended-clean.csv", sep=";")
    return df[["IATA", "Name"]].dropna().drop_duplicates(subset=["IATA"])

def process_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate delays and extract date-related features."""
    # Build timestamps for scheduled and actual times
    std_date_str = df["STD"].dt.strftime("%Y-%m-%d")
    scheduled_ts = pd.to_datetime(std_date_str + " " + df["STA_STD_ltc"].astype(str), errors="coerce")
    actual_ts = pd.to_datetime(std_date_str + " " + df["ATA_ATD_ltc"].astype(str), errors="coerce")
    delay_min = (actual_ts - scheduled_ts).dt.total_seconds().div(60) # Calculate delay in minutes
    # Feature extraction
    df = df.assign(
        delay_minutes=delay_min, # Delay in minutes
        delay_positive=delay_min.clip(lower=0), # Positive delay
        hour=scheduled_ts.dt.hour, # Hour of the day
        day_of_week=scheduled_ts.dt.dayofweek, # Day of the week
        year=df["STD"].dt.year, # Year
        month=df["STD"].dt.month, # Month
        carrier=df["FLT"].str.extract(r"^([A-Z0-9]{2,3})", expand=False) # Carrier
    )
    return df

# --- UI Components ---
def render_sidebar(df: pd.DataFrame):
    """Render sidebar filters and return filtered dataframe."""
    st.sidebar.header("Filters")
    years = sorted(df["year"].dropna().unique().astype(int))
    selected_years = st.sidebar.multiselect("Selecteer Jaren", years, default=years)
    mask = df["year"].isin(selected_years)
    return df[mask].dropna(subset=["delay_minutes"]).copy()

def render_metrics(df: pd.DataFrame):
    """Display key summary metrics."""
    m1, m2, m3, m4 = st.columns(4)
    total_flights = len(df)
    avg_delay = df["delay_minutes"].mean()
    pct_delayed = (df["delay_minutes"] > 15).mean() * 100
    unique_dest = df["Org/Des"].nunique()
    m1.metric("Totaal aantal vluchten", f"{total_flights:,}")
    m2.metric("Gem. Vertraging", f"{avg_delay:.1f} min")
    m3.metric("Vertraagde vluchten (>15m)", f"{pct_delayed:.1f}%")
    m4.metric("Unieke Bestemmingen", unique_dest)

def render_temporal_analysis(df: pd.DataFrame):
    """Display delay distribution and monthly trends."""
    st.subheader("Tijd patronen")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Verdeling van vertragingen**")
        fig_hist = px.histogram(
            df[df["delay_minutes"] > 0],
            x="delay_minutes",
            nbins=80,
            labels={"delay_minutes": "Vertraging (minuten)"},
            color_discrete_sequence=["indianred"]
        )
        fig_hist.update_layout(
            showlegend=False,
            yaxis_title="Aantal vluchten",
            margin=dict(t=30)
        )
        st.plotly_chart(fig_hist, width='stretch')
        
    with c2:
        st.markdown("**Maandelijkse Trend: Vluchtvolume & Gemiddelde Vertragingen**")
        monthly = df.groupby(["year", "month"]).agg(
            flights=("delay_minutes", "count"),
            avg_delay=("delay_minutes", "mean")
        ).reset_index()
        monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly["date"],
            y=monthly["flights"],
            name="Vluchten",
            yaxis="y1",
            opacity=0.6
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly["date"],
            y=monthly["avg_delay"],
            name="Gem. Vertraging",
            yaxis="y2",
            line=dict(color="red")
        ))
        
        fig_trend.update_layout(
            yaxis=dict(title="Aantal Vluchten"),
            yaxis2=dict(title="Gem. Vertraging (min)", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h"),
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_trend, width='stretch')

def render_performance_analysis(df: pd.DataFrame):
    """Display carrier and aircraft performance rankings."""
    st.divider()
    st.subheader("Luchtvaartmaatschappijen & Vliegtuigen")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Luchtvaartmaatschappijen met de meeste vertragingen**")
        perf = df.groupby("carrier")["delay_positive"].agg(["mean", "count"]).reset_index()
        perf = perf[perf["count"] > 50].sort_values("mean", ascending=False).head(10)
        fig = px.bar(
            perf,
            x="mean",
            y="carrier",
            orientation="h",
            color="mean",
            color_continuous_scale="OrRd",
            labels={"mean": "Gem. Vertraging (min)", "carrier": "Airline"}
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, margin=dict(t=30))
        st.plotly_chart(fig, width='stretch')
        
    with c2:
        st.markdown("**Vliegtuigtype vs. Vertraging**")
        perf = df.groupby("ACT")["delay_positive"].agg(["mean", "count"]).reset_index()
        perf = perf[perf["count"] > 20].sort_values("mean", ascending=False).head(10)
        fig = px.bar(
            perf,
            x="ACT",
            y="mean",
            color="mean",
            color_continuous_scale="Plasma",
            labels={"mean": "Gem. Vertraging (min)", "ACT": "Vliegtuigtype"}
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, margin=dict(t=30))
        st.plotly_chart(fig, width='stretch')

def render_geographic_analysis(df: pd.DataFrame, airports: pd.DataFrame):
    """Display destination rankings and movement type analysis."""
    st.divider()
    st.subheader("🌍 Bestemmingsinzichten")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("*15 Bestemmingen met meeste vertragingen **")
        perf = df.groupby("Org/Des")["delay_positive"].agg(["mean", "count"]).reset_index()
        perf = perf.merge(airports, left_on="Org/Des", right_on="IATA", how="left")
        perf["Label"] = perf["Name"].fillna(perf["Org/Des"])
        top_dest = perf[perf["count"] >= 30].sort_values("mean", ascending=False).head(15)
        fig = px.bar(
            top_dest,
            x="mean",
            y="Label",
            orientation="h",
            text="count",
            labels={"mean": "Gem. Vertraging (min)", "Label": "Bestemming"}
        )
        fig.update_layout(margin=dict(t=30))
        st.plotly_chart(fig, width='stretch')
        
    with c2:
        st.markdown("**Vertrek vs. Aankomst Vertragingen (LSV)**")
        fig = px.box(
            df,
            x="LSV",
            y="delay_minutes",
            color="LSV",
            points=False,   
            labels={"LSV": "Type (L=Landing, S=Vertrek)"}
        )
        fig.update_layout(yaxis=dict(range=[-20, 60]), showlegend=False, margin=dict(t=30))
        st.plotly_chart(fig, width='stretch')

# --- Main Application ---
def main():
    # Data pipeline
    raw_data = load_flight_data()
    airports = load_airport_metadata()
    processed_df = process_flight_features(raw_data)
    # Sidebar & Filtering
    filtered_df = render_sidebar(processed_df)
    # Content
    render_metrics(filtered_df)
    st.divider()
    render_temporal_analysis(filtered_df)
    render_performance_analysis(filtered_df)
    render_geographic_analysis(filtered_df, airports)
if __name__ == "__main__":
    main()
