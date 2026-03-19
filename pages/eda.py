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
    """Load and clean airport names, ICAO codes, and coordinates."""
    df = pd.read_csv(DATA_DIR / "airports-extended-clean.csv", sep=";")
    df["Latitude"] = pd.to_numeric(df["Latitude"].astype(str).str.replace(",", "."), errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"].astype(str).str.replace(",", "."), errors="coerce")
    return df[["ICAO", "Name", "Latitude", "Longitude"]].dropna(subset=["ICAO"]).drop_duplicates(subset=["ICAO"])

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
        carrier=df["FLT"].str.extract(r"^([A-Z0-9]{2,3})", expand=False), # Carrier
        movement_type=df["LSV"].map({"L": "Arrival", "S": "Departure"}) # English data values
    )
    return df


def build_destination_summary(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Aggregate destination performance and enrich it with airport metadata."""
    perf = df.groupby("Org/Des")["delay_minutes"].agg(
        mean="mean",
        count="count",
        p50=lambda s: s.quantile(0.50),
        p75=lambda s: s.quantile(0.75),
        p90=lambda s: s.quantile(0.90),
    ).reset_index()
    perf["delay_positive"] = perf["mean"].clip(lower=0)
    perf = perf.merge(airports, left_on="Org/Des", right_on="ICAO", how="left")
    perf["Label"] = perf["Name"].fillna(perf["Org/Des"])
    perf["MapLabel"] = perf["Name"].fillna("Onbekende luchthaven") + " (" + perf["Org/Des"] + ")"
    return perf

# --- UI Components ---
def render_sidebar(df: pd.DataFrame):
    """Render sidebar filters and return filtered dataframe."""
    st.sidebar.header("Filters")
    years = sorted(df["year"].dropna().unique().astype(int))
    selected_years = st.sidebar.multiselect("Selecteer Jaren", years, default=years)
    
    movement_map = {"Arrival": "Aankomst", "Departure": "Vertrek"}
    movement_options = sorted(df["movement_type"].dropna().unique().tolist())
    selected_types = st.sidebar.multiselect(
        "Vluchttype", 
        movement_options, 
        default=movement_options,
        format_func=lambda x: movement_map.get(x, x)
    )
    
    mask = df["year"].isin(selected_years)
    if selected_types:
        mask &= (df["movement_type"].isin(selected_types))
        
    return df[mask].dropna(subset=["delay_minutes"]).copy(), selected_types

def render_metrics(df: pd.DataFrame, selected_types: list):
    """Display key summary metrics."""
    st.markdown(
        """
        <style>
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"] {
            white-space: normal !important;
            overflow-wrap: anywhere;
            line-height: 1.2;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 1.05rem;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    total_flights = len(df)
    unique_dest = df["Org/Des"].nunique()
    
    m1.metric("Totaal aantal vluchten", f"{total_flights:,}") 
    m4.metric("Unieke Bestemmingen", unique_dest)

    if len(selected_types) > 1:
        # Split metrics for Arrival and Departure
        df_arr = df[df["movement_type"] == "Arrival"]
        df_dep = df[df["movement_type"] == "Departure"]
        
        avg_arr = df_arr["delay_minutes"].mean() if not df_arr.empty else 0
        avg_dep = df_dep["delay_minutes"].mean() if not df_dep.empty else 0
        m2.metric("Gem. Vertraging", f"Aankomst: {avg_arr:.1f} / Vertrek: {avg_dep:.1f} min")
        
        pct_arr = (df_arr["delay_minutes"] > 15).mean() * 100 if not df_arr.empty else 0
        pct_dep = (df_dep["delay_minutes"] > 15).mean() * 100 if not df_dep.empty else 0
        m3.metric("Vertraagd (>15m)", f"Aankomst: {pct_arr:.1f}% / Vertrek: {pct_dep:.1f}%")
    elif len(selected_types) == 1:
        avg_delay = df["delay_minutes"].mean()
        pct_delayed = (df["delay_minutes"] > 15).mean() * 100
        m2.metric("Gem. Vertraging", f"{avg_delay:.1f} min")
        m3.metric("Vertraagde vluchten (>15m)", f"{pct_delayed:.1f}%")
    else:
        st.warning("Selecteer ten minste één vluchttype.")

def render_temporal_analysis(df: pd.DataFrame):
    """Display delay distribution and monthly trends."""
    st.subheader("Tijd patronen")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Verdeling van vertragingen")
        fig_hist = px.histogram(
            df[df["delay_minutes"] > 0],
            x="delay_minutes",
            color="movement_type",
            barmode="overlay",
            nbins=80,
            labels={"delay_minutes": "Vertraging (minuten)", "movement_type": "Vluchttype"},
            color_discrete_map={"Arrival": "indianred", "Departure": "royalblue"},
            category_orders={"movement_type": ["Arrival", "Departure"]}
        )
        fig_hist.for_each_trace(lambda t: t.update(name={"Arrival": "Aankomst", "Departure": "Vertrek"}.get(t.name, t.name)))
        fig_hist.update_layout(
            showlegend=True,
            yaxis_title="Aantal vluchten",
            margin=dict(t=30),
            legend=dict(x=1, y=1, xanchor="right")
        )
        st.plotly_chart(fig_hist, width='stretch')
        
    with c2:
        st.markdown("Vluchtvolume & Gemiddelde Vertragingen per Maand")
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


def render_hourly_weekday_heatmap(df: pd.DataFrame):
    """Display average delay by hour of day and weekday."""
    st.markdown("Vertraging per uur × weekdag")
    hourly = (
        df.dropna(subset=["hour", "day_of_week", "delay_minutes"])
        .groupby(["day_of_week", "hour"])["delay_minutes"]
        .mean()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24))
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=hourly.values,
            x=[f"{hour:02d}:00" for hour in hourly.columns],
            y=[DAY_NAMES[day] for day in hourly.index],
            colorscale="YlOrRd",
            colorbar=dict(title="Gem. vertraging (min)"),
            hovertemplate="Dag: %{y}<br>Uur: %{x}<br>Gem. vertraging: %{z:.1f} min<extra></extra>",
        )
    )
    fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), xaxis_title="Uur", yaxis_title="Dag")
    st.plotly_chart(fig, width='stretch')


def render_calendar_heatmap(df: pd.DataFrame):
    """Display a calendar-style daily average delay heatmap."""
    st.markdown("Kalender heatmap per dag")
    daily = (
        df.dropna(subset=["STD", "delay_minutes"])
        .assign(flight_date=lambda x: x["STD"].dt.normalize())
        .groupby("flight_date")
        .agg(avg_delay=("delay_minutes", "mean"), pct_delayed=("delay_minutes", lambda s: (s > 15).mean() * 100))
        .reset_index()
    )
    if daily.empty:
        st.info("Geen dagelijkse data beschikbaar voor de huidige filters.")
        return

    full_dates = pd.DataFrame({"flight_date": pd.date_range(daily["flight_date"].min(), daily["flight_date"].max(), freq="D")})
    daily = full_dates.merge(daily, on="flight_date", how="left")
    daily["weekday_num"] = daily["flight_date"].dt.dayofweek
    daily["weekday_label"] = daily["weekday_num"].map(DAY_NAMES)
    daily["week_start"] = daily["flight_date"] - pd.to_timedelta(daily["weekday_num"], unit="D")
    daily["flight_date_label"] = daily["flight_date"].dt.strftime("%d-%m-%Y")

    weekday_order = [DAY_NAMES[day] for day in range(7)]
    avg_delay = daily.pivot(index="week_start", columns="weekday_label", values="avg_delay").reindex(columns=weekday_order)
    date_labels = daily.pivot(index="week_start", columns="weekday_label", values="flight_date_label").reindex(columns=weekday_order)
    pct_delayed = daily.pivot(index="week_start", columns="weekday_label", values="pct_delayed").reindex(columns=weekday_order)
    date_values = date_labels.fillna("").to_numpy()
    pct_values = pct_delayed.to_numpy()
    customdata = [
        [
            [date_values[row_idx, col_idx], pct_values[row_idx, col_idx]]
            for col_idx in range(len(weekday_order))
        ]
        for row_idx in range(len(avg_delay.index))
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=avg_delay.values,
            x=weekday_order,
            y=avg_delay.index.strftime("%d-%m-%Y"),
            customdata=customdata,
            colorscale="YlOrRd",
            colorbar=dict(title="Gem. vertraging (min)"),
            hovertemplate="Datum: %{customdata[0]}<br>Weekdag: %{x}<br>Gem. vertraging: %{z:.1f} min<br>Vertraagd >15 min: %{customdata[1]:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), xaxis_title="Weekdag", yaxis_title="Week start")
    st.plotly_chart(fig, width='stretch')


def render_additional_temporal_insights(df: pd.DataFrame):
    """Display detailed temporal heatmaps."""
    c1, c2 = st.columns(2)
    with c1:
        render_hourly_weekday_heatmap(df)
    with c2:
        render_calendar_heatmap(df)

def render_performance_analysis(df: pd.DataFrame):
    """Display carrier and aircraft performance rankings."""
    st.divider()
    st.subheader("Luchtvaartmaatschappijen & Vliegtuigen")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("Luchtvaartmaatschappijen met de meeste vluchten")
        top_carriers = (
            df.groupby("carrier")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        fig_carriers = px.bar(
            top_carriers,
            x="count",
            y="carrier",
            orientation="h",
            text="count",
            labels={"count": "Aantal vluchten", "carrier": "Airline"}
        )
        fig_carriers.update_layout(showlegend=False, margin=dict(t=30))
        st.plotly_chart(fig_carriers, width='stretch')

    with c2:
        st.markdown("Vliegtuigtype vs. Vertraging")
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

    with c1:
        st.markdown("Vliegtuigtypes met de meeste vluchten")
        top_aircraft = (
            df.groupby("ACT")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        fig_aircraft = px.bar(
            top_aircraft,
            x="count",
            y="ACT",
            orientation="h",
            text="count",
            labels={"count": "Aantal vluchten", "ACT": "Vliegtuigtype"}
        )
        fig_aircraft.update_layout(showlegend=False, margin=dict(t=30))
        st.plotly_chart(fig_aircraft, width='stretch')

    with c2:
        st.markdown("Luchtvaartmaatschappijen met de meeste vertragingen")
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


def render_geographic_analysis(df: pd.DataFrame, airports: pd.DataFrame):
    """Display destination rankings and movement type analysis."""
    st.divider()
    st.subheader("Bestemmingen")
    perf = build_destination_summary(df, airports)
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("Bestemmingen met meeste vertragingen")
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

        st.markdown("Bestemmingen met meeste vluchten")
        top_flights = perf.sort_values("count", ascending=False).head(15)
        fig_flights = px.bar(
            top_flights,
            x="count",
            y="Label",
            orientation="h",
            text="count",
            labels={"count": "Aantal vluchten", "Label": "Bestemming"}
        )
        fig_flights.update_layout(margin=dict(t=30))
        st.plotly_chart(fig_flights, width='stretch')
        
    with c2:
        st.markdown("Vertrek vs. Aankomst Vertragingen")
        fig = px.box(
            df,
            x="movement_type",
            y="delay_minutes",
            color="movement_type",
            points=False,   
            labels={"movement_type": "Type", "delay_minutes": "Vertraging (min)"},
            color_discrete_map={"Arrival": "indianred", "Departure": "royalblue"},
            category_orders={"movement_type": ["Arrival", "Departure"]}
        )
        fig.update_xaxes(tickvals=["Arrival", "Departure"], ticktext=["Aankomst", "Vertrek"])
        fig.update_layout(yaxis=dict(range=[-20, 60]), showlegend=False, margin=dict(t=30))
        st.plotly_chart(fig, width='stretch')

    st.markdown("Bestemmingskaart")
    map_data = perf.dropna(subset=["Latitude", "Longitude"]).sort_values("count", ascending=False)
    if map_data.empty:
        st.info("Geen bestemmingen met bekende coördinaten beschikbaar voor de huidige filters.")
        return

    fig_map = px.scatter_geo(
        map_data,
        lat="Latitude",
        lon="Longitude",
        size="count",
        color="mean",
        hover_name="MapLabel",
        hover_data={
            "count": ":,.0f",
            "mean": ":.1f",
            "p50": ":.1f",
            "p75": ":.1f",
            "p90": ":.1f",
            "Latitude": False,
            "Longitude": False,
        },
        labels={
            "count": "Aantal vluchten",
            "mean": "Gem. vertraging (min)",
            "p50": "p50",
            "p75": "p75",
            "p90": "p90",
        },
        color_continuous_scale="YlOrRd",
        projection="natural earth",
    )
    fig_map.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    fig_map.update_geos(
        fitbounds="locations",
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#8FA3B8",
        coastlinewidth=0.8,
        showcountries=True,
        countrycolor="#B8C2CC",
        countrywidth=0.7,
        showland=True,
        landcolor="#F6F2E9",
        showocean=True,
        oceancolor="#E7F0F8",
        showlakes=True,
        lakecolor="#E7F0F8",
        bgcolor="#FFFFFF",
        resolution=50,
    )
    fig_map.update_layout(
        margin=dict(t=30, b=0, l=0, r=0),
        coloraxis_colorbar=dict(title="Gem. vertraging (min)"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig_map, width='stretch')

# --- Main Application ---
def main():
    # Data pipeline
    raw_data = load_flight_data()
    airports = load_airport_metadata()
    processed_df = process_flight_features(raw_data)
    # Sidebar & Filtering
    filtered_df, selected_type = render_sidebar(processed_df)
    # Content
    render_metrics(filtered_df, selected_type)
    st.divider()
    render_temporal_analysis(filtered_df)
    render_additional_temporal_insights(filtered_df)
    render_performance_analysis(filtered_df)
    render_geographic_analysis(filtered_df, airports)
if __name__ == "__main__":
    main()
