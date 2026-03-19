import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from src.config import PROCESSED_CSV, AIRPORTS_CSV, ZRH_LAT, ZRH_LON

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

st.title("Luchthaven Zürich Vertragingen Dashboard")
st.markdown("""
Dit dashboard biedt een overzicht van de vertragingen op luchthaven Zürich (ZRH) gedurende 2019-2020.

### Gegevensbronnen
- **Vluchten**: Historische vluchtschema's en operatiegegevens van ZRH (2019-2020)
- **Luchthavens**: Gegevens van 200+ internationale luchthavens en hun geografische locaties
- **Weer**: Weerdata van OpenMeteo

### Vertraagdefinitie
Conform de [FAA-regulatie 14 CFR §234.2](https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-234/section-234.2):
- **Vertraagd**: Vertrek of aankomst meer dan 15 minuten later dan gepland
""")
