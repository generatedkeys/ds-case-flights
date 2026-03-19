import streamlit as st
import pandas as pd

from src.config import BEST_MODEL_PATH
from src.predict import load_all_models, load_feature_meta

st.title("Simulator")
st.markdown("Pas parameters aan en bekijk de kans op vertraging >15 min.")

if not BEST_MODEL_PATH.exists():
    st.warning("Geen getraind model gevonden. Voer eerst het notebook uit.")
    st.stop()


@st.cache_resource
def _load_models():
    return load_all_models()


@st.cache_data
def _load_meta():
    return load_feature_meta()


models = _load_models()
meta = _load_meta()
le = meta["label_encoders"]

selected_model = st.sidebar.selectbox("Model", list(models.keys()))
model = models[selected_model]

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Gepland uur", 0, 23, 12)
    minute = st.slider("Geplande minuut", 0, 59, 0, step=5)
    dagen = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]
    day_of_week = st.selectbox(
        "Dag", list(range(7)), format_func=lambda x: dagen[x]
    )
    maanden = [
        "Jan",
        "Feb",
        "Mrt",
        "Apr",
        "Mei",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Okt",
        "Nov",
        "Dec",
    ]
    month = st.selectbox(
        "Maand", list(range(1, 13)), format_func=lambda x: maanden[x - 1]
    )
    carrier = st.selectbox("Maatschappij", le["carrier"])
    lsv = st.selectbox(
        "Beweging",
        le["LSV"],
        format_func=lambda x: "Aankomst" if x == "L" else "Vertrek",
    )

with col2:
    st.markdown("**Weer**")
    temperature = st.slider("Temperatuur (°C)", -15.0, 40.0, 10.0, step=0.5)
    wind_speed = st.slider("Windsnelheid (m/s)", 0.0, 30.0, 5.0, step=0.5)
    wind_gusts = st.slider("Windstoten (m/s)", 0.0, 50.0, 10.0, step=1.0)
    precipitation = st.slider("Neerslag (mm)", 0.0, 20.0, 0.0, step=0.5)
    snowfall = st.slider("Sneeuwval (cm)", 0.0, 10.0, 0.0, step=0.1)
    cloud_cover = st.slider("Bewolking (%)", 0, 100, 50)

aircraft = st.sidebar.selectbox("Vliegtuigtype", le["ACT"])
rwc = st.sidebar.selectbox("Baanconfiguratie", le["RWC"])
gate_zone = st.sidebar.selectbox("Gate-zone", le["gate_zone"])
distance = st.sidebar.slider("Afstand (km)", 100, 12000, 1000, step=100)
congestion = st.sidebar.slider("Bewegingen/uur", 5, 80, 30)


def _enc(col, val):
    classes = le[col]
    return classes.index(val) if val in classes else 0


row = pd.DataFrame(
    [
        {
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": int(day_of_week >= 5),
            "temperature_2m": temperature,
            "wind_speed_10m": wind_speed,
            "wind_gusts_10m": wind_gusts,
            "precipitation": precipitation,
            "snowfall": snowfall,
            "cloud_cover": cloud_cover,
            "distance_km": distance,
            "movements_per_hour": congestion,
            "carrier_enc": _enc("carrier", carrier),
            "LSV_enc": _enc("LSV", lsv),
            "ACT_enc": _enc("ACT", aircraft),
            "RWC_enc": _enc("RWC", rwc),
            "gate_zone_enc": _enc("gate_zone", gate_zone),
        }
    ]
)[meta["feature_cols"]]

st.divider()
proba = model.predict_proba(row)[0, 1]

c1, c2 = st.columns(2)
c1.metric("Voorspelling", "Vertraagd" if proba >= 0.5 else "Op tijd")
c2.metric("Kans op vertraging", f"{proba:.1%}")

if proba >= 0.7:
    st.error(f"Hoog risico ({proba:.0%})")
elif proba >= 0.5:
    st.warning(f"Matig risico ({proba:.0%})")
else:
    st.success(f"Laag risico ({proba:.0%})")

st.subheader("Alle modellen op dit scenario")
all_p = {n: m.predict_proba(row)[0, 1] for n, m in models.items()}
p_df = pd.DataFrame.from_dict(
    all_p, orient="index", columns=["Kans op vertraging"]
).sort_values("Kans op vertraging", ascending=False)
st.dataframe(
    p_df.style.format("{:.1%}").bar(
        subset=["Kans op vertraging"], color="#ff6b6b", vmin=0, vmax=1
    ),
    use_container_width=True,
)
