import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from src.config import MODEL_COMPARISON_PATH
from src.predict import (
    load_model_comparison,
    load_best_model,
    load_feature_meta,
)

st.title("Modelvergelijking")

if not MODEL_COMPARISON_PATH.exists():
    st.warning("Geen modelresultaten gevonden. Voer eerst het notebook uit.")
    st.stop()

comparison = load_model_comparison()
best_name = comparison.pop("_best_model", None)
comparison.pop("_thresholds", None)

st.subheader("Classificatie-metrics (testset)")
comp_df = pd.DataFrame(comparison).T.sort_values("roc_auc", ascending=False)
comp_df.index.name = "Model"

st.dataframe(
    comp_df.style.format("{:.4f}")
    .highlight_max(axis=0, color="#90EE90")
    .highlight_min(axis=0, color="#FFB6B6"),
    use_container_width=True,
)
if best_name:
    st.success(f"Beste model (op ROC-AUC): **{best_name}**")

st.subheader("Vergelijking per metric")
melted = comp_df.reset_index().melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score",
)
fig = px.bar(
    melted,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    text_auto=".3f",
)
fig.update_layout(height=400, yaxis_range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature-importantie (beste model)")

LABELS = {
    "hour": "Uur",
    "minute": "Minuut",
    "day_of_week": "Dag v/d week",
    "month": "Maand",
    "is_weekend": "Weekend",
    "temperature_2m": "Temperatuur",
    "wind_speed_10m": "Windsnelheid",
    "wind_gusts_10m": "Windstoten",
    "precipitation": "Neerslag",
    "snowfall": "Sneeuwval",
    "cloud_cover": "Bewolking",
    "distance_km": "Afstand (km)",
    "movements_per_hour": "Bewegingen/uur",
    "carrier_enc": "Maatschappij",
    "LSV_enc": "Aankomst/Vertrek",
    "ACT_enc": "Vliegtuigtype",
    "RWC_enc": "Baanconfiguratie",
    "gate_zone_enc": "Gate-zone",
}

meta = load_feature_meta()
clf = load_best_model().named_steps["clf"]

if hasattr(clf, "feature_importances_"):
    values = clf.feature_importances_
    xlabel = "Importantie"
else:
    values = np.abs(clf.coef_[0])
    xlabel = "|Coëfficiënt|"

imp = pd.Series(values, index=meta["feature_cols"]).sort_values()
imp_df = imp.tail(12).reset_index()
imp_df.columns = ["feature", "value"]
imp_df["label"] = imp_df["feature"].map(LABELS).fillna(imp_df["feature"])

fig_imp = go.Figure(
    go.Bar(x=imp_df["value"], y=imp_df["label"], orientation="h")
)
fig_imp.update_layout(height=400, xaxis_title=xlabel, yaxis_title="")
st.plotly_chart(fig_imp, use_container_width=True)
