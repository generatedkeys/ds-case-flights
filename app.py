import streamlit as st

st.set_page_config(
  page_title="Data Science - Vluchtvertraging Voorspelling",
  layout="wide",
)

overview = st.Page(
  "pages/overview.py",
  title="Overzicht",
)

data_exploration = st.Page(
  "pages/eda.py",
  title="Exploratieve Data Analyse",
)

prediction_model = st.Page(
  "pages/prediction_model.py",
  title="Voorspellingsmodel",
)

insights = st.Page(
  "pages/insights.py",
  title="Inzichten",
)

pg = st.navigation([overview, data_exploration, prediction_model, insights])
pg.run()