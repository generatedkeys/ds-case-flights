import streamlit as st

st.set_page_config(
  page_title="Data Science - Flight Delay Prediction",
  layout="wide",
)

overview = st.Page(
  "pages/overview.py",
  title="Overview",
)

data_exploration = st.Page(
  "pages/eda.py",
  title="Exploratory Data Analysis",
)

prediction_model = st.Page(
  "pages/prediction_model.py",
  title="Prediction Model",
)

insights = st.Page(
  "pages/insights.py",
  title="Insights",
)

# Add other pages to navigation
pg = st.navigation([overview, data_exploration, prediction_model, insights])
pg.run()