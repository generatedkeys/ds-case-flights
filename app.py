import streamlit as st

st.set_page_config(
    page_title="ZRH Vluchtvertraging Dashboard",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page("pages/overview.py", title="Overzicht"),
        st.Page("pages/eda.py", title="Data-analyse"),
        st.Page("pages/day_simulation.py", title="Dagsimulatie"),
        st.Page("pages/prediction_model.py", title="Modelvergelijking"),
        st.Page("pages/simulator.py", title="Simulator"),
    ]
)
pg.run()
