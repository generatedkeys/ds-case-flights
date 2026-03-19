import streamlit as st

st.header("Overzicht")
st.caption("Voorspellen van vluchtvertragingen met machine learning.")
st.subheader("Data gebruikt in deze app")
st.markdown(
	"""
	Deze app gebruikt de dataset **schedule_airport.csv** met vluchtgegevens van
	**Zurich Airport** voor de jaren **2019 en 2020**.
  """
)
st.subheader("**Vluchtvertraging**")
st.markdown(
	"""
	In deze app definiëren we vluchtvertraging als:
  Delay = actual time - scheduled time.
  
  De vertraging wordt uitgedrukt in minuten.

  Een positieve waarde betekent te laat, 0 betekent op tijd en een negatieve waarde betekent vroeger dan gepland.
	"""
)

col1, col2 = st.columns(2)
with col1:
	st.info (
		"""
		**Arrival delay**

		Voor aankomsten geldt: 
		**Arrival delay = actual arrival time - scheduled arrival time**
		"""
	)
with col2:
	st.info(
		"""
		**Departure delay**

		Voor vertrekken geldt: 
		**Departure delay = actual departure time - scheduled departure time**
		"""
	)
