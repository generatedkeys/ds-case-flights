import streamlit as st
import pandas as pd

st.title("Exploratory Data Analysis")
st.write("This page is under construction. Add your EDA code here.")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("data/schedule_airport.csv")
    return data

data = load_data()

# Display the data
st.subheader("Dataset Overview")
st.dataframe(data)

# Basic EDA
st.subheader("Basic Statistics")
st.write(data.describe())

st.subheader("Missing Values")
st.write(data.isnull().sum())