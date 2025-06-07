import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# App title
st.title("Weather Anomalies Visualization Tool")

# Sidebar controls
st.sidebar.header("Configuration")
date_range = st.sidebar.date_input(
    "Select date range",
    value=[datetime(2020, 1, 1), datetime(2020, 12, 31)]
)
location = st.sidebar.selectbox(
    "Location",
    ["Global", "North America", "Europe", "Asia", "Custom..."]
)
anomaly_type = st.sidebar.radio(
    "Anomaly Type",
    ["Temperature", "Precipitation", "Wind Speed"]
)

# Load sample data (in a real app, you'd load from API or database)
@st.cache_data
def load_data():
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    data = {
        "date": dates,
        "temperature": np.random.normal(15, 5, len(dates)).cumsum(),
        "precipitation": np.random.gamma(1, 2, len(dates)),
        "wind_speed": np.random.weibull(2, len(dates))
    }
    return pd.DataFrame(data)

df = load_data()

# Calculate anomalies (simple z-score approach)
df[f"{anomaly_type.lower()}_zscore"] = (
    (df[anomaly_type.lower()] - df[anomaly_type.lower()].mean()) 
    / df[anomaly_type.lower()].std()
)
df["anomaly"] = abs(df[f"{anomaly_type.lower()}_zscore"]) > 2

# Main visualization
st.header(f"{anomaly_type} Anomalies")
fig = px.line(
    df,
    x="date",
    y=anomaly_type.lower(),
    color="anomaly",
    color_discrete_map={False: "blue", True: "red"},
    title=f"{anomaly_type} Over Time with Anomalies Highlighted"
)
st.plotly_chart(fig)

# Anomaly statistics
anomaly_count = df["anomaly"].sum()
st.metric(
    label="Number of Anomalies Detected",
    value=anomaly_count,
    delta=f"{anomaly_count/len(df)*100:.1f}% of days"
)

# Raw data view
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(df)