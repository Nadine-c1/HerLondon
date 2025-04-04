import streamlit as st
import plotly.express as px
import pandas as pd

# Sample dataset
df = pd.DataFrame({
    "lat": [51.5074, 48.8566, 40.7128],
    "lon": [-0.1278, 2.3522, -74.0060]
})

fig = px.scatter_map(df, lat="lat", lon="lon", zoom=3, map_style="open-street-map")
st.plotly_chart(fig)