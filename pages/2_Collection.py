import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st



data = pd.read_csv("Electric_Vehicle_Population_Data.csv")

st.title("Electric Vehicle Population Data Report")

st.write(data.head())

st.title("Brief definitions of Data Frame columns")

st.write(data.columns)

st.text("""
    1. **VIN (1-10)**: The Vehicle Identification Number, truncated.
    2. **County**: The county where the vehicle is registered.
    3. **City**: The city where the vehicle is registered.
    4. **State**: The state where the vehicle is registered.
    5. **Postal Code**: The postal code where the vehicle is registered.
    6. **Model Year**: The year the vehicle model was released.
    7. **Make**: The manufacturer of the vehicle.
    8. **Model**: The specific model of the vehicle.
    9. **Electric Vehicle Type**: Type of electric vehicle (e.g., BEV - Battery Electric Vehicle, PHEV - Plug-in Hybrid Electric Vehicle).
    10. **CAFV Eligibility**: Eligibility for Clean Alternative Fuel Vehicle.
    11. **Electric Range**: The range of the vehicle on electric power alone.
    12. **Base MSRP**: Manufacturer's Suggested Retail Price.
    13. **Legislative District**: The legislative district where the vehicle is registered.
    14. **DOL Vehicle ID**: Department of Licensing vehicle ID.
    15. **Vehicle Location**: The geographic coordinates of the vehicle's registration location.
    16. **Electric Utility**: The utility provider for the vehicle's charging location.
    17. **2020 Census Tract**: Census tract for demographic and geographical data.
""")


st.text("""Let's explore our Data Frame""")

st.write(data.info())

st.write(data.shape)

st.write(data["Make"])

st.text("Some of the important data from our data set")

st.write("Number of cities in our data set: ",len(data["City"].unique()))

st.write("Number of counties in our data set",len(data["County"].unique()))

st.write("Total number of car makes in our data set",len(data["Make"].unique()))

st.write("Total number of car models in our data set",len(data["Model"].unique()))

st.write(data.groupby("Make")["Model Year"].mean().astype(int))




st.text("Summary statistics of Data Frame")
 
st.write(data.describe())




