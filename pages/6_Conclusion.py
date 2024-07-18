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

# Conclusion
st.write("### Conclusion")
st.write("In conclusion, the data indicates that electric vehicle ranges have been improving over the years. Manufacturers like Tesla lead in providing higher electric ranges, making them a preferred choice for consumers.")
