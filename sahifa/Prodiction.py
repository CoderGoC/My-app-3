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


# Example: predicting electric range based on model year and make
X = data[['Model Year', 'Make']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to numeric
y = data['Electric Range']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')




# Prediction model
future_model_year_input = st.number_input('Enter future model year for prediction', min_value=2024)
future_make_input = st.selectbox('Select make for prediction', data['Make'].unique())
if st.button('Predict Electric Range'):
    future_model_year = pd.DataFrame({'Model Year': [future_model_year_input], 'Make': [future_make_input]})
    future_model_year = pd.get_dummies(future_model_year).reindex(columns=X.columns, fill_value=0)
    future_range = model.predict(future_model_year)
    st.write(f'Predicted Electric Range for a {future_model_year_input} {future_make_input}: {future_range[0]}')
