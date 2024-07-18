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


# Electric range distribution
fig, ax = plt.subplots()
sns.histplot(data['Electric Range'], bins=30, kde=True, ax=ax)
ax.set_title('Electric Range Distribution', fontsize=16)
ax.set_xlabel('Electric Range', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
st.pyplot(fig)

# Electric range trend over years
range_trend = data.groupby('Model Year')['Electric Range'].mean().reset_index()
fig, ax = plt.subplots()
ax.plot(range_trend['Model Year'], range_trend['Electric Range'], marker='o')
ax.set_title('Electric Range Trend Over Years', fontsize=16)
ax.set_xlabel('Model Year', fontsize=14)
ax.set_ylabel('Average Electric Range', fontsize=14)
ax.grid(True)
st.pyplot(fig)

# Boxplot of electric ranges by vehicle make
fig, ax = plt.subplots()
abd = sns.boxplot(x='Make', y='Electric Range', data=data, ax=ax)
ax.set_title('Electric Ranges by Vehicle Make', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
ax.set_xlabel('Make', fontsize=14)
ax.set_ylabel('Electric Range', fontsize=14)
abd.tick_params(labelsize=7)

st.pyplot(fig)

# Boxplot of electric ranges by vehicle make
plt.figure(figsize=(14, 7))
sns.boxplot(x='Make', y='Electric Range', data=data)
plt.title('Electric Ranges by Vehicle Make')
plt.xticks(rotation=70)
plt.show()

fig = px.scatter(data, x='Model Year', y='Electric Range', color='Make', title='Electric Range vs. Model Year')
fig.show()


# Bar plot of average electric range by vehicle make
make_avg_range = data.groupby('Make')['Electric Range'].mean().reset_index()
fig, ax = plt.subplots()
abc = sns.barplot(x='Make', y='Electric Range', data=make_avg_range, ax=ax)
ax.set_title('Average Electric Range by Vehicle Make', fontsize=16)
ax.set_xlabel('Make', fontsize=14)
ax.set_ylabel('Average Electric Range', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
abc.tick_params(labelsize=7)
st.pyplot(fig)

# Bar chart of total number of vehicles by make
make_counts = data['Make'].value_counts().reset_index()
make_counts.columns = ['Make', 'Count']
fig = px.bar(make_counts, x='Make', y='Count', title='Total Number of Vehicles by Make', 
             labels={'Make': 'Make', 'Count': 'Total Count'})
fig.update_layout(xaxis={'categoryorder': 'total descending'}, xaxis_tickfont_size=12)
fig.update_xaxes(title_font=dict(size=16))
fig.update_yaxes(title_font=dict(size=16))
st.plotly_chart(fig)

# Insights and Recommendations
st.write("### Insights and Recommendations")
st.write("1. Most electric vehicles have a range between 20 and 300 miles.")
st.write("2. Tesla vehicles tend to have higher electric ranges compared to other makes.")
