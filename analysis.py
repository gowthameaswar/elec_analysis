import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

@st.cache_data()
def load_data():
    return pd.read_csv('Continent_Consumption_TWH.csv')

@st.cache_data()
def fit_arima_model(data, column):
    model = ARIMA(data[column], order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit

@st.cache_data()
def forecast_future_values(model_fit, years):
    future_years = [2023, 2024, 2025]
    forecast = model_fit.predict(start=len(years), end=len(years) + len(future_years) - 1)
    return forecast

data = load_data()

# Collecting information about the data
st.write("Data Shape:")
st.write(data.shape)

st.write("Data Description:")
st.write(data.describe())

st.write("Data Info:")
st.write(data.info())

# Exploring the features
st.write("Missing Values:")
st.write(data.isnull().sum())

# Univariate data analysis
continents = ['OECD', 'BRICS', 'Europe', 'North America', 'Latin America', 'Asia', 'Pacific', 'Africa', 'Middle-East', 'CIS']
for continent in continents:
    years = data['Year']
    data_values = data[continent]

    # Plot the bar chart
    plt.bar(years, data_values, color='green', edgecolor='white')

    # Add labels and title
    plt.xlabel("Year")
    plt.ylabel('Power consumption')
    plt.title('Bar Plot of ' + continent)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    # Display the plot using Streamlit's `pyplot` command
    st.pyplot(plt)

    # Clear the plot for the next iteration
    plt.clf()

# Sum of values by year
x_axis = data['Year']
y_axis = data.drop('Year', axis=1)
sum_by_year = y_axis.groupby(x_axis).sum()

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(15, 8))
for category in sum_by_year.columns:
    ax.bar(sum_by_year.index, sum_by_year[category], label=category)

# Setting labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Sum')
ax.set_title('Sum of Values by Year')

# Setting legend
ax.legend()

# Rotating x-axis tick labels for better readability
plt.xticks(rotation=45)

# Displaying the plot using Streamlit's `pyplot` command
st.pyplot(plt)

# Forecasting future consumption using ARIMA model
continent_models = {
    'Asia': fit_arima_model(data, 'Asia'),
    'Europe': fit_arima_model(data, 'Europe'),
    'North America': fit_arima_model(data, 'North America'),
    'Latin America': fit_arima_model(data, 'Latin America'),
    'Pacific': fit_arima_model(data, 'Pacific'),
    'Africa': fit_arima_model(data, 'Africa'),
    'Middle-East': fit_arima_model(data, 'Middle-East')
}

future_years = [2023, 2024, 2025]

for continent, model in continent_models.items():
    forecast = forecast_future_values(model, data['Year'])
    plt.plot(data['Year'], data[continent], label='Original Data')
    plt.plot(future_years, forecast, label='Forecasted Data')
    plt.xlabel('Year')
    plt.ylabel(continent)
    plt.title('Future Prediction')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()
