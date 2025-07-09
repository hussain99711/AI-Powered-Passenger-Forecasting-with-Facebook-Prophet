
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Set the title of the app
st.title("📈 AI-Powered Passenger Forecasting")

# Load the dataset
url = "https://raw.githubusercontent.com/hussain99711/AI-Powered-Passenger-Forecasting-with-Facebook-Prophet/main/passengers.csv"
df = pd.read_csv(url)

# Combine year and month into a single date column
df['month'] = df['month'].astype(str)
df['date'] = pd.to_datetime(df['month'] + ' ' + df['year'].astype(str), format='%b %Y')
df = df[['date', 'passengers']]
df.columns = ['ds', 'y']  # Prophet expects columns 'ds' and 'y'

# Show data preview
st.subheader("Raw Dataset")
st.write(df.head())

# Train the Prophet model
model = Prophet()
model.fit(df)

# Forecast for 24 future months
future = model.make_future_dataframe(periods=24, freq='M')
forecast = model.predict(future)

# Show forecast data
st.subheader("Forecast Data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot forecast
st.subheader("Forecast Chart")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Plot components
st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)


