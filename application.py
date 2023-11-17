from datetime import datetime

import streamlit as st
import os
import numpy as np
from src.mlProject.pipeline.prediction import PredictionPipeline

st.set_page_config(
    page_title="Hourly Consumption Prediction",
    page_icon="⏰",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("Hourly Consumption Prediction")

# Sidebar
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose an option", ["Home", "Train Model", "Predict"])

if app_mode == "Home":
    st.write("Welcome to the Hourly Consumption Prediction App!")

if app_mode == "Train Model":
    if st.button("Train Model"):
        os.system("python main.py")
        st.success("Training Successful!")

if app_mode == "Predict":
    st.write("Fill in the select the Sensor to make a prediction:")
    # Define input fields
    sensor = st.selectbox("Select Sensor", {"Omaxe NRI City": "62a9920f75c931.62399458", "Royal Court": "5f718c439c7a78.65267835"})
    Clock  = st.date_input("Select Date", value=datetime.date.today())

    # Create a feature list
    feature_list = [
        Clock, sensor
    ]
    features = np.array(feature_list).reshape(1, -1)

    if st.button("Predict"):
        obj = PredictionPipeline()
        predict = obj.predict(features)
        st.success(f"Prediction: {predict}")
