import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Australian Vehicle Prices",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualizations", "Model"])

# Main content based on the selected page
if page == "Home":
    st.title("Australian Vehicle Prices")
    st.image("porsche-911-sally-cars-1.jpg", use_column_width=True)

    st.markdown(
        """
        ## Overview
        A comprehensive dataset for exploring the car market in Australia.

        ### About Dataset
        **Description**:
        This dataset contains the latest information on car prices in Australia for the year 2023. It covers various brands, models, types, and features of cars sold in the Australian market. It provides useful insights into the trends and factors influencing the car prices in Australia. The dataset includes information such as brand, year, model, car/suv, title, used/new, transmission, engine, drive type, fuel type, fuel consumption, kilometres, colour (exterior/interior), location, cylinders in engine, body type, doors, seats, and price. The dataset has over 16,000 records of car listings from various online platforms in Australia.

        ### Key Features
        - **Brand**: Name of the car manufacturer
        - **Year**: Year of manufacture or release
        - **Model**: Name or code of the car model
        - **Car/Suv**: Type of the car (car or suv)
        - **Title**: Title or description of the car
        - **UsedOrNew**: Condition of the car (used or new)
        - **Transmission**: Type of transmission (manual or automatic)
        - **Engine**: Engine capacity or power (in litres or kilowatts)
        - **DriveType**: Type of drive (front-wheel, rear-wheel, or all-wheel)
        - **FuelType**: Type of fuel (petrol, diesel, hybrid, or electric)
        - **FuelConsumption**: Fuel consumption rate (in litres per 100 km)
        - **Kilometres**: Distance travelled by the car (in kilometres)
        - **ColourExtInt**: Colour of the car (exterior and interior)
        - **Location**: Location of the car (city and state)
        - **CylindersinEngine**: Number of cylinders in the engine
        - **BodyType**: Shape or style of the car body (sedan, hatchback, coupe, etc.)
        - **Doors**: Number of doors in the car
        - **Seats**: Number of seats in the car
        - **Price**: Price of the car (in Australian dollars)

        ### Potential Use Cases
        - **Price prediction**: Predict the price of a car based on its features and location using machine learning models.
        - **Market analysis**: Explore the market trends and demand for different types of cars in Australia using descriptive statistics and visualization techniques.
        - **Feature analysis**: Identify the most important features that affect car prices and how they vary across different brands, models, and locations using correlation and regression analysis.
        """,
        unsafe_allow_html=True,
    )

elif page == "Visualizations":
    st.title("Visualizations")
    st.write("This page will contain visualizations based on the dataset.")

elif page == "Model":
    st.title("Model")
    st.write("This page will contain the model for predicting car prices.")

