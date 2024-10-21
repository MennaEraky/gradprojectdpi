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

    # Custom CSS for layout
    st.markdown(
        """
        <style>
        .container {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: flex-start;
        }
        .text-content {
            flex: 3;
            margin-right: 20px;
        }
        .image {
            flex: 2;
            margin-left: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="container">
            <div class="text-content">
                <h2>Overview</h2>
                <p>A comprehensive dataset for exploring the car market in Australia.</p>
                
                <h3>About Dataset</h3>
                <p><strong>Description:</strong> This dataset contains the latest information on car prices in Australia for the year 2023. It covers various brands, models, types, and features of cars sold in the Australian market. It provides useful insights into the trends and factors influencing the car prices in Australia. The dataset includes information such as brand, year, model, car/suv, title, used/new, transmission, engine, drive type, fuel type, fuel consumption, kilometres, colour (exterior/interior), location, cylinders in engine, body type, doors, seats, and price. The dataset has over 16,000 records of car listings from various online platforms in Australia.</p>

                <h3>Key Features</h3>
                <ul>
                    <li><strong>Brand:</strong> Name of the car manufacturer</li>
                    <li><strong>Year:</strong> Year of manufacture or release</li>
                    <li><strong>Model:</strong> Name or code of the car model</li>
                    <li><strong>Car/Suv:</strong> Type of the car (car or suv)</li>
                    <li><strong>Title:</strong> Title or description of the car</li>
                    <li><strong>UsedOrNew:</strong> Condition of the car (used or new)</li>
                    <li><strong>Transmission:</strong> Type of transmission (manual or automatic)</li>
                    <li><strong>Engine:</strong> Engine capacity or power (in litres or kilowatts)</li>
                    <li><strong>DriveType:</strong> Type of drive (front-wheel, rear-wheel, or all-wheel)</li>
                    <li><strong>FuelType:</strong> Type of fuel (petrol, diesel, hybrid, or electric)</li>
                    <li><strong>FuelConsumption:</strong> Fuel consumption rate (in litres per 100 km)</li>
                    <li><strong>Kilometres:</strong> Distance travelled by the car (in kilometres)</li>
                    <li><strong>ColourExtInt:</strong> Colour of the car (exterior and interior)</li>
                    <li><strong>Location:</strong> Location of the car (city and state)</li>
                    <li><strong>CylindersinEngine:</strong> Number of cylinders in the engine</li>
                    <li><strong>BodyType:</strong> Shape or style of the car body (sedan, hatchback, coupe, etc.)</li>
                    <li><strong>Doors:</strong> Number of doors in the car</li>
                    <li><strong>Seats:</strong> Number of seats in the car</li>
                    <li><strong>Price:</strong> Price of the car (in Australian dollars)</li>
                </ul>

                <h3>Potential Use Cases</h3>
                <ul>
                    <li><strong>Price prediction:</strong> Predict the price of a car based on its features and location using machine learning models.</li>
                    <li><strong>Market analysis:</strong> Explore the market trends and demand for different types of cars in Australia using descriptive statistics and visualization techniques.</li>
                    <li><strong>Feature analysis:</strong> Identify the most important features that affect car prices and how they vary across different brands, models, and locations using correlation and regression analysis.</li>
                </ul>
            </div>
            <div class="image">
                <img src="https://raw.githubusercontent.com/mohamedseif-10/Graduation-project-DEPI/main/web_app/Background.jpg" style="width: 100%; height: auto; border-radius: 10px;">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Visualizations":
    st.title("Visualizations")
    st.write("This page will contain visualizations based on the dataset.")

elif page == "Model":
    st.title("Model")
    st.write("This page will contain the model for predicting car prices.")
