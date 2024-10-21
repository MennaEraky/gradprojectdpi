import streamlit as st
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="Australian Vehicle Prices",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Australian Vehicle Prices")

# Custom CSS for layout
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: #e4f0ff;
    }

    /* Container for the image and the text */
    .container {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
    }

    /* Image on the left */
    .image {
        flex: 1;
        margin-right: 20px;
    }

    /* Text content on the right */
    .text-content {
        flex: 2;
        margin-left: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Displaying the content in a custom HTML container
st.markdown(
    """
    <div class="container">
        <div class="image">
            <img src="https://github.com/MennaEraky/gradprojectdpi/blob/main/porsche-911-sally-cars-1.jpg?raw=true" style="width: 100%; height: auto;">
        </div>
        <div class="text-content">
            <h2>Australian Vehicle Prices</h2>
            <p>A comprehensive dataset for exploring the car market in Australia.</p>
            <h3>About Dataset</h3>
            <p>This dataset contains the latest information on car prices in Australia for the year 2023. It covers various brands, models, types, and features of cars sold in the Australian market. It provides useful insights into the trends and factors influencing the car prices in Australia. The dataset includes information such as brand, year, model, car/suv, title, used/new, transmission, engine, drive type, fuel type, fuel consumption, kilometres, colour (exterior/interior), location, cylinders in engine, body type, doors, seats, and price. The dataset has over 16,000 records of car listings from various online platforms in Australia.</p>

            <h3>Key Features:</h3>
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

            <h3>Potential Use Cases:</h3>
            <ul>
                <li><strong>Price prediction:</strong> Predict the price of a car based on its features and location using machine learning models.</li>
                <li><strong>Market analysis:</strong> Explore the market trends and demand for different types of cars in Australia using descriptive statistics and visualization techniques.</li>
                <li><strong>Feature analysis:</strong> Identify the most important features that affect the car prices and how they vary across different brands, models, and locations using correlation and regression analysis.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Feedback file location
feedback_file = "feedback.csv"

# Load existing feedback if the file exists
if os.path.isfile(feedback_file):
    feedback_df = pd.read_csv(feedback_file)
else:
    feedback_df = pd.DataFrame(columns=["Name", "Comments"])  # Create an empty DataFrame if file doesn't exist

# Feedback form
with st.form(key="feedback_form"):
    name = st.text_input("Your Name")
    comments = st.text_area("Your comments or suggestions about our web app and dataset", height=100)
    submit_button = st.form_submit_button("Submit Feedback")

    if submit_button:
        # Create a new feedback entry
        feedback_data = {
            "Name": name,
            "Comments": comments,
        }
        new_feedback_df = pd.DataFrame([feedback_data])

        # Append new feedback to the existing DataFrame
        feedback_df = pd.concat([feedback_df, new_feedback_df], ignore_index=True)

        # Save the updated feedback DataFrame to the local CSV file
        feedback_df.to_csv(feedback_file, index=False)

        st.success("Thank you for your feedback! We appreciate your input.")
