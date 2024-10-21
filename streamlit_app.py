import streamlit as st
import pandas as pd
from my_streamlit_app.Model import *
from my_streamlit_app.Visualizations import *

# Set page configuration
st.set_page_config(
    page_title="Australian Vehicle Prices",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualizations", "Model"])


# Load the dataset
data_url = "Australian Vehicle Prices.csv"  # Update with your actual file path
df = pd.read_csv(data_url)

# Convert 'Price' column to numeric and drop NaN values
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df.dropna(subset=['Price'], inplace=True)

# Main content based on the selected page
if page == "Home":
    st.markdown("<h1>🚗 Australian Vehicle Prices</h1>", unsafe_allow_html=True)

    # Layout for text and image
    col1, col2 = st.columns([2, 2])  # 2 parts for text, 2 parts for image

    with col1:
        st.markdown(
            """
            <h2>📊 Overview</h2>
            <p>A comprehensive dataset for exploring the car market in Australia.</p>
            <h2>ℹ About Dataset</h2>
            <p><strong>Description:</strong> This dataset contains the latest information on car prices in Australia for the year 2023. It covers various brands, models, types, and features of cars sold in the Australian market. It provides useful insights into the trends and factors influencing the car prices in Australia. The dataset includes information such as brand, year, model, car/suv, title, used/new, transmission, engine, drive type, fuel type, fuel consumption, kilometres, colour (exterior/interior), location, cylinders in engine, body type, doors, seats, and price. The dataset has over 16,000 records of car listings from various online platforms in Australia.</p>
            <h2>🔑 Key Features</h2>
            <ul>
                <li><strong>Brand</strong>: 🚗 Name of the car manufacturer</li>
                <li><strong>Year</strong>: 📅 Year of manufacture or release</li>
                <li><strong>Model</strong>: 🏷 Name or code of the car model</li>
                <li><strong>Car/Suv</strong>: 🚙 Type of the car (car or suv)</li>
                <li><strong>Title</strong>: 📝 Title or description of the car</li>
                <li><strong>UsedOrNew</strong>: 🔄 Condition of the car (used or new)</li>
                <li><strong>Transmission</strong>: ⚙ Type of transmission (manual or automatic)</li>
                <li><strong>Engine</strong>: 🛠 Engine capacity or power (in litres or kilowatts)</li>
                <li><strong>DriveType</strong>: 🚘 Type of drive (front-wheel, rear-wheel, or all-wheel)</li>
                <li><strong>FuelType</strong>: ⛽ Type of fuel (petrol, diesel, hybrid, or electric)</li>
                <li><strong>FuelConsumption</strong>: 📊 Fuel consumption rate (in litres per 100 km)</li>
                <li><strong>Kilometres</strong>: 🛣 Distance travelled by the car (in kilometres)</li>
                <li><strong>ColourExtInt</strong>: 🎨 Colour of the car (exterior and interior)</li>
                <li><strong>Location</strong>: 📍 Location of the car (city and state)</li>
                <li><strong>CylindersinEngine</strong>: 🔧 Number of cylinders in the engine</li>
                <li><strong>BodyType</strong>: 🚙 Shape or style of the car body (sedan, hatchback, coupe, etc.)</li>
                <li><strong>Doors</strong>: 🚪 Number of doors in the car</li>
                <li><strong>Seats</strong>: 🪑 Number of seats in the car</li>
                <li><strong>Price</strong>: 💰 Price of the car (in Australian dollars)</li>
            </ul>
            <h2>🚀 Potential Use Cases</h2>
            <ul>
                <li><strong>Price prediction</strong>: Predict the price of a car based on its features and location using machine learning models.</li>
                <li><strong>Market analysis</strong>: Explore the market trends and demand for different types of cars in Australia using descriptive statistics and visualization techniques.</li>
                <li><strong>Feature analysis</strong>: Identify the most important features that affect car prices and how they vary across different brands, models, and locations using correlation and regression analysis.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("https://raw.githubusercontent.com/MennaEraky/gradprojectdpi/main/porsche-911-sally-cars-1.jpg", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    <h2>Kaggle Dataset Link</h2>
    <p>"Dataset: [Australian Vehicle Prices on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices)"</p>


elif page == "Visualizations":
    st.title("📈 Visualizations")
    if __name__ == "__main__":
        mainn()

elif page == "Model":
    st.title("🤖 Model")
    if __name__ == "__main__":
        main()
