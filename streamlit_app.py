import streamlit as st

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
    st.markdown("<h1 style='font-size: 2.5em;'>🚗 Australian Vehicle Prices</h1>", unsafe_allow_html=True)

    # Layout for text and image
    col1, col2 = st.columns([2, 2])  # 2 parts for text, 1 part for image

    with col1:
        st.markdown(
            """
            <h2 style='font-size: 2em;'>📊 Overview</h2>
            <p style='font-size: 1.25em;'>A comprehensive dataset for exploring the car market in Australia.</p>

            <h2 style='font-size: 2em;'>ℹ️ About Dataset</h2>
            <p style='font-size: 1.25em;'><strong>Description:</strong> This dataset contains the latest information on car prices in Australia for the year 2023. It covers various brands, models, types, and features of cars sold in the Australian market. It provides useful insights into the trends and factors influencing the car prices in Australia. The dataset includes information such as brand, year, model, car/suv, title, used/new, transmission, engine, drive type, fuel type, fuel consumption, kilometres, colour (exterior/interior), location, cylinders in engine, body type, doors, seats, and price. The dataset has over 16,000 records of car listings from various online platforms in Australia.</p>

            <h2 style='font-size: 2em;'>🔑 Key Features</h2>
            <ul style='font-size: 2em;'>
                <li style='font-size: 1.3em;'><strong>Brand</strong>: 🚗 Name of the car manufacturer</li>
                <li style='font-size: 1.3em;'><strong>Year</strong>: 📅 Year of manufacture or release</li>
                <li style='font-size: 1.3em;'><strong>Model</strong>: 🏷️ Name or code of the car model</li>
                <li style='font-size: 1.3em;'><strong>Car/Suv</strong>: 🚙 Type of the car (car or suv)</li>
                <li style='font-size: 1.3em;'><strong>Title</strong>: 📝 Title or description of the car</li>
                <li style='font-size: 1.3em;'><strong>UsedOrNew</strong>: 🔄 Condition of the car (used or new)</li>
                <li style='font-size: 1.3em;'><strong>Transmission</strong>: ⚙️ Type of transmission (manual or automatic)</li>
                <li style='font-size: 1.3em;'><strong>Engine</strong>: 🛠️ Engine capacity or power (in litres or kilowatts)</li>
                <li style='font-size: 1.3em;'><strong>DriveType</strong>: 🚘 Type of drive (front-wheel, rear-wheel, or all-wheel)</li>
                <li style='font-size: 1.3em;'><strong>FuelType</strong>: ⛽ Type of fuel (petrol, diesel, hybrid, or electric)</li>
                <li style='font-size: 1.3em;'><strong>FuelConsumption</strong>: 📊 Fuel consumption rate (in litres per 100 km)</li>
                <li style='font-size: 1.3em;'><strong>Kilometres</strong>: 🛣️ Distance travelled by the car (in kilometres)</li>
                <li style='font-size: 1.3em;'><strong>ColourExtInt</strong>: 🎨 Colour of the car (exterior and interior)</li>
                <li style='font-size: 1.3em;'><strong>Location</strong>: 📍 Location of the car (city and state)</li>
                <li style='font-size: 1.3em;'><strong>CylindersinEngine</strong>: 🔧 Number of cylinders in the engine</li>
                <li style='font-size: 1.3em;'><strong>BodyType</strong>: 🚙 Shape or style of the car body (sedan, hatchback, coupe, etc.)</li>
                <li style='font-size: 1.3em;'><strong>Doors</strong>: 🚪 Number of doors in the car</li>
                <li style='font-size: 1.3em;'><strong>Seats</strong>: 🪑 Number of seats in the car</li>
                <li style='font-size: 1.3em;'><strong>Price</strong>: 💰 Price of the car (in Australian dollars)</li>
            </ul>
            
            <h2 style='font-size: 2em;'>🚀 Potential Use Cases</h2>
            <ul>
                <li style='font-size: 1.3em;'><strong>Price prediction 💰</strong>: Predict the price of a car based on its features and location using machine learning models.</li>
                <li style='font-size: 1.3em;'><strong>Market analysis 📊</strong>: Explore the market trends and demand for different types of cars in Australia using descriptive statistics and visualization techniques.</li>
                <li style='font-size: 1.3em;'><strong>Feature analysis 📊 </strong>: Identify the most important features that affect car prices and how they vary across different brands, models, and locations using correlation and regression analysis.</li>
            </ul>

            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.image("https://raw.githubusercontent.com/MennaEraky/gradprojectdpi/main/porsche-911-sally-cars-1.jpg", use_column_width=True)

elif page == "Visualizations":
    st.title("📈 Visualizations")
    st.write("This page will contain visualizations based on the dataset.")

elif page == "Model":
    st.title("🤖 Model")
    st.write("This page will contain the model for predicting car prices.")
