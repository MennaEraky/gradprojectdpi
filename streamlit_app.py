import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Australian Vehicle Prices",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS for larger font sizes
st.markdown(
    """
    <style>
        body {
            font-size: 1.2em;  /* Adjust the base font size */
        }
        h1 {
            font-size: 2.5em;  /* Title size */
        }
        h2 {
            font-size: 2em;    /* Section header size */
        }
        p {
            font-size: 1em; /* Paragraph text size */
        }
        ul {
            font-size: 1.6em; /* List item text size */
        }
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualizations", "Model"])

# Add a link to the dataset
st.sidebar.markdown("Dataset: [Australian Vehicle Prices on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices)")

# Load the dataset
# Load the dataset
data_url = "Australian Vehicle Prices.csv"  # Update with your actual file path
df = pd.read_csv(data_url)


# Convert 'Price' column to numeric
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop NaN values from 'Price' column
df.dropna(subset=['Price'], inplace=True)

# Now perform the groupby operation
price_by_year = df.groupby('Year')['Price'].mean()  # Adjusted to show mean price


# Main content based on the selected page
if page == "Home":
    st.markdown("<h1>🚗 Australian Vehicle Prices</h1>", unsafe_allow_html=True)

    # Layout for text and image
    col1, col2 = st.columns([2, 12)  # 2 parts for text, 1 part for image

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

elif page == "Visualizations":
    st.title("📈 Visualizations")
    st.write("This page will contain visualizations based on the dataset.")

    # Show dataset information
    st.write("Dataset Information:")
    st.dataframe(df)

    # Unique Values
    st.write("Unique Values in Columns:")
    for column in df.columns:
        st.write(f"{column}: {df[column].nunique()} unique values")

    # Visualization: Distribution of Car Types
    st.subheader("Distribution of Car Types")
    car_type_counts = df['Car/Suv'].value_counts()
    st.bar_chart(car_type_counts)

    # Visualization: Prices Over the Years
    st.subheader("Car Prices Over the Years")
    price_by_year = df.groupby('Year')['Price'].mean()  # Adjusted to show mean price
    st.line_chart(price_by_year)

    # Visualization: Engine Types
    st.subheader("Distribution of Engine Types")
    engine_type_counts = df['Engine'].value_counts().head(10)  # Show top 10
    st.bar_chart(engine_type_counts)



elif page == "Model":
    st.title("🤖 Model")
    st.write("This page will contain the model for predicting car prices.")
