import streamlit as st
import pandas as pd
from my_streamlit_app.Model import *
from my_streamlit_app.Visualizations import show_visualizations

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

# Add a link to the dataset
st.sidebar.markdown("Dataset: [Australian Vehicle Prices on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices)")

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

elif page == "Visualizations":
    show_visualizations(df)

elif page == "Model":
    st.title("🤖 Model")
    
    st.write("Enter the vehicle details below to predict its price.")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year 📅", min_value=1900, max_value=2024, value=2020, key="year")
        used_or_new = st.selectbox("Used or New 🏷", ["Used", "New"], key="used_or_new")
        transmission = st.selectbox("Transmission ⚙", ["Manual", "Automatic"], key="transmission")
        engine = st.number_input("Engine Size (L) 🔧", min_value=0.0, value=2.0, step=0.1, key="engine")
        drive_type = st.selectbox("Drive Type 🛣", ["FWD", "RWD", "AWD"], key="drive_type")
        fuel_type = st.selectbox("Fuel Type ⛽", ["Petrol", "Diesel", "Electric", "Hybrid"], key="fuel_type")

    with col2:
        fuel_consumption = st.number_input("Fuel Consumption (L/100km) ⛽", min_value=0.0, value=8.0, step=0.1, key="fuel_consumption")
        kilometres = st.number_input("Kilometres 🛣", min_value=0, value=50000, step=1000, key="kilometres")
        cylinders_in_engine = st.number_input("Cylinders in Engine 🔢", min_value=1, value=4, key="cylinders_in_engine")
        body_type = st.selectbox("Body Type 🚙", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"], key="body_type")
        doors = st.selectbox("Number of Doors 🚪", [2, 3, 4, 5], key="doors")

    # Load model only once and store in session state
    if 'model' not in st.session_state:
        model_file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID for model
        st.session_state.model = load_model_from_drive(model_file_id)

    # Make prediction automatically based on inputs
    if st.session_state.model is not None:
        input_data = {
            'Year': year,
            'UsedOrNew': used_or_new,
            'Transmission': transmission,
            'Engine': engine,
            'DriveType': drive_type,
            'FuelType': fuel_type,
            'FuelConsumption': fuel_consumption,
            'Kilometres': kilometres,
            'CylindersinEngine': cylinders_in_engine,
            'BodyType': body_type,
            'Doors': doors
        }
        input_df = preprocess_input(input_data, st.session_state.model)

        try:
            prediction = st.session_state.model.predict(input_df)

            # Styled prediction display
            st.markdown(f"""
                <div style="font-size: 24px; padding: 10px; background-color: #f0f4f8; border: 2px solid #3e9f7d; border-radius: 5px; text-align: center;">
                    <strong>Predicted Price:</strong> ${prediction[0]:,.2f}
                </div>
            """, unsafe_allow_html=True)

            # Displaying input data and prediction as a table
            st.subheader("Input Data and Prediction")
            input_data['Predicted Price'] = f"${prediction[0]:,.2f}"
            input_df_display = pd.DataFrame(input_data, index=[0])
            st.dataframe(input_df_display)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Load the dataset and preprocess it for visualization
    dataset_file = st.file_uploader("Upload a CSV file containing vehicle data 📂", type="csv")
    if dataset_file is not None:
        df = load_dataset(dataset_file)
        if df is not None:
            df_cleaned = clean_data(df)

            # Display visualizations
            visualize_correlations(df_cleaned)
            additional_visualizations(df_cleaned)
            visualize_model_performance()

if __name__ == "__main__":
    main()
