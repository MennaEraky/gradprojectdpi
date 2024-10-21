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
    col1, col2 = st.columns([2, 2])  # 2 parts for text, 1 part for image

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

    # Load model from Google Drive
    def load_model_from_drive(file_id):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, 'vehicle_price_model.pkl', quiet=False)
        with open('vehicle_price_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

    # Load dataset
    def load_dataset(file):
        return pd.read_csv(file)

    # Clean dataset
    def clean_data(df):
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        return df.dropna()

    # Preprocess input data
    def preprocess_input(input_data, model):
        input_df = pd.DataFrame([input_data])
        return input_df  # Modify according to your preprocessing

    # Visualize correlations
    def visualize_correlations(df):
        st.subheader("Feature Correlation Heatmap")
        corr_matrix = df.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig)

    # Visualize model performance metrics
    def visualize_model_performance():
        models = [
            "LinearRegression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "DecisionTreeRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "SVR",
            "KNeighborsRegressor",
            "MLPRegressor",
            "AdaBoostRegressor",
            "BaggingRegressor",
            "ExtraTreesRegressor"
        ]

        scores = [
            [0.38643429, 0.35310009, 0.36801071],
            [0.38620243, 0.35350286, 0.36843282],
            [0.38620616, 0.35349711, 0.36843277],
            [0.33686675, 0.31415677, 0.32787848],
            [0.62213917, 0.40638212, 0.47242902],
            [0.74799343, 0.70412406, 0.70161075],
            [0.73002938, 0.70887856, 0.70533151],
            [-0.03261018, -0.05532926, -0.05188942],
            [0.64170728, 0.63380643, 0.64356449],
            [-0.38015855, -0.41194531, -0.41229902],
            [0.0021934, -0.43429876, -0.28546934],
            [0.72923447, 0.70932019, 0.67318744],
            [0.74919345, 0.70561132, 0.68979889]
        ]

        mean_scores = [np.mean(score) for score in scores]
        
        # Create DataFrame for plotting
        performance_df = pd.DataFrame({
            'Model': models,
            'Mean CrossVal Score': mean_scores
        })
        
        max_accuracy_model = performance_df.loc[performance_df['Mean CrossVal Score'].idxmax()]

        # Plot the performance
        st.subheader("Model Performance Comparison")
        fig_performance = px.bar(performance_df, x='Model', y='Mean CrossVal Score', 
                                  title='Mean CrossVal Score of Regression Models', 
                                  labels={'Mean CrossVal Score': 'Mean CrossVal Score'},
                                  color='Mean CrossVal Score', 
                                  color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_performance)

        # Display model with largest accuracy
        st.markdown(f"""
            <div style="font-size: 20px; padding: 10px; background-color: #e8f5e9; border: 2px solid #4caf50; border-radius: 5px;">
                <strong>Best Model:</strong> {max_accuracy_model['Model']} with Mean CrossVal Score: {max_accuracy_model['Mean CrossVal Score']:.2f}
            </div>
        """, unsafe_allow_html=True)

    # Load the model only once and store it in session state
    if 'model' not in st.session_state:
        model_file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID for model
        st.session_state.model = load_model_from_drive(model_file_id)

    # Input form for vehicle details
    st.subheader("Input Vehicle Details for Price Prediction")
    input_data = {
        'Year': st.number_input("Year 📅", min_value=1900, max_value=2024, value=2020),
        'UsedOrNew': st.selectbox("Used or New 🏷", ["Used", "New"]),
        'Transmission': st.selectbox("Transmission ⚙", ["Manual", "Automatic"]),
        'Engine': st.number_input("Engine Size (L) 🔧", min_value=0.0, value=2.0, step=0.1),
        'DriveType': st.selectbox("Drive Type 🛣", ["FWD", "RWD", "AWD"]),
        'FuelType': st.selectbox("Fuel Type ⛽", ["Petrol", "Diesel", "Electric", "Hybrid"]),
        'FuelConsumption': st.number_input("Fuel Consumption (L/100km) ⛽", min_value=0.0, value=8.0, step=0.1),
        'Kilometres': st.number_input("Kilometres 🛣", min_value=0, value=50000, step=1000),
        'CylindersinEngine': st.number_input("Cylinders in Engine 🔢", min_value=1, value=4),
        'BodyType': st.selectbox("Body Type 🚙", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"]),
        'Doors': st.selectbox("Number of Doors 🚪", [2, 3, 4, 5])
    }

    # Make prediction and display results
    if st.button("Predict Price"):
        input_df = preprocess_input(input_data, st.session_state.model)
        try:
            prediction = st.session_state.model.predict(input_df)
            st.markdown(f"<h3>Predicted Price: ${prediction[0]:,.2f}</h3>", unsafe_allow_html=True)

            # Display input data and prediction as a table
            input_data['Predicted Price'] = f"${prediction[0]:,.2f}"
            st.dataframe(pd.DataFrame(input_data, index=[0]))

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Load the dataset and visualize correlations
    dataset_file = st.file_uploader("Upload a CSV file containing vehicle data 📂", type="csv")
    if dataset_file is not None:
        df = load_dataset(dataset_file)
        df_cleaned = clean_data(df)

        # Display visualizations
        visualize_correlations(df_cleaned)
        visualize_model_performance()
