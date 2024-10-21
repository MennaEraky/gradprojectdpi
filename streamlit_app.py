import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Australian Vehicle Prices",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS for larger font sizes
st.markdown(
    """
    <style>
        body {
            font-size: 1.5em;  /* Adjust the base font size */
        }
        h1 {
            font-size: 2.5em;  /* Title size */
        }
        h2 {
            font-size: 2em;    /* Section header size */
        }
        p {
            font-size: 1.25em; /* Paragraph text size */
        }
        ul {
            font-size: 1.25em; /* List item text size */
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

# Function to download and load the model using gdown
def load_model_from_drive(file_id):
    output = 'vehicle_price_model.pkl'
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        with open(output, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Preprocess the input data
def preprocess_input(data, model):
    input_df = pd.DataFrame(data, index=[0])  # Create DataFrame with an index
    # One-Hot Encoding for categorical features based on the training model's features
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Reindex to ensure it matches the model's expected input
    model_features = model.feature_names_in_  # Get the features used during training
    input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)  # Fill missing columns with 0
    return input_df_encoded

# Load the dataset from Google Drive
def load_dataset(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Data cleaning and preprocessing function
def clean_data(df):
    # Replace certain values with NaN
    df.replace(['POA', '-', '- / -'], np.nan, inplace=True)
    
    # Convert relevant columns to numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Kilometres'] = pd.to_numeric(df['Kilometres'], errors='coerce')
    
    # Extract numeric values from string columns
    df['FuelConsumption'] = df['FuelConsumption'].str.extract(r'(\d+\.\d+)').astype(float)
    df['Doors'] = df['Doors'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['Seats'] = df['Seats'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['CylindersinEngine'] = df['CylindersinEngine'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['Engine'] = df['Engine'].str.extract(r'(\d+)').fillna(0).astype(int)

    # Fill NaN values for specific columns
    df[['Kilometres', 'FuelConsumption']] = df[['Kilometres', 'FuelConsumption']].fillna(df[['Kilometres', 'FuelConsumption']].median())
    df.dropna(subset=['Year', 'Price'], inplace=True)
    
    # Drop unnecessary columns
    df.drop(columns=['Brand', 'Model', 'Car/Suv', 'Title', 'Location', 'ColourExtInt', 'Seats'], inplace=True)

    # Label encoding for categorical features
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    return df

# Create a function to visualize correlations
def visualize_correlations(df):
    # Calculate the correlation matrix
    correlation = df.corr()
    correlation_with_price = correlation['Price']
    
    # Plot correlation
    st.subheader("Correlation with Price")
    st.write(correlation_with_price)

    # Heatmap of the correlation matrix
    fig = px.imshow(correlation, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig)

# Create additional visualizations
def additional_visualizations(df):
    st.subheader("Price vs Engine Size")
    fig_engine = px.scatter(df, x='Engine', y='Price', title='Price vs Engine Size', 
                             labels={'Engine': 'Engine Size (L)', 'Price': 'Price'},
                             trendline='ols')
    st.plotly_chart(fig_engine)

    st.subheader("Price vs Number of Cylinders")
    fig_cylinders = px.box(df, x='CylindersinEngine', y='Price', 
                            title='Price Distribution by Number of Cylinders',
                            labels={'CylindersinEngine': 'Cylinders in Engine', 'Price': 'Price'})
    st.plotly_chart(fig_cylinders)

    st.subheader("Price vs Fuel Consumption")
    fig_fuel = px.scatter(df, x='FuelConsumption', y='Price', title='Price vs Fuel Consumption',
                          labels={'FuelConsumption': 'Fuel Consumption (L/100 km)', 'Price': 'Price'},
                          trendline='ols')
    st.plotly_chart(fig_fuel)

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

# Main Streamlit app
def main():
    st.title("🚗 Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price.")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year 📅", min_value=1900, max_value=2024, value=2020)
        used_or_new = st.selectbox("Used or New 🚙", ["Used", "New"])
        transmission = st.selectbox("Transmission 🚦", ["Automatic", "Manual"])
        engine = st.number_input("Engine Size (L) 🔧", min_value=0.0, max_value=10.0, value=2.0)
    
    with col2:
        drive_type = st.selectbox("Drive Type 🚗", ["FWD", "RWD", "AWD"])
        fuel_type = st.selectbox("Fuel Type ⛽", ["Petrol", "Diesel", "Electric", "Hybrid"])
        fuel_consumption = st.number_input("Fuel Consumption (L/100km) 📊", min_value=0.0, max_value=50.0, value=7.5)
        kilometres = st.number_input("Kilometres Driven (km) 🛣️", min_value=0, max_value=500000, value=50000)
        cylinders = st.number_input("Cylinders in Engine 🔩", min_value=0, max_value=16, value=4)
        body_type = st.selectbox("Body Type 🚘", ["Sedan", "Hatchback", "SUV", "Coupe", "Convertible"])
        doors = st.number_input("Number of Doors 🚪", min_value=2, max_value=5, value=4)

    if st.button("Predict Price 💰"):
        # Load the model
        model_file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Replace with your model file ID
        model = load_model_from_drive(model_file_id)

        if model is not None:
            # Prepare input data
            input_data = {
                'Year': year,
                'UsedOrNew': used_or_new,
                'Transmission': transmission,
                'Engine': engine,
                'DriveType': drive_type,
                'FuelType': fuel_type,
                'FuelConsumption': fuel_consumption,
                'Kilometres': kilometres,
                'CylindersinEngine': cylinders,
                'BodyType': body_type,
                'Doors': doors
            }

            # Preprocess input and make prediction
            input_df = preprocess_input(input_data, model)
            prediction = model.predict(input_df)
            st.success(f"The predicted price for the vehicle is: ${prediction[0]:,.2f}")

    # Load and clean the dataset
    dataset_file = "https://drive.google.com/uc?id=1BMO9pcLUsx970KDTw1kHNkXg2ghGJVBs"  # Replace with your dataset file ID
    df = load_dataset(dataset_file)

    if df is not None:
        cleaned_df = clean_data(df)

        # Visualize correlations and additional insights
        visualize_correlations(cleaned_df)
        additional_visualizations(cleaned_df)
        visualize_model_performance()

if __name__ == "__main__":
    main()
