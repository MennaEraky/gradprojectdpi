import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

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
    st.write(f"The model with the highest accuracy is: **{max_accuracy_model['Model']}** with a score of **{max_accuracy_model['Mean CrossVal Score']:.2f}**")

# Model prediction page content
if page == "Model":
    st.title("🤖 Model")
    st.write("Use the form below to predict vehicle prices based on selected features.")
    
    # Load model
    model_id = 'your_google_drive_model_id_here'  # Replace with your Google Drive file ID
    model = load_model_from_drive(model_id)

    # User inputs
    st.sidebar.header("Input Features")
    
    year = st.sidebar.number_input("Year", min_value=1990, max_value=2023, value=2020)
    used_or_new = st.sidebar.selectbox("Condition", ["New", "Used"])
    transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
    engine = st.sidebar.number_input("Engine Size (in litres)", min_value=0.1, value=2.0)
    drive_type = st.sidebar.selectbox("Drive Type", ["Front", "Rear", "All-Wheel"])
    fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
    fuel_consumption = st.sidebar.number_input("Fuel Consumption (L/100 km)", min_value=0.1, value=8.0)
    kilometres = st.sidebar.number_input("Kilometres Driven", min_value=0, value=50000)
    cylinders_in_engine = st.sidebar.number_input("Cylinders in Engine", min_value=1, value=4)
    body_type = st.sidebar.selectbox("Body Type", ["Sedan", "Hatchback", "SUV", "Coupe", "Wagon", "Van"])
    doors = st.sidebar.number_input("Number of Doors", min_value=2, max_value=5, value=4)
    
    input_data = {
        "Year": year,
        "UsedOrNew": used_or_new,
        "Transmission": transmission,
        "Engine": engine,
        "DriveType": drive_type,
        "FuelType": fuel_type,
        "FuelConsumption": fuel_consumption,
        "Kilometres": kilometres,
        "CylindersinEngine": cylinders_in_engine,
        "BodyType": body_type,
        "Doors": doors
    }
    
    # Prediction button
    if st.sidebar.button("Predict Price"):
        preprocessed_input = preprocess_input(input_data, model)
        predicted_price = model.predict(preprocessed_input)
        st.success(f"The predicted price for the vehicle is: **${predicted_price[0]:,.2f}**")
    
    # Visualize model performance metrics
    visualize_model_performance()
