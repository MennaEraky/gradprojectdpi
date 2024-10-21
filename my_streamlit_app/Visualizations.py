import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def show_visualizations(df):
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
    car_type_counts = df['BodyType'].value_counts()
    fig = px.bar(car_type_counts, x=car_type_counts.index, y=car_type_counts.values, title="Distribution of Car Types")
    st.plotly_chart(fig)

    # Visualization: Price Distribution
    st.subheader("Price Distribution")
    fig2 = px.histogram(df, x='Price', nbins=30, title="Price Distribution", labels={'Price': 'Price in AUD'})
    st.plotly_chart(fig2)

    # Visualization: Fuel Type vs Price
    # st.subheader("Fuel Type vs Price")
    # fig3 = px.box(df, x='FuelType', y='Price', title="Fuel Type vs Price")
    # st.plotly_chart(fig3)

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

# Load model from Google Drive
def load_model_from_drive(model_file_id):
    model_file_url = f"https://drive.google.com/uc?id={model_file_id}"
    model_file_path = "model.pkl"
    gdown.download(model_file_url, model_file_path, quiet=False)

    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Main Streamlit app
def mainn():
    # Load the dataset and preprocess it for visualization
    dataset_file = st.file_uploader("Upload a CSV file containing vehicle data 📂", type="csv")
    if dataset_file is not None:
        df = load_dataset(dataset_file)
        if df is not None:
            df_cleaned = clean_data(df)

            # Show visualizations
            show_visualizations(df_cleaned)
            additional_visualizations(df_cleaned)
            visualize_correlations(df_cleaned)
        

            

if __name__ == "__main__":
    mainn()
