import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import gdown
import pickle
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

# Function to log transform price
def log_transform_price(df):
    df['Price_log'] = np.log1p(df['Price'])  # Apply log transformation
    return df

# Function to show price distribution before and after log transformation
def show_price_distribution(df):
    st.subheader("Price Distribution Before and After Log Transformation")
    
    # Price Distribution Before Transformation
    fig_before = px.histogram(df, x='Price', nbins=30, title="Price Distribution (Before Log Transformation", labels={'Price': 'Price in AUD'})
    st.plotly_chart(fig_before)

    # Price Distribution After Transformation
    fig_after = px.histogram(df, x='Price_log', nbins=30, title="Price Distribution (After Log Transformation)", labels={'Price_log': 'Log Price'})
    st.plotly_chart(fig_after)

# Visualizations for the main function
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

    # Visualization: Price Distribution (Already included in the log transformation)
    # (This can be commented out or removed if using show_price_distribution)
    # st.subheader("Price Distribution")
    # fig2 = px.histogram(df, x='Price', nbins=30, title="Price Distribution", labels={'Price': 'Price in AUD'})
    # st.plotly_chart(fig2)

# Main Streamlit app
def mainn():
    # Load the dataset and preprocess it for visualization
    dataset_file = st.file_uploader("Upload a CSV file containing vehicle data 📂", type="csv")
    if dataset_file is not None:
        df = load_dataset(dataset_file)
        if df is not None:
            df_cleaned = clean_data(df)

            # Apply log transformation
            df_cleaned = log_transform_price(df_cleaned)

            # Show price distributions
            show_price_distribution(df_cleaned)

            # Show visualizations
            show_visualizations(df_cleaned)
            additional_visualizations(df_cleaned)
            visualize_correlations(df_cleaned)

if __name__ == "__main__":
    mainn()
