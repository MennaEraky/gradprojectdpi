import streamlit as st
import pandas as pd

# Set up the sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualizations", "Model"])

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the dataset into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Home Page
    if page == "Home":
        st.title("🚗 Australian Vehicle Prices")

        # Layout for text and image
        col1, col2 = st.columns([2, 1])  # 2 parts for text, 1 part for image

        with col1:
            st.markdown(
                """
                <h2>📊 Overview</h2>
                <p>A comprehensive dataset for exploring the car market in Australia.</p>
                <h2>ℹ️ About Dataset</h2>
                <p><strong>Description:</strong> This dataset contains the latest information on car prices in Australia for the year 2023...</p>
                <h2>🔑 Key Features</h2>
                <ul>
                    <li><strong>Brand</strong>: 🚗 Name of the car manufacturer</li>
                    <li><strong>Year</strong>: 📅 Year of manufacture or release</li>
                    <li><strong>Model</strong>: 🏷️ Name or code of the car model</li>
                </ul>
                <h2>🚀 Potential Use Cases</h2>
                <ul>
                    <li><strong>Price prediction</strong>: Predict the price of a car based on its features...</li>
                </ul>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.image("https://raw.githubusercontent.com/MennaEraky/gradprojectdpi/main/porsche-911-sally-cars-1.jpg", use_column_width=True)

    # Visualizations Page
    elif page == "Visualizations":
        import visualization  # Import the visualizations module
        visualization.show_visualizations(df)  # Call the function to display visualizations

    # Model Page
    elif page == "Model":
        import model  # Import the model module
        model.show_model(df)  # Call the function to display the model interface

else:
    st.info("Please upload a CSV file to get started.")
