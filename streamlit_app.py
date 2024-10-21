import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_url = "path_to_your_file/Australian Vehicle Prices.csv"  # Update with your actual file path
df = pd.read_csv(data_url)

# Streamlit App Title
st.title("Australian Vehicle Prices Analysis")
st.write("This dataset tells you about car prices in Australia in 2023. It has data on different kinds of cars and their features.")

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
price_by_year = df.groupby('Year')['Title'].count()  # Adjust as needed
st.line_chart(price_by_year)

# Visualization: Engine Types
st.subheader("Distribution of Engine Types")
engine_type_counts = df['Engine'].value_counts().head(10)  # Show top 10
st.bar_chart(engine_type_counts)

# Filter Options
st.sidebar.header("Filter Options")
car_brand = st.sidebar.selectbox("Select Car Brand", options=df['Brand'].unique())
filtered_data = df[df['Brand'] == car_brand]
st.dataframe(filtered_data)

