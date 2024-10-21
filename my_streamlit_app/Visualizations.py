import streamlit as st
import pandas as pd
import plotly.express as px

def show_visualizations(df):
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
    car_type_counts = df['BodyType'].value_counts()
    fig = px.bar(car_type_counts, x=car_type_counts.index, y=car_type_counts.values, title="Distribution of Car Types")
    st.plotly_chart(fig)

    # Visualization: Price Distribution
    st.subheader("Price Distribution")
    fig2 = px.histogram(df, x='Price', nbins=30, title="Price Distribution", labels={'Price': 'Price in AUD'})
    st.plotly_chart(fig2)

    # Visualization: Fuel Type vs Price
    st.subheader("Fuel Type vs Price")
    fig3 = px.box(df, x='FuelType', y='Price', title="Fuel Type vs Price")
    st.plotly_chart(fig3)

 
