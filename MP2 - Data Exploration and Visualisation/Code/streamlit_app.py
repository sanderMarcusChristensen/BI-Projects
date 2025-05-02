# streamlit_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots
import plotly.express as px  # optional for interactive 3D

# Load cleaned data
wine_df = pd.read_csv("../Data/wine_df_clean.csv")
pca_df = pd.read_csv("../Data/pca_df.csv")

st.set_page_config(layout="wide")
st.title("🍷 Wine Quality Exploration App")

st.markdown("""
This app allows you to interactively explore the wine dataset.  
Use the sidebar to choose different visualizations.  
""")

# Sidebar for choosing visual
plot_type = st.sidebar.selectbox("Choose a chart", [
    "Sample Data",
    "2D: Quality Count",
    "2D: Alcohol vs Quality",
    "2D: Residual Sugar vs Quality",
    "2D: PCA Scatter",
    "3D: PCA (Plotly)"
])

# Show a sample of the dataset
if plot_type == "Sample Data":
    st.subheader("Sample of Wine Dataset")
    st.write(wine_df.sample(10))

elif plot_type == "2D: Quality Count":
    fig, ax = plt.subplots()
    sns.countplot(data=wine_df, x="quality", hue="type", palette={"Red": "red", "White": "blue"}, ax=ax)
    ax.set_title("Wine Quality Count by Type")
    st.pyplot(fig)

elif plot_type == "2D: Alcohol vs Quality":
    fig, ax = plt.subplots()
    sns.scatterplot(data=wine_df, x="alcohol", y="quality", hue="type", palette={"Red": "red", "White": "blue"}, ax=ax)
    ax.set_title("Alcohol vs Quality")
    st.pyplot(fig)

elif plot_type == "2D: Residual Sugar vs Quality":
    fig, ax = plt.subplots()
    sns.scatterplot(data=wine_df, x="residual sugar", y="quality", hue="type", palette={"Red": "red", "White": "blue"}, ax=ax)
    ax.set_title("Residual Sugar vs Quality")
    st.pyplot(fig)

elif plot_type == "2D: PCA Scatter":
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="type", palette={"Red": "red", "White": "blue"}, ax=ax)
    ax.set_title("2D PCA Scatter Plot")
    st.pyplot(fig)

elif plot_type == "3D: PCA (Plotly)":
    st.subheader("3D PCA Visualization (Interactive)")
    fig = px.scatter_3d(
        pca_df.assign(PC3=0),  # simulate a PC3 axis
        x="PC1", y="PC2", z="PC3",
        color="type",
        symbol="quality",
        title="3D PCA Plot (PC1, PC2, dummy PC3)"
    )
    st.plotly_chart(fig)
