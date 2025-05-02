import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---
@st.cache_data
def load_data():
    wine_df = pd.read_csv("../Data/wine_df_clean.csv")
    pca_df = pd.read_csv("../Data/pca_df.csv")
    return wine_df, pca_df

wine_df, pca_df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🍷 Wine App Menu")
page = st.sidebar.radio("Choose a section", ["Home", "Wine Data", "Visualizations", "PCA View", "Wine Facts"])

# --- Page: Home ---
if page == "Home":
    st.title("🍇 Wine Quality Exploration App")
    st.write("""
        Welcome! This simple app helps you explore red and white wine quality data.

        You can:
        - View and explore the cleaned wine dataset
        - Visualize differences between red and white wines
        - See PCA (dimensionality reduction) results
        - Learn interesting facts about wine quality
    """)
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtF38ohg7rOTYLLccfTwvueJS56KIZrr5g0A&s", width=300)

# --- Page: Wine Data ---
elif page == "Wine Data":
    st.title("📊 Wine Dataset Preview")
    st.write("Here's a sample of the cleaned wine dataset:")
    st.dataframe(wine_df.sample(10))

    st.write("Summary Statistics:")
    st.dataframe(wine_df.describe())

# --- Page: Visualizations ---
elif page == "Visualizations":
    st.title("📈 Wine Visual Analysis")

    st.subheader("Wine Quality Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=wine_df, x="quality", hue="type", palette={"Red": "red", "White": "blue"}, ax=ax1)
    ax1.set_title("Quality by Wine Type")
    st.pyplot(fig1)

    st.subheader("Alcohol vs Quality")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=wine_df, x="alcohol", y="quality", hue="type", palette={"Red": "red", "White": "blue"}, ax=ax2)
    ax2.set_title("Alcohol Content and Quality")
    st.pyplot(fig2)

# --- Page: PCA View ---
elif page == "PCA View":
    st.title("🔬 PCA – Wine Feature Projection")

    st.write("This is a 2D projection of the wine features using Principal Component Analysis.")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="type", palette={"Red": "red", "White": "blue"}, ax=ax3)
    ax3.set_title("PCA: PC1 vs PC2")
    st.pyplot(fig3)

    st.write("Random rows from PCA-reduced data:")
    st.dataframe(pca_df.sample(10))

# --- Page: Wine Facts ---
elif page == "Wine Facts":
    st.title("📚 Wine Quality Facts")

    st.markdown("""
    **What affects wine quality?**
    
    1. **Alcohol** – Stronger wines are often rated higher.
    2. **Volatile acidity** – Too much acidity lowers quality.
    3. **Sulphates** – Help preserve and stabilize wine.
    
    **Other factors include:**
    - Grape type and harvest time
    - Region (soil, weather)
    - Aging and fermentation technique
    - Human taste preferences

    **Further Reading:**
    - [Wikipedia: Wine](https://en.wikipedia.org/wiki/Wine)
    - [Wine Tasting Basics (YouTube)](https://www.youtube.com/watch?v=ZzN5ZkERXJw)
    """)

    st.video("https://www.youtube.com/watch?v=ZzN5ZkERXJw")
