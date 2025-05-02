# Business Intelligence – F25
Welcome to Group 12's Git repository for the Business Intelligence course (F25). This repo contains all our mini-projects for the course.

## Mini Project 1: Data Ingestion and Wrangling
This project was part of the Data Ingestion and Wrangling assignment. Our goal was to collect, load, clean, visualize, and store data from multiple file formats.

### Overview
We worked with data in the following formats:

- CSV
- JSON
- TXT

We created Python functions to:

- Load data into pandas DataFrames
- Clean and preprocess the data (e.g., handle missing values)
- Visualize data using Matplotlib and Seaborn

## Note
We initially misunderstood the task and used our own datasets located in the /data folder. However, we later created a small project called "World of Harry Potter" where we fetched data from an API. 


## Mini Project 2: Data Exploration and Visualisation  
This project is part of the "Data Exploration and Visualisation" assignment.  
Our goal was to load, clean, explore, and visualize wine quality data using Python and Streamlit.

--- 

### Overview  
We worked with red and white wine datasets from Excel files.  
The project includes:

- Data loading and cleaning
- Feature exploration and correlation analysis
- Outlier removal
- Dimensionality reduction using PCA
- Visualization of differences in wine types
- An interactive web app using **Streamlit**

--- 

### Technologies Used

- **Python 3.12**
- **pandas** – Data manipulation  
- **matplotlib / seaborn** – Charts and plots  
- **scikit-learn** – PCA and scaling  
- **streamlit** – Web app for data interaction  

--- 

### What We Explored

- Which wine type (red/white) has better average quality?
- How alcohol and sugar levels relate to quality
- Top features that influence wine ratings
- Visual patterns between acidity (pH), density, and quality
- PCA transformation for dimensionality reduction



## ▶ How to Run the Streamlit App

### 1. Install requirements  
- Use: bash or powershell termainal

Open a terminal and run:

``` 
run : pip install streamlit pandas matplotlib seaborn scikit-learn
```

### 2. Run the app
Navigate to the folder where streamlit_app.py is located, then run:

```bash or powershell 
run : python -m streamlit run streamlit_app.py
```
