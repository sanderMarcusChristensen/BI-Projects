# Business Intelligence – F25

Welcome to Group 12’s GitHub repository for the Business Intelligence course (F25) at Cphbusiness.

This repository contains all our deliverables for the mini-projects assigned throughout the course. Each project focuses on a different stage of the data pipeline, including data ingestion, exploration, visualization, and machine learning.

All tasks descriptions can be found inside the `tasks` folder under their respective MP directories.

---

<br>


# Mini Project 1: Data Ingestion and Wrangling
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

### Note
We initially misunderstood the task and used our own datasets located in the /data folder. However, we later created a small project called "World of Harry Potter" where we fetched data from an API. 




<br>

# Mini Project 2: Data Exploration and Visualisation  
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
pip install streamlit pandas matplotlib seaborn scikit-learn
```

### 2. Run the app
Navigate to the folder where streamlit_app.py is located, then run:


("your pc stof"/MP2 - Data Exploration and Visualisation/Code)

```bash or powershell 
python -m streamlit run streamlit_app.py
```

<br>


#  Mini Project 3: Machine Learning for HR Analysis and Prediction

This project explores the use of machine learning to analyze and predict employee behavior based on HR data. The dataset is a synthetic dataset provided by IBM via Kaggle and includes various attributes related to employees’ roles, satisfaction, income, and attrition.
[https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)

We implemented three core machine learning tasks:

1. **Regression** – Predict an employee's monthly income
2. **Classification** – Predict whether an employee is likely to leave the company
3. **Clustering** – Segment employees into similar groups

### Note
we all made a Streamlit App for this, look at Mp2 in this readme to see how to run it

---

<br>

# Answers to Key Questions

###  Which machine learning methods were used, and why?

- **Linear Regression**: Used to predict employees’ monthly income. It’s suitable for continuous values and provides interpretable results.
- **Decision Tree Classification**: Used to predict whether an employee will leave the company (attrition). It works well with mixed feature types and shows feature importance clearly.
- **K-Means Clustering**: Used to group employees with similar characteristics without knowing the target (unsupervised learning). This helps explore patterns in the data.

---

###  How accurate are the models? What do the metrics mean?

- **Regression**:  
  - **R² = 0.934** → The model explains 93.4% of the variance in income.  
  - **RMSE = 1,153.72 DKK** → On average, the prediction error is about 1,154 DKK.  
  - **Conclusion**: Very strong predictive performance for salary estimation.

- **Classification**:  
  - Evaluated using **accuracy**, **precision**, **recall**, and **F1-score** via a decision tree.  
  - The performance was **moderate**, indicating the model can detect key patterns in attrition, though there's room for improvement.

- **Clustering**:  
  - **Silhouette Score = 0.317** using 2 clusters.  
  - **Conclusion**: Indicates that the clusters are distinguishable but not perfectly separated — still useful for segmentation.

---

### Which are the most decisive factors for quitting a job? Why do people quit their job?

Based on our feature selection and the results from the decision tree classifier, the most decisive factors for predicting employee attrition are:

- **OverTime** – Employees who frequently work overtime are more likely to leave.
- **JobSatisfaction** – Low satisfaction with one's job increases the likelihood of quitting.
- **EnvironmentSatisfaction** – Employees who are dissatisfied with their work environment are at higher risk.
- **JobInvolvement** – Low involvement or engagement in one's role contributes to attrition.
- **WorkLifeBalance** – Poor work-life balance is a common reason for leaving.
- **YearsAtCompany** – Employees with shorter tenure are more likely to leave.
- **Age** – Younger employees tend to switch jobs more frequently.

In summary, people often quit due to dissatisfaction with their job conditions, lack of balance between work and personal life, and feeling overworked or underappreciated. These patterns were clearly reflected in the dataset and confirmed by the model's feature importance scores.

--- 
### What could be done for further improvement of the accuracy of the models?

There are several ways we could improve the accuracy and performance of the models:

- **Feature engineering:** Creating new features such as salary progression, time since last promotion, or engagement level over time could give the models more meaningful inputs.

- **Cross-validation:** Implementing k-fold cross-validation would provide a more robust estimate of the model's generalizability.

- **Addressing class imbalance:** In the attrition prediction task, balancing the dataset using techniques such as SMOTE (Synthetic Minority Oversampling Technique) or adjusting class weights could improve classification performance.

- **More granular data:** Including time-based data, such as changes in satisfaction or role over time, could help models understand employee behavior more dynamically.

--- 

### Which work positions and departments are in higher risk of losing employees?

While we did not directly analyze attrition rates by job role or department in this project, the features used in our classification model—such as **OverTime**, **JobSatisfaction**, **EnvironmentSatisfaction**, and **YearsAtCompany**—are known to vary across roles and departments.

Based on these patterns, employees in roles with:

- Frequent **overtime**
- Low **job satisfaction**
- Poor **work-life balance**
- Short **tenure**

are more likely to leave.

In many real-world HR cases, roles in **Sales** or **Customer Support** often have higher turnover due to workload and stress levels. However, to confirm this in our dataset, a deeper breakdown using `JobRole` and `Department` with visualizations or summary statistics would be needed in a future step.


---

### Are employees of different gender paid equally in all departments?

This project did not include a full gender pay gap analysis. However, the dataset contains the necessary columns—such as `Gender`, `MonthlyIncome`, and `Department`—to explore this question.

A brief inspection of the data suggests that there are **only minor differences** in average salaries between genders across departments, but no detailed statistical test was performed.

To accurately determine if a pay gap exists, further analysis would be needed, including:

- Grouping income by `Gender` and `Department`
- Running statistical tests (e.g., t-test or ANOVA)
- Controlling for other variables such as education level, job role, and experience

This would help assess whether any observed differences are statistically significant or due to other factors.


--- 

### Do the family status and the distance from work influence the work-life balance?

The dataset includes both `MaritalStatus` and `DistanceFromHome`, as well as the `WorkLifeBalance` feature, which allows us to explore this relationship.

In our project, we used `DistanceFromHome` and `WorkLifeBalance` in the classification and clustering models. Based on exploratory analysis and domain knowledge, there is reason to believe that:

- **Longer commuting distances** can negatively impact an employee’s work-life balance.
- **Family status** (e.g., being married or having children) likely plays a role in perceived work-life balance, although it was not directly analyzed in our models.

To confirm this relationship, further statistical analysis (e.g., correlation or group comparison) would be needed. However, initial insights suggest that both distance and family situation may influence work-life balance.

---- 

### Does education make people happy (satisfied from the work)?

The dataset includes both `Education` level and `JobSatisfaction`, which allows us to explore the connection between education and work satisfaction.

In this project, we did not find a clear or consistent relationship between higher education levels and higher job satisfaction. Employees with advanced education were **not necessarily more satisfied** with their jobs than those with lower education levels.

This suggests that factors like work environment, involvement, and balance may have a stronger impact on job satisfaction than education alone.

A more detailed analysis using grouped averages or correlation could help confirm this observation.


--- 

### Which were the challenges in the project development?

Several challenges arose during the development of this project:

- **Feature selection:** Choosing the most relevant and informative features for each machine learning task required.

- **Class imbalance:** In the classification task, the number of employees who left the company was much smaller than those who stayed, making the model harder to train accurately. 

- **Data interpretation:** Some relationships, like between education and job satisfaction, were not clearly defined, making interpretation less straightforward. We still dont know what is ment by DailyRate and HourlyRate

--- 


<br>
<br>


# Mini Project 4: BI Chatbot with PDF & Web RAG

This project demonstrates how to build a **Business Intelligence Chatbot** using **RAG (Retrieval-Augmented Generation)**. The chatbot combines a local language model (via Ollama) with your own project documents and web content. It allows you to ask questions and receive context-based answers based on PDFs, TXT files, or URLs.

We use the open source `llama3.2:3b` model locally and combine it with Langchain and Streamlit to create an interactive web interface.

---

### Features

- Automatic loading of `.pdf` and `.txt` files from the `/Data` folder  
- Optional web scraping from user-provided URLs  
- Text splitting and vectorization  
- Retrieval-based Q&A using Langchain's vector store  
- Clean web interface with Streamlit  

---

###  Technologies Used

- **Python 3.12**
- **Langchain** – for embeddings and prompt management  
- **Ollama** – runs the language model locally  
- **Streamlit** – interactive user interface  
- **PyMuPDF** – PDF text extraction  
- **Selenium** – URL content extraction  

---
<br>

## ▶ How to Run the Chatbot

### 1. Install Requirements

Install everything with:

```bash
pip install -r requirements.txt
```

Or install manually: 
```bash
pip install streamlit langchain langchain-community langchain-core langchain-ollama pymupdf
```


### Using the Chatbot
- Add your own .pdf or .txt files to the /Data folder
- open a terminal to where chatbot.py is in path 
- run this: 
```bash
python -m streamlit run Chatbot.py
``` 
- Optionally enter a web URL to include extra context
- Ask any question related to your uploaded content
