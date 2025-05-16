# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

st.set_page_config(page_title="MACHINE LEARNING FOR ANALYSIS AND PREDICTION ")

# Sidebar
st.sidebar.title("Project Menu")
page = st.sidebar.radio("Select a task", [
    "1. Load and clean data",
    "2. Income",
    "3. Attrition",
    "4. Clustering",
    "5. Notes"
])


# ------------------------------------------ Page 1 ------------------------------------------
# Page content
if page == "1. Load and clean data":
    st.title("📊 Descriptive Statistics")
    st.markdown("Here you can show summary statistics, distributions, etc.")
    # fx: st.write(df.describe())
    

# ------------------------------------------ Page 2 ------------------------------------------


elif page == "2. Income":
    st.title("📈 Visualizations")
    st.markdown("Show some interesting graphs here")
    # fx: st.pyplot(fig)


# ------------------------------------------ Page 3 ------------------------------------------

elif page == "3. Attrition":
    st.title("Employee Attrition - Supervised Classification")

    # Load dataset
    emp_data = pd.read_csv("../Data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    st.markdown("### Step 1: Load and check data")
    st.write("Missing values per column:")
    st.write(emp_data.isnull().sum())
    st.write("Number of duplicate rows:", emp_data.duplicated().sum())

    st.markdown("### Step 2: Target variable distribution")
    st.write(emp_data['Attrition'].value_counts())
    fig, ax = plt.subplots()
    sns.countplot(x='Attrition', data=emp_data, ax=ax)
    ax.set_title("Attrition Distribution")
    st.pyplot(fig)

    # Convert target variable to numeric
    emp_data['Attrition'] = emp_data['Attrition'].map({'Yes': 1, 'No': 0})
    y = emp_data['Attrition']

    # Selected features for models
    selected_features = [
        'OverTime', 'Age', 'YearsInCurrentRole', 'YearsAtCompany',
        'JobInvolvement', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
    ]
    X = emp_data[selected_features]
    X['OverTime'] = X['OverTime'].map({'Yes': 1, 'No': 0})

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    st.markdown("### First Decision Tree Model (Baseline)")
    clf_baseline = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test)

    st.subheader("Confusion Matrix (Baseline)")
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    st.dataframe(pd.DataFrame(cm_baseline, columns=["Pred. Stay", "Pred. Leave"], index=["Actual Stay", "Actual Leave"]))

    st.subheader("Classification Report (Baseline)")
    report_baseline = classification_report(y_test, y_pred_baseline, output_dict=True)
    st.dataframe(pd.DataFrame(report_baseline).transpose().round(2))

    st.markdown("### Second Decision Tree Model (Improved with max_depth=4)")
    clf_improved = DecisionTreeClassifier(random_state=42, max_depth=4, class_weight='balanced')
    clf_improved.fit(X_train, y_train)
    y_pred_improved = clf_improved.predict(X_test)

    st.subheader("Confusion Matrix (Improved)")
    cm_improved = confusion_matrix(y_test, y_pred_improved)
    st.dataframe(pd.DataFrame(cm_improved, columns=["Pred. Stay", "Pred. Leave"], index=["Actual Stay", "Actual Leave"]))

    st.subheader("Classification Report (Improved)")
    report_improved = classification_report(y_test, y_pred_improved, output_dict=True)
    st.dataframe(pd.DataFrame(report_improved).transpose().round(2))

    st.markdown("### Feature Importance via ANOVA F-test")
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_train, y_train)

    scores = pd.DataFrame({
        'Feature': X_train.columns,
        'Score': selector.scores_
    }).sort_values(by='Score', ascending=False)

    st.dataframe(scores)

    st.markdown("---")
    st.markdown("### 🔍 Model Comparison: Baseline vs Improved")

    st.markdown("""
    **Precision (class 1 = 'left the company'):**  
    - Baseline: {:.2f} — Only {:.0%} of predicted leavers actually left.  
    - Improved: {:.2f} — Improved precision means fewer false positives.

    **Recall (class 1):**  
    - Baseline: {:.2f} — Found only {:.0%} of actual leavers.  
    - Improved: {:.2f} — Now catches over half of those who truly left.

    **F1-score (class 1):**  
    - Baseline: {:.2f} — Low overall effectiveness detecting attrition.  
    - Improved: {:.2f} — Better balance between precision and recall.

    **Accuracy:**  
    - Baseline: {:.2f}  
    - Improved: {:.2f}  
    Note: Accuracy can be misleading due to class imbalance.
    """.format(
        report_baseline['1']['precision'], report_baseline['1']['precision'],
        report_improved['1']['precision'], 
        report_baseline['1']['recall'], report_baseline['1']['recall'],
        report_improved['1']['recall'],
        report_baseline['1']['f1-score'], report_improved['1']['f1-score'],
        report_baseline['accuracy'], report_improved['accuracy']
    ))


# ------------------------------------------ Page 4 ------------------------------------------


elif page == "4. Clustering":
    st.title("🔍 Clustering & PCA")
    st.markdown("Show PCA plots or clustering results.")
    

# ------------------------------------------ Page 5 ------------------------------------------


elif page == "5. Notes":
    st.title("Skal vi også skrive github noterne her????")
    st.markdown("")
