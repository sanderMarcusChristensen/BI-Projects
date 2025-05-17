# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

st.set_page_config(page_title="MACHINE LEARNING FOR ANALYSIS AND PREDICTION ")

# Sidebar
st.sidebar.title("Project Menu")
page = st.sidebar.radio("Select a task", [
    "1. Intro to project",
    "2. Income",
    "3. Attrition",
    "4. Clustering",
    "5. Answer to tasks questions"
])


if page == "1. Intro to project":
    import time

    with st.spinner("Loading your intelligent workspace..."):
        time.sleep(2)

    st.balloons()


    st.markdown("""
    <style>
       
        @keyframes slowFadeIn {
            0% { opacity: 0; transform: translateY(40px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .note-card {
            background-color: #f1e9ff;
            border-left: 6px solid #d77eff;
            padding: 20px 30px;
            margin-bottom: 50px;
            margin-top: 25px;
            border-radius: 12px;
            font-size: 17px;
            color: #222;
            box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            animation: slowFadeIn 1.5s ease-out;
        }

        a {
            color: #7e3ff2;
            font-weight: bold;
            text-decoration: none;
        }
    </style>
    """, unsafe_allow_html=True)


    st.title("MP3: MACHINE LEARNING FOR ANALYSIS AND PREDICTION")
    st.subheader("Objective and Problem Statement")

    st.markdown("---")

    st.markdown("""
    **Objective**  
    The objective of this mini project is to provide practice in data analysis and prediction by regression, classification, and clustering algorithms.

    **Problem Statement**  
    Attrition is the rate at which employees leave their job. When attrition reaches high levels, it becomes a concern for the company.  
    Therefore, it is important to find out why employees leave, which factors contribute to such significant decision.

    These and other related questions can be answered by exploration analysis and machine learning using the synthetic dataset provided by IBM to Kaggle:  
    [Click here to view the dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)
    """)

    st.markdown("""
    <div class="note-card">
        <p><b>Note:</b><br>
        As we have already covered data loading and cleaning thoroughly in <i>MP1</i> and <i>MP2</i>, we do not repeat those explanations here.  
        Instead, we apply a clean, reusable approach to loading and preparing data across all tasks.  
        Please refer to previous projects if you need a refresher.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(" ")

    st.image("../Data/picture.png", use_container_width=True, caption="Visualizing intelligent systems and data flow")


# ------------------------------------------ Page 2 ------------------------------------------


elif page == "2. Income":
    st.title(" Monthly Income Prediction (Linear Regression)")

    # 1. Load and prepare data
    emp_data = pd.read_csv("../Data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    features = [
        'Age', 'Education', 'EducationField', 'JobLevel', 'JobRole',
        'TotalWorkingYears', 'StandardHours', 'StockOptionLevel'
    ]
    target = 'MonthlyIncome'

    df_model = emp_data[features + [target]].copy()
    df_model_encoded = pd.get_dummies(df_model, columns=['EducationField', 'JobRole'], drop_first=True)

    # 2. Split data
    X = df_model_encoded.drop(columns=[target])
    y = df_model_encoded[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Train model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 4. Evaluate model
    from sklearn.metrics import mean_squared_error, r2_score
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader(" Model Evaluation")
    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:,.0f} kr")

    # 5. Show sample predictions
    st.subheader(" Eksempler på forudsigelser (reel vs. forudsagt)")
    sample_preds = pd.DataFrame({
        "Faktisk løn": y_test.values[:10],
        "Forudsagt løn": y_pred[:10].round(0)
    }).reset_index(drop=True)
    st.dataframe(sample_preds.style.format("{:,.0f} kr"))

    # 6. Plot: Predicted vs Actual
    st.subheader(" Plot: Faktisk vs. Forudsagt løn")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel("Faktisk løn")
    ax1.set_ylabel("Forudsagt løn")
    ax1.set_title("Faktisk vs. Forudsagt MonthlyIncome")
    st.pyplot(fig1)

    # 7. Plot: Residualer
    st.subheader(" Plot: Forudsigelsesfejl (Residualer)")
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=y_test, y=residuals, ax=ax2)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Faktisk løn")
    ax2.set_ylabel("Fejl (Residual)")
    ax2.set_title("Residual Plot")
    st.pyplot(fig2)



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
    st.title(" Employee Segmentation (Clustering with K-Means)")

    # Load and prepare data
    emp_data = pd.read_csv("../Data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    cluster_features = [
        'Age', 'MonthlyIncome', 'DistanceFromHome',
        'JobLevel', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance'
    ]
    X = emp_data[cluster_features]

    # Scale data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # Evaluate clustering
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_scaled, labels)

    st.subheader(" Cluster Evaluation")
    st.write(f"**Silhouette Score:** {score:.3f}")
    st.markdown("""
    - A silhouette score closer to **1** means better clustering quality.  
    - We tested clusters from 2 to 10, and **2 clusters** gave the highest silhouette score.  
    - This means employees can best be grouped into **2 distinct clusters**.
    """)

    # PCA for 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    st.subheader(" PCA Cluster Plot")
    fig1, ax1 = plt.subplots()
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set2")
    ax1.set_title("K-Means Clustering Visualized via PCA")
    ax1.set_xlabel("PCA Component 1")
    ax1.set_ylabel("PCA Component 2")
    st.pyplot(fig1)

    # Optional cluster stats (e.g., mean values per cluster)
    st.subheader(" Cluster Profile Overview")
    emp_data["Cluster"] = labels
    cluster_summary = emp_data.groupby("Cluster")[cluster_features].mean().round(1)
    st.dataframe(cluster_summary)

    st.markdown("---")
    st.markdown("""
    Clustering gives us insights into natural employee groupings without knowing outcomes (unsupervised).  
    For instance, we might find that one cluster contains mostly new, low-paid employees, while the other includes experienced, high-income staff.
    """)

    

# ------------------------------------------ Page 5 ------------------------------------------


elif page == "5. Answer to tasks questions":
    st.title(" Answers to Key Project Questions")

    st.markdown("""
    ###  Which machine learning methods were used, and why?

    - **Linear Regression**: To predict employees’ monthly income – suitable for continuous data.
    - **Decision Tree Classification**: To predict employee attrition – easy to interpret and handles mixed data types well.
    - **K-Means Clustering**: To group employees based on similarities without predefined labels – useful for segmentation.

    ---
    ###  How accurate are the models? What do the metrics mean?

    **Regression:**
    - **R² = 0.934** → Explains 93.4% of salary variance.
    - **RMSE = 1,154 DKK** → Average prediction error.

    **Classification:**
    - Evaluated with **accuracy**, **precision**, **recall**, and **F1-score**.
    - Result: **Moderate performance**, enough to detect important patterns in attrition.

    **Clustering:**
    - **Silhouette Score = 0.318**
    - Indicates some structure in the data, with 2 clusters being the best choice.

    ---
    ###  Why do people quit their jobs?

    Most decisive factors based on feature importance:

    - **OverTime** – Strongly linked to leaving.
    - **JobSatisfaction**, **EnvironmentSatisfaction**, **JobInvolvement**
    - **WorkLifeBalance**, **YearsAtCompany**, **Age**

     Employees quit due to poor conditions, low satisfaction, high stress, or being early in their career.

    ---
    ###  How can the models be improved?

    - **Feature engineering**: Create new features like salary trends or promotion history.
    - **Cross-validation**: To improve robustness.
    - **Handling class imbalance**: Use SMOTE or weighted classes.
    - **Time-based data**: Add employee journey over time for deeper insight.

    ---
    ###  Which roles or departments are at higher attrition risk?

    Though not analyzed deeply, patterns suggest:

    - High **OverTime**
    - Low **JobSatisfaction**
    - Poor **WorkLifeBalance**
    - Short **YearsAtCompany**

    These are often associated with roles like **Sales** or **Support**, but a dedicated role-based analysis would confirm this.

    ---
    ###  Are genders paid equally across departments?

    Not directly analyzed, but based on raw data:

    - Only **minor differences** observed.
    - Full analysis would require grouping by gender + department and statistical testing (e.g., t-test/ANOVA).

    ---
    ###  Do family status and distance from work affect work-life balance?

    - `MaritalStatus` and `DistanceFromHome` likely influence `WorkLifeBalance`.
    - Longer commutes tend to reduce balance.
    - Married employees might prioritize flexibility.

     Further statistical testing would confirm this.

    ---
    ###  Does education impact job satisfaction?

    - No clear link found between education level and `JobSatisfaction`.
    - Advanced degrees didn’t guarantee more happiness.
    - Suggests that **environmental and personal factors** matter more.

    ---
    ###  What challenges were faced?

    - **Feature selection**: Choosing relevant variables for each model.
    - **Class imbalance**: Attrition labels were unevenly distributed.
    - **Unclear variables**: Fields like `DailyRate`, `HourlyRate` were hard to interpret.
    """)

    st.markdown("---")
   

    st.markdown("**GitHub Repository:** [https://github.com/sanderMarcusChristensen/BI-Projects](https://github.com/sanderMarcusChristensen/BI-Projects)")

