import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the load_json function from the 'scripts' folder
from scripts.Load_Json import load_json

# Load JSON data
df = load_json('data/Json_data.json')

# --- Exploration ---
print("🧾 Preview of data:")
print(df.head())
print("\n📊 Data Info:")
print(df.info())

# --- Clean & Flatten ---
# Expand nested grades
grades_df = df['grades'].apply(pd.Series)
df = df.drop('grades', axis=1).join(grades_df)

# Ensure correct data types
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# --- Anonymization ---
def anonymise(value):
    return hashlib.sha256(value.encode()).hexdigest()[:10]

df['name'] = df['name'].apply(anonymise)
df['email'] = df['email'].apply(anonymise)

# --- Visualizations ---
sns.set(style='whitegrid')

# 1. Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['age'], kde=True, bins=5)
plt.title('Student Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visuals/age_distribution.png')

# 2. Grades by Subject
plt.figure(figsize=(8,5))
sns.boxplot(data=df[['math', 'english', 'science']])
plt.title('Grade Distribution by Subject')
plt.tight_layout()
plt.savefig('visuals/grade_boxplot.png')

# 3. Active Students Count
plt.figure(figsize=(4,3))
sns.countplot(x='active', data=df)
plt.title('Active vs Inactive Students')
plt.tight_layout()
plt.savefig('visuals/active_status.png')

print("\n✅ Process complete! Visuals saved in 'visuals/' folder.")
