import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Add parent directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the load_json function from the 'Code' folder
from data_loader import load_json

# Load the JSON data using the data_loader script
df = load_json('Data/Json_data.json')

# --- Exploration ---
print("🧾 Preview of data:")
print(df.head())
print("\n📊 Data Info:")
print(df.info())

# --- Clean & Flatten ---
# Expand nested grades into separate columns
# We do not drop these columns yet because we need them for visualizations
grades_df = df[['grades.math', 'grades.english', 'grades.science']]

# Ensure correct data types
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# --- Anonymization ---
def anonymise(value):
    return hashlib.sha256(value.encode()).hexdigest()[:10]

df['name'] = df['name'].apply(anonymise)
df['email'] = df['email'].apply(anonymise)

# --- Visualizations ---
sns.set(style='whitegrid')

# Ensure the 'Visuals' directory exists, if not, create it
if not os.path.exists('Visuals'):
    os.makedirs('Visuals')

# 1. Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['age'], kde=True, bins=5)
plt.title('Student Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Visuals/age_distribution.png')

# 2. Grades by Subject
# Now, we are using the grades columns before they are dropped
plt.figure(figsize=(8,5))
sns.boxplot(data=grades_df)  # Using the grades_df DataFrame here
plt.title('Grade Distribution by Subject')
plt.tight_layout()
plt.savefig('Visuals/grade_boxplot.png')

# 3. Active Students Count
plt.figure(figsize=(4,3))
sns.countplot(x='active', data=df)
plt.title('Active vs Inactive Students')
plt.tight_layout()
plt.savefig('Visuals/active_status.png')

print("\n✅ Process complete! Visuals saved in 'Visuals/' folder.")
