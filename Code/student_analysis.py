import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# This function reads a JSON file and loads it into a pandas DataFrame
df = pd.read_json('Data/Json_data.json')

# shows the first 5 rows of the DataFrame
print(df.head())

# Ensure the 'Visuals' folder exists
if not os.path.exists('Visuals'):
    os.makedirs('Visuals')

# --- Visualizations ---

# 1. Age Distribution (Simple Histogram)
plt.figure(figsize=(6,4))
sns.histplot(df['age'], kde=True, bins=5)
plt.title('Student Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Visuals/age_distribution.png')

# 2. Active vs Inactive Students (Simple Countplot)
plt.figure(figsize=(4,3))
sns.countplot(x='active', data=df)
plt.title('Active vs Inactive Students')
plt.tight_layout()
plt.savefig('Visuals/active_status.png')

# Notify that visuals are saved
print("\n✅ Visuals saved in 'Visuals/' folder.")
