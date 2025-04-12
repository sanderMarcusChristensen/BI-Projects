import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'Data', 'Json_data.json')
VISUALS_DIR = os.path.join(BASE_DIR, '..', '..', 'Visuals')

# Create Visuals folder if not existing
os.makedirs(VISUALS_DIR, exist_ok=True)

# --- Load Data ---
df = pd.read_json(DATA_PATH)

# --- Visualizations ---
# 1. Age Distribution (Bar Plot)
age_counts = df['age'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
sns.barplot(x=age_counts.index, y=age_counts.values, palette='viridis')
plt.title('Student Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'age_distribution_simple.png'))

# 2. Active vs Inactive Students (Count Plot)
plt.figure(figsize=(4, 3))
sns.countplot(x='active', data=df)
plt.title('Active vs Inactive Students')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'active_status.png'))

print("✅ Visuals saved in 'Visuals/' folder.")
