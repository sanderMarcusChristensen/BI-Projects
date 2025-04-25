import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Path Setup ---
# Find den mappe hvor denne Python-fil ligger
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Gå op til 'Json'-mappen
JSON_DIR = os.path.dirname(CURRENT_DIR)

# Gå op til 'Code'-mappen
CODE_DIR = os.path.dirname(JSON_DIR)

# Gå op til 'MP1 - Data Ingestion and Wrangling'-mappen
MP1_DIR = os.path.dirname(CODE_DIR)

# Find Data-folderen
DATA_DIR = os.path.join(MP1_DIR, 'Data')

# Find Visuals-folderen (den ligger under 'Json')
VISUALS_DIR = os.path.join(JSON_DIR, 'Visuals')

# Filstier
DATA_PATH = os.path.join(DATA_DIR, 'Json_data.json')

# Sørg for at 'Visuals'-folderen eksisterer
os.makedirs(VISUALS_DIR, exist_ok=True)

# --- Load Data ---
df = pd.read_json(DATA_PATH)

# --- Visualizations ---
# 1. Age Distribution (Bar Plot)
age_counts = df['age'].value_counts().sort_index()

plt.figure(figsize=(6, 4))
sns.barplot(
    x=age_counts.index,
    y=age_counts.values,
    hue=age_counts.index,         # Tilføjet hue
    palette='viridis',
    legend=False                  # Ingen legend
)
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
