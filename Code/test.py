import pandas as pd

# --- Extract ---
file_path = "/Users/marcushjorth/Desktop/BI/BI-PROJECT-1-DATA-INGESTION-AND-WRANGLING/Data/Weather_data.txt"
df_raw = pd.read_csv(file_path, sep="\t")

# --- Transform ---
# Convert 'Date' column to datetime
df_raw['Date'] = pd.to_datetime(df_raw['Date'])

# Ensure Temperature and Humidity are floats
df_raw['Temperature'] = df_raw['Temperature'].astype(float)
df_raw['Humidity'] = df_raw['Humidity'].astype(float)

# Clean column names (optional, e.g., lowercase)
df_raw.columns = [col.lower() for col in df_raw.columns]

# --- Load ---
# Now df_raw is your cleaned and loaded DataFrame
print(df_raw.head())