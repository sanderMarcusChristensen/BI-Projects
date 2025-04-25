import pandas as pd

# --- Extract ---
file_path = "../Data/Weather_data.txt"

# Read the file
df_raw = pd.read_csv(file_path, sep="\t", encoding='utf-8')

# Check the shape of the DataFrame (rows, columns) and display it fully
print("DataFrame Shape:", df_raw.shape)
#print(df_raw)

# Display all rows and columns if the data is large
#pd.set_option('display.max_rows', None)  # Show all rows
#pd.set_option('display.max_columns', None)  # Show all columns
#print(df_raw)  # Print all the rows and columns of the DataFrame

# --- Transform ---
# Convert 'Date' column to datetime
df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

# Ensure Temperature and Humidity are floats
df_raw['Temperature'] = df_raw['Temperature'].astype(float)
df_raw['Humidity'] = df_raw['Humidity'].astype(float)

# Clean column names (optional, e.g., lowercase)
#df_raw.columns = [col.lower() for col in df_raw.columns]

# --- Load ---
# Now df_raw is your cleaned and loaded DataFrame
print(df_raw)  # Print the first few rows to confirm changes