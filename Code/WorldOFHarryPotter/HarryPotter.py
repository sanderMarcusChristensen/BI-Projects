import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create Visuals folder if not exists
os.makedirs('Visuals', exist_ok=True)

# Fetch data from the Harry Potter API
url = "https://hp-api.onrender.com/api/characters"
response = requests.get(url)

# If the request is successful, process the data
if response.status_code == 200:
    data = response.json()  # Get data as JSON
    df = pd.DataFrame(data)  # Convert data to DataFrame

    # Replace missing 'species' with 'Unknown' instead of removing them
    df['species'].fillna('Unknown', inplace=True)

    # Keep only 'species' column since 'name' is not needed for the plot
    df_cleaned = df[['species']]

    # Count how many characters there are in each species
    species_counts = df_cleaned['species'].value_counts()

    # Total number of characters in the dataset
    total_characters = len(df_cleaned)

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.barplot(x=species_counts.values, y=species_counts.index, palette='muted', orient='h')
    plt.title('Most Common Species in Harry Potter')
    plt.xlabel('Number of Characters')
    plt.ylabel('Species')

    # Add text for the total character count on the plot
    plt.text(0.95, 0.95, f'Total Characters: {total_characters}', 
             ha='center', va='top', transform=plt.gca().transAxes,
             fontsize=12, color='black', weight='bold')

    # Save the plot in 'Visuals' folder
    plt.tight_layout()
    plt.savefig('Visuals/most_common_species.png')

    print("✅ The plot has been saved in the 'Visuals' folder.")
else:
    print("Failed to fetch data from the API.")
