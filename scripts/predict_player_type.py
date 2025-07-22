import pandas as pd
import numpy as np
import pickle

# Define features (same ones always)
FEATURES = ['PTS', 'Age', 'Usage Rate', 'AST', 'TRB', 'STL', 'BLK', '3P%']

# Ask user which comparison
choice = input("Would you like a Regular Season (R) or Playoff (P) comparison? ").strip().lower()

if choice == 'r':
    print("\nYou selected: Regular Season Comparison")
    scaler = pickle.load(open('models/scaler_regular.pkl', 'rb'))
    kmeans = pickle.load(open('models/kmeans_regular.pkl', 'rb'))
    key_stats = pd.read_csv('data/key_stats/Key_Stats_Regular_Season.csv')
    clustered = pd.read_csv('data/clustered/Clustered_Manual_Regular_Season.csv')
elif choice == 'p':
    print("\nYou selected: Playoff Comparison")
    scaler = pickle.load(open('models/scaler_playoffs.pkl', 'rb'))
    kmeans = pickle.load(open('models/kmeans_playoffs.pkl', 'rb'))
    key_stats = pd.read_csv('data/key_stats/Key_Stats_Playoffs.csv')
    clustered = pd.read_csv('data/clustered/Clustered_Manual_Playoffs.csv')
else:
    print("\nInvalid input. Please restart and enter 'R' or 'P'.")
    exit()

# Prepare player dataset
X = key_stats[FEATURES]
X_scaled = scaler.transform(X)

# Ask user for input
print("\nPlease enter the following stats:")
user_input = []
for feature in FEATURES:
    value = float(input(f"{feature}: "))
    user_input.append(value)

# Scale user input
user_df = pd.DataFrame([user_input], columns=FEATURES)
user_scaled = scaler.transform(user_df)

# Predict cluster number
predicted_cluster = kmeans.predict(user_scaled)[0]

# Find players in the same cluster
key_stats['Cluster'] = kmeans.labels_
same_cluster_players = key_stats[key_stats['Cluster'] == predicted_cluster]

# Find closest player
cluster_scaled = X_scaled[key_stats['Cluster'] == predicted_cluster]
distances = np.linalg.norm(cluster_scaled - user_scaled, axis=1)
closest_idx = np.argmin(distances)
closest_player_row = same_cluster_players.iloc[closest_idx]

# Find their name
closest_player_name = closest_player_row['Player']

# Find the real label from clustered_manual file
matched_row = clustered[clustered['Player'] == closest_player_name]

if not matched_row.empty:
    real_label = matched_row.iloc[0]['Cluster_Label']
else:
    real_label = "Unknown"

# Output
print("\n--- Prediction Results ---")
print(f"Predicted Player Type: {real_label}")
print(f"Most similar NBA Player: {closest_player_name}")
