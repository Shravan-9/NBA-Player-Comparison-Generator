import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os

# Paths
regular_season_path = "data/processed/ml_ready/ML_Ready_NBA_Player_Stats_Regular_Season.csv"
playoffs_path = "data/processed/ml_ready/ML_Ready_NBA_Player_Stats_Playoffs.csv"
models_dir = r"C:\Users\Shrav\OneDrive\Desktop\NBA_Stats_Project\models"


# Make sure models directory exists
os.makedirs(models_dir, exist_ok=True)

# Features we will use
selected_features = [
    "PTS",
    "Age",
    "Usage Rate",
    "AST",
    "TRB",
    "STL",
    "BLK",
    "3P%"
]

def train_and_save_models(input_csv, scaler_path, kmeans_path):
    print(f"Loading {input_csv}...")
    data = pd.read_csv(input_csv, delimiter=';')
    
    # Select only the desired 8 features
    X = data[selected_features]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Save the scaler and KMeans model
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)

    print(f"Saved scaler to {scaler_path}")
    print(f"Saved kmeans model to {kmeans_path}\n")

# Train and save for regular season
train_and_save_models(
    regular_season_path,
    os.path.join(models_dir, "scaler_regular.pkl"),
    os.path.join(models_dir, "kmeans_regular.pkl")
)

# Train and save for playoffs
train_and_save_models(
    playoffs_path,
    os.path.join(models_dir, "scaler_playoffs.pkl"),
    os.path.join(models_dir, "kmeans_playoffs.pkl")
)

print(" Finished training scalers and kmeans models for both datasets!")