import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import os

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_correlation_matrix(data, dataset_name):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=False, cmap="coolwarm", center=0)
    plt.title(f"Feature Correlation Matrix - {dataset_name}")
    ensure_directory_exists("images")
    plt.savefig(f"images/correlation_matrix_{dataset_name}.png")
    plt.close()

def plot_pca_variance(data, dataset_name):
    pca = PCA()
    pca.fit(data)

    # Scree plot (individual explained variance)
    plt.figure(figsize=(10, 6))
    plt.plot(pca.explained_variance_ratio_, marker='o')
    plt.xlabel("Principal Component Index")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot (Explained Variance per Component)")
    ensure_directory_exists("images")
    plt.savefig(f"images/pca_scree_plot_{dataset_name}.png")
    plt.close()


def feature_importance(data, dataset_name):
    X = data.drop(columns=['PIE'], errors='ignore')  # Target variable is PIE
    y = data['PIE'] if 'PIE' in data.columns else data.mean(axis=1)  # Default to mean if missing
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importance.values, y=importance.index, palette="viridis")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    ensure_directory_exists("images")
    plt.savefig(f"images/feature_importance_{dataset_name}.png")
    plt.close()

def analyze_features(file_path, dataset_name):
    print(f"Analyzing Features for: {dataset_name}")
    data = pd.read_csv(file_path, delimiter=';')
    
    plot_correlation_matrix(data, dataset_name)
    plot_pca_variance(data, dataset_name)
    feature_importance(data, dataset_name)

if __name__ == "__main__":
    analyze_features("data/processed/ml_ready/ML_Ready_NBA_Player_Stats_Regular_Season.csv", "Regular_Season")
    analyze_features("data/processed/ml_ready/ML_Ready_NBA_Player_Stats_Playoffs.csv", "Playoffs")
