# flask_app/app.py
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load scaler and KMeans models
scaler = joblib.load('models/scaler_regular.pkl')  # Default (Regular Season scaler)
kmeans = joblib.load('models/kmeans_regular.pkl')

# Load manually clustered datasets
regular_season_data = pd.read_csv('data/clustered/Clustered_Manual_Regular_Season.csv')
playoff_data = pd.read_csv('data/clustered/Clustered_Manual_Playoffs.csv')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Collect form data
        pts = float(request.form['PTS'])
        age = float(request.form['Age'])
        usage = float(request.form['Usage_Rate'])
        ast = float(request.form['AST'])
        trb = float(request.form['TRB'])
        stl = float(request.form['STL'])
        blk = float(request.form['BLK'])
        three_pct = float(request.form['ThreeP'])

        # Season choice
        season = request.form['Season']

        # Select dataset and models based on season
        if season == 'P':
            selected_data = playoff_data
            scaler_path = 'models/scaler_playoffs.pkl'
            kmeans_path = 'models/kmeans_playoffs.pkl'
        else:
            selected_data = regular_season_data
            scaler_path = 'models/scaler_regular.pkl'
            kmeans_path = 'models/kmeans_regular.pkl'

        # Load appropriate scaler and kmeans
        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)

        # Prepare user input
        user_input = np.array([[pts, age, usage, ast, trb, stl, blk, three_pct]])
        user_scaled = scaler.transform(user_input)

        # Predict cluster
        predicted_cluster = kmeans.predict(user_scaled)[0]

        # Get label directly from the clustered dataset
        cluster_label = selected_data[selected_data['Cluster_Label'] == predicted_cluster]['Cluster_Label'].values
        if len(cluster_label) > 0:
            player_type = cluster_label[0]
        else:
            player_type = "Unknown"

        # Find most similar player
        features = ['PTS', 'Age', 'Usage Rate', 'AST', 'TRB', 'STL', 'BLK', '3P%']
        X = selected_data[features]
        X_scaled = scaler.transform(X)
        distances = np.linalg.norm(X_scaled - user_scaled, axis=1)
        closest_idx = np.argmin(distances)
        closest_player = selected_data.iloc[closest_idx]['Player']

        return render_template('result.html', player_type=player_type, closest_player=closest_player)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
