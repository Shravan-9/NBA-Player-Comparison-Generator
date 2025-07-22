import pandas as pd
import os

# Input and output paths
regular_season_input = "data/processed/cleaned/Cleaned_NBA_Player_Stats_Regular_Season.csv"
playoffs_input = "data/processed/cleaned/Cleaned_NBA_Player_Stats_Playoffs.csv"
output_dir = "data/key_stats/"

# Features we want to keep
FEATURES = ['Player', 'PTS', 'Age', 'Usage Rate', 'AST', 'TRB', 'STL', 'BLK', '3P%']

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_file(input_path, output_name):
    df = pd.read_csv(input_path, delimiter=';')
    df_key = df[FEATURES]
    df_key.to_csv(os.path.join(output_dir, output_name), index=False)
    print(f"Saved {output_name} to {output_dir}")

# Run for both regular season and playoffs
process_file(regular_season_input, "Key_Stats_Regular_Season.csv")
process_file(playoffs_input, "Key_Stats_Playoffs.csv")
