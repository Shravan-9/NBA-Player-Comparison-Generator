import pandas as pd
import os

# Define your rules
def assign_cluster(row):
    if row['PTS'] < 5 or row['Usage Rate'] < 0.2:
        return "Bench Player"
    elif row['PTS'] >= 23 or (row['Usage Rate'] >= 28 and row['PTS'] >= 20):
        return "All Star"
    elif (row['STL'] + row['BLK']) >= 2.0:
        return "Defensive Specialist"
    elif row['3P%'] >= 40 and row['Usage Rate'] >= 0.5:
        return "3PT Specialist"
    else:
        return "Role Player"

# Paths
input_dir = "data/key_stats"
output_dir = "data/clustered"
os.makedirs(output_dir, exist_ok=True)

# Files
regular_season_input = os.path.join(input_dir, "Key_Stats_Regular_Season.csv")
playoffs_input = os.path.join(input_dir, "Key_Stats_Playoffs.csv")

# Process a single file
def process_file(input_path, output_filename):
    df = pd.read_csv(input_path)  
    
    # Apply the rules
    df['Cluster_Label'] = df.apply(assign_cluster, axis=1)

    # Save to clustered_manual/
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved clustered data to {output_path}")

# Run
process_file(regular_season_input, "Clustered_Manual_Regular_Season.csv")
process_file(playoffs_input, "Clustered_Manual_Playoffs.csv")
