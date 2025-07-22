import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

def standardize_positions(data):
    """
    Standardizes the 'Pos' column by randomly selecting one position
    if a player has a dual position (e.g., 'PG-SG').
    """
    print("Standardizing positions...")
    def resolve_position(pos):
        if '-' in pos:  # If there's a dual position like 'PG-SG'
            return random.choice(pos.split('-'))
        return pos  # Return the position as-is if no '-'

    data['Pos'] = data['Pos'].apply(resolve_position)
    print("Positions standardized successfully.")
    return data

def scale_numeric_columns(data):
    """
    Scales numeric columns using StandardScaler, excluding categorical or one-hot-encoded columns.
    """
    print("Scaling numeric columns...")
    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Exclude one-hot encoded position columns
    one_hot_columns = [col for col in data.columns if col.startswith('Pos_')]
    numeric_columns = [col for col in numeric_columns if col not in one_hot_columns + ['Player']]

    # Apply StandardScaler
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    print("Numeric columns scaled successfully.")

    return data

def preprocess_data(input_file, output_file, mapping_output_file):
    """
    Reads data from the input file, standardizes positions, removes unnecessary columns,
    scales numeric columns, saves the processed data, and stores a separate ID-to-Player mapping.
    """
    try:
        # Load the cleaned data
        data = pd.read_csv(input_file, delimiter=';')
        print(f"Processing {input_file}...")

        # Save a mapping of ID to Player Name for later interpretation
        player_mapping = data[['ID', 'Player']]
        player_mapping.to_csv(mapping_output_file, index=False, sep=';')
        print(f"Saved Player Mapping to {mapping_output_file}.")

        # Drop unnecessary columns: ID, Player (Names), and PF (Personal Fouls)
        columns_to_drop = ['ID', 'Player', 'PF']
        data = data.drop(columns=columns_to_drop, errors='ignore')
        print("Dropped unnecessary columns: ID, Player, PF.")

        # Standardize the positions
        data = standardize_positions(data)

        # Apply one-hot encoding to 'Pos'
        position_dummies = pd.get_dummies(data['Pos'], prefix='Pos')
        data = pd.concat([data, position_dummies], axis=1)
        data = data.drop(columns=['Pos'])
        print("One-hot encoding applied to 'Pos' column.")

        # Scale numeric columns
        data = scale_numeric_columns(data)

        # Save the ML-ready data
        data.to_csv(output_file, index=False, sep=';')
        print(f"ML-ready data saved to {output_file}.")

    except Exception as e:
        print(f"An error occurred while processing {input_file}: {e}")

if __name__ == "__main__":
    # Process the regular season data
    preprocess_data(
        'data/processed/cleaned/Cleaned_NBA_Player_Stats_Regular_Season.csv',
        'data/processed/ml_ready/ML_Ready_NBA_Player_Stats_Regular_Season.csv',
        'data/processed/ml_ready/Player_Mapping_Regular_Season.csv'
    )

    # Process the playoffs data
    preprocess_data(
        'data/processed/cleaned/Cleaned_NBA_Player_Stats_Playoffs.csv',
        'data/processed/ml_ready/ML_Ready_NBA_Player_Stats_Playoffs.csv',
        'data/processed/ml_ready/Player_Mapping_Playoffs.csv'
    )
