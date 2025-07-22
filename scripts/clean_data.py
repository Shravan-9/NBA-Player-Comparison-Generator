import pandas as pd

def drop_tm_column(input_file, output_file, is_regular_season=True):
    try:
        data = pd.read_csv(input_file, delimiter=';', encoding='utf-8')
        print(f"Processing {input_file}...")

        data = fix_player_names(data)

        if 'Tm' in data.columns:
            data = data.drop(columns=['Tm'])
            print("'Tm' column dropped successfully.")

        if 'Rk' in data.columns:
            data = data.drop(columns=['Rk'])
            print("'Rk' column dropped successfully.")

        if is_regular_season:
            # Drop players with fewer than 20 games
            data = drop_players_with_few_games(data, min_games=20)

        data = add_id_column(data)
        data = move_column_to_index(data, 'ID', 0)
        data = average_duplicate_players(data)
        data = add_advanced_metrics(data)

        # Save human-readable backup
        human_readable_backup = output_file.replace('Cleaned', 'Human_Readable_Cleaned')
        data.to_csv(human_readable_backup, index=False, sep=';')
        print(f"Human-readable backup saved to {human_readable_backup}.\n")
        
        # Save cleaned data
        data.to_csv(output_file, index=False, sep=';')
        print(f"Cleaned data saved to {output_file}.\n")

    except Exception as e:
        print(f"An error occurred while processing {input_file}: {e}")


def fix_player_names(data):
    """
    Fixes known problematic player names with question marks in them.
    """
    name_replacements = {
        'D?vis Bert?ns': 'Davis Bertans',
        'Bogdan Bogdanovi?': 'Bogdan Bogdanovic',
        'Bojan Bogdanovi?': 'Bojan Bogdanovic',
        'Vlatko ?an?ar': 'Vlatko Cancar',
        'Luka Don?i?': 'Luka Doncic',
        'Goran Dragi?': 'Goran Dragic',
        'Nikola Joki?': 'Nikola Jokic',
        'Boban Marjanovi?': 'Boban Marjanovic',
        'Jusuf Nurki?': 'Jusuf Nurkic',
        'Kristaps Porzi??is': 'Kristaps Porzingis',
        'Jonas Valan?i?nas': 'Jonas Valanciunas',
        'Nikola Vu?evi?': 'Nikola Vucevic',
        'Nikola Jovi?': 'Nikola Jovic',
    }
    # Replace the problematic names
    data['Player'] = data['Player'].replace(name_replacements)
    print("Fixed player names successfully.")
    return data

def add_id_column(data):
    """
    Adds an 'ID' column to the DataFrame, with values starting from 1 and incrementing by 1.
    """
    data['ID'] = range(1, len(data) + 1)
    print("Added 'ID' column successfully.")
    return data

def move_column_to_index(data, column_name, index):
    """
    Moves a specified column to the desired index in the DataFrame.
    """
    col = data.pop(column_name)
    data.insert(index, column_name, col)
    print(f"Moved '{column_name}' column to index {index}.")
    return data

def average_duplicate_players(data):
    """
    Averages out the statistics of players with duplicate entries, excluding non-numeric columns.
    Recomputes the ID column and rounds numeric columns to one decimal place.
    """
    # Identify numeric columns for averaging (exclude 'ID')
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    numeric_columns.remove('ID')  # Exclude 'ID' from averaging

    # Group by 'Player' and compute the mean for numeric columns only
    averaged_data = data.groupby('Player', as_index=False)[numeric_columns].mean()

    # Retain non-numeric columns (e.g., Position) from the first occurrence of each player
    non_numeric_columns = [col for col in data.columns if col not in numeric_columns + ['Player']]
    non_numeric_data = data.drop_duplicates(subset='Player')[['Player'] + non_numeric_columns]

    # Merge the averaged numeric data with non-numeric data
    final_data = pd.merge(averaged_data, non_numeric_data, on='Player', how='left')

    # Round all numeric columns to one decimal place
    final_data[numeric_columns] = final_data[numeric_columns].round(1)

    # Recompute ID column after processing duplicates
    final_data['ID'] = range(1, len(final_data) + 1)

    # Move 'ID' back to index 0
    if 'ID' in final_data.columns:
        final_data = move_column_to_index(final_data, 'ID', 0)

    print("Averaged statistics for duplicate players successfully.")
    return final_data

def drop_players_with_few_games(data, min_games=20):
    """
    Drops players who played fewer than the specified number of games.
    
    Parameters:
        data (pd.DataFrame): The DataFrame to filter.
        min_games (int): The minimum number of games a player must have played to be retained.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if 'G' in data.columns:
        original_count = len(data)
        data = data[data['G'] >= min_games]
        filtered_count = len(data)
        print(f"Dropped {original_count - filtered_count} players with fewer than {min_games} games.")
    else:
        print("'G' column not found in the dataset. Skipping game count filtering.")
    return data


def add_advanced_metrics(data):
    """
    Adds advanced metrics to the dataset, including scoring dependencies, per-minute metrics,
    and additional performance ratios.
    """
    # Avoid division by zero
    data['PTS'] = data['PTS'].replace(0, 1)
    data['MP'] = data['MP'].replace(0, 1)
    data['TOV'] = data['TOV'].replace(0, 1)
    data['FGA'] = data['FGA'].replace(0, 1)

    # Scoring dependencies
    data['3P Dependency'] = ((data['3P'] * 3) / data['PTS']).round(3)
    data['2P Dependency'] = ((data['2P'] * 2) / data['PTS']).round(3)
    data['FT Dependency'] = (data['FT'] / data['PTS']).round(3)

    # Per-minute metrics
    data['PTS/MP'] = (data['PTS'] / data['MP']).round(3)
    data['AST/MP'] = (data['AST'] / data['MP']).round(3)
    data['TRB/MP'] = (data['TRB'] / data['MP']).round(3)
    data['STL/MP'] = (data['STL'] / data['MP']).round(3)
    data['BLK/MP'] = (data['BLK'] / data['MP']).round(3)

    # Assist-to-Turnover Ratio
    data['AST/TOV'] = (data['AST'] / data['TOV']).round(3)

    # Rebound Ratio (simplified)
    data['Rebound Ratio'] = ((data['ORB'] + data['DRB']) / data['G']).round(3)

    # Scoring Efficiency
    data['PTS/FGA'] = (data['PTS'] / data['FGA']).round(3)

    # Free Throw Rate
    data['FT Rate'] = (data['FTA'] / data['FGA']).round(3)

    # Usage Rate (Simplified)
    data['Usage Rate'] = ((data['FGA'] + 0.44 * data['FTA'] + data['TOV']) / data['MP']).round(3)

    # Player Impact Estimate (PIE)
    data['PIE'] = (
        (data['PTS'] + data['FG'] + data['FT'] - data['FGA'] - data['FTA'] +
         data['ORB'] + data['DRB'] + data['AST'] + data['STL'] + data['BLK'] - data['TOV'])
    ).round(3)

    print("Advanced metrics added successfully.")
    return data


# Regular season: Apply the filter
drop_tm_column(
    'data/raw/Combined NBA Player Stats - Regular.csv',
    'data/processed/Cleaned_NBA_Player_Stats_Regular_Season.csv',
    is_regular_season=True  # Regular season flag
)

# Playoffs: Skip the filter
drop_tm_column(
    'data/raw/Combined NBA Player Stats - Playoffs.csv',
    'data/processed/Cleaned_NBA_Player_Stats_Playoffs.csv',
    is_regular_season=False  # Playoffs flag
)

