"""
Feature engineering for AFL betting model.
Input: player_game_stats.csv, games_clean.csv
Output: features.csv (modeling table) and targets.
"""
import pandas as pd
import numpy as np
import os
from typing import List

PROCESSED_DIR = "data/processed"

def load_data():
    pg = pd.read_csv(os.path.join(PROCESSED_DIR, "player_game_stats.csv"))
    games = pd.read_csv(os.path.join(PROCESSED_DIR, "games_clean.csv"))
    return pg, games

def add_rolling_features(df: pd.DataFrame, stats: List[str], windows: List[int]) -> pd.DataFrame:
    """Add rolling mean and std for each stat, grouped by player, preserving all columns."""
    result = df.copy()
    # Ensure sorting by player and game_order within each player
    if 'game_order' in df.columns:
        result = result.sort_values(['player', 'game_order']).reset_index(drop=True)
    else:
        result = result.sort_values(['player', 'year']).reset_index(drop=True)
    for stat in stats:
        for w in windows:
            # Compute rolling mean and std using groupby rolling
            rolling_mean = result.groupby('player')[stat].rolling(w, min_periods=1).mean()
            rolling_std = result.groupby('player')[stat].rolling(w, min_periods=1).std().fillna(0)
            # Align indices with original DataFrame
            result[f"{stat}_last_{w}"] = rolling_mean.reset_index(level=0, drop=True)
            result[f"{stat}_std_{w}"] = rolling_std.reset_index(level=0, drop=True)
    return result

def create_target_variables(df: pd.DataFrame, line: float = 25.5) -> pd.DataFrame:
    """Create binary target: did player go over the line?"""
    df = df.copy()
    df['target_over'] = (df['disposals'] > line).astype(int)
    return df

def build_feature_table() -> pd.DataFrame:
    pg, games = load_data()
    # Compute rolling features
    stats = ['disposals', 'kicks', 'handballs', 'marks']
    windows = [5, 10, 20]
    pg = add_rolling_features(pg, stats, windows)
    # Ensure player column exists
    if 'player' not in pg.columns:
        raise ValueError("player column missing after feature engineering")
    # Create targets for common lines
    for line in [19.5, 24.5, 29.5]:
        pg = create_target_variables(pg, line)
        pg.rename(columns={'target_over': f'target_over_{line}'}, inplace=True)
    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pg.to_csv(os.path.join(PROCESSED_DIR, "features.csv"), index=False)
    print(f"Feature table created with {len(pg)} rows and columns: {list(pg.columns)}")
    return pg

if __name__ == "__main__":
    build_feature_table()