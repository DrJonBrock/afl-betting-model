"""
Prepare inputs for context feature pipeline from transformed real data.
"""
import pandas as pd
import os

# Load transformed full dataset
df = pd.read_csv('data/processed/player_game_stats.csv')
# Ensure column names
df = df.rename(columns={'disposals': 'disposals'})

# 1. Create player_game_stats_clean.csv (historical 2020-2025)
hist = df[df['year'] < 2026].copy()
hist.to_csv('data/processed/player_game_stats_clean.csv', index=False)
print(f"Created player_game_stats_clean.csv: {len(hist)} rows (2020-2025)")

# 2. Create afl2026_matches_enriched.csv (2026 only, keep required columns)
df2026 = df[df['year'] == 2026].copy()
# Keep essential columns: year, player, team, round, disposals, opponent, venue, is_home
cols_keep = ['year','player','team','round','disposals','opponent','venue','is_home']
# If is_home missing (should be present), fill with 0
if 'is_home' not in df2026.columns:
    df2026['is_home'] = 0
df2026 = df2026[cols_keep]
df2026.to_csv('data/raw/afl2026_matches_enriched.csv', index=False)
print(f"Created afl2026_matches_enriched.csv: {len(df2026)} rows (2026)")
