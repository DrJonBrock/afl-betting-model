"""
Integrate betting market feature: total score close line (over/under) for each match.
"""
import pandas as pd
import os

# 1. Build unique match table from fixture mappings (2020-2026)
fixture_frames = []
for year in range(2020, 2027):
    map_file = f'data/raw/fixture_mapping_{year}.csv'
    if os.path.exists(map_file):
        df_map = pd.read_csv(map_file)
        df_map['year'] = year
        # Filter to home team rows to get one row per match
        home_df = df_map[df_map['is_home'] == 1].copy()
        home_df = home_df.rename(columns={'team': 'home_team', 'opponent': 'away_team'})
        # Keep only needed columns
        home_df = home_df[['year','round','home_team','away_team','venue']]
        fixture_frames.append(home_df)
    else:
        print(f"Missing {map_file}")
fixture_matches = pd.concat(fixture_frames, ignore_index=True)
print(f"Unique matches from fixtures: {len(fixture_matches)} (years {fixture_matches['year'].min()}-{fixture_matches['year'].max()})")

# Standardize team names to match odds dataset
team_mapping = {
    'GWS': 'GWS Giants',
    # others are same
}
def std_name(name):
    return team_mapping.get(name, name)
fixture_matches['home_team'] = fixture_matches['home_team'].apply(std_name)
fixture_matches['away_team'] = fixture_matches['away_team'].apply(std_name)

# 2. Load odds and select total close line
odds_path = 'external/afl-data-analysis/odds_data/odds_data_2009_to_present.xlsx'
odds = pd.read_excel(odds_path)
odds['Date'] = pd.to_datetime(odds['Date'])
odds['year'] = odds['Date'].dt.year
odds = odds.rename(columns={
    'Total Score Close': 'total_score_close',
    'Home Team': 'Home Team',  # keep as is
    'Away Team': 'Away Team'
})
# Standardize odds team names
odds['Home Team'] = odds['Home Team'].apply(std_name)
odds['Away Team'] = odds['Away Team'].apply(std_name)

# Keep only relevant columns and drop duplicates (some years have multiple odds entries per match? use first)
odds_match = odds[['year','Home Team','Away Team','total_score_close']].drop_duplicates(subset=['year','Home Team','Away Team'])
print(f"Odds matches: {len(odds_match)} (years {odds_match['year'].min()}-{odds_match['year'].max()})")

# 3. Merge odds onto fixture matches (inner join to keep only matches with odds)
matches_with_odds = pd.merge(
    fixture_matches,
    odds_match,
    how='inner',
    left_on=['year','home_team','away_team'],
    right_on=['year','Home Team','Away Team']
)
matches_with_odds = matches_with_odds.drop(columns=['Home Team','Away Team'])
print(f"Matches with odds after merge: {len(matches_with_odds)}")

# Save match-level odds feature
out_matches = 'data/processed/match_odds.csv'
matches_with_odds.to_csv(out_matches, index=False)
print(f"Saved match odds to {out_matches}")

# 4. Prepare to merge into player-game stats
# Load player-game stats (with is_home)
pg = pd.read_csv('data/processed/player_game_stats.csv')
# Standardize team names in pg
pg['team'] = pg['team'].apply(std_name)
pg['opponent'] = pg['opponent'].apply(std_name)

# Create home_team and away_team columns for merging
pg['home_team'] = pg.apply(lambda r: r['team'] if r['is_home']==1 else r['opponent'], axis=1)
pg['away_team'] = pg.apply(lambda r: r['opponent'] if r['is_home']==1 else r['team'], axis=1)

# Merge with match odds
pg_merged = pd.merge(
    pg,
    matches_with_odds[['year','round','home_team','away_team','total_score_close']],
    how='left',
    left_on=['year','round','home_team','away_team'],
    right_on=['year','round','home_team','away_team']
)
missing = pg_merged['total_score_close'].isna().sum()
print(f"Player-game rows: {len(pg_merged)}, missing total_score_close: {missing}")

# Save updated player-game with odds feature
pg_merged.to_csv('data/processed/player_game_stats_with_odds.csv', index=False)
print("Saved player-game with odds feature.")
