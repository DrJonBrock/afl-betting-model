"""
Merge match total over/under odds from AFL Data Analysis repo to player-game dataset.
Updated with proper team name normalization to match odds dataset.
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
        home_df = df_map[df_map['is_home'] == 1].copy()
        home_df = home_df.rename(columns={'team': 'home_team', 'opponent': 'away_team'})
        home_df = home_df[['year','round','home_team','away_team','venue']]
        fixture_frames.append(home_df)
fixture_matches = pd.concat(fixture_frames, ignore_index=True)
print(f"Unique matches from fixtures: {len(fixture_matches)} (years {fixture_matches['year'].min()}-{fixture_matches['year'].max()})")

# Normalize team names to match odds dataset
def normalize_team_name(name):
    mapping = {
        'Brisbane Lions': 'Brisbane',
        'Greater Western Sydney': 'GWS Giants',
        'West Coast Eagles': 'West Coast',  # if present
        'Gold Coast Suns': 'Gold Coast',
    }
    return mapping.get(name, name)

fixture_matches['home_team'] = fixture_matches['home_team'].apply(normalize_team_name)
fixture_matches['away_team'] = fixture_matches['away_team'].apply(normalize_team_name)

# 2. Load odds
odds_path = 'external/afl-data-analysis/odds_data/odds_data_2009_to_present.xlsx'
odds = pd.read_excel(odds_path)
odds['Date'] = pd.to_datetime(odds['Date'])
odds['year'] = odds['Date'].dt.year
odds = odds.rename(columns={
    'Total Score Close': 'total_score_close',
    'Home Team': 'Home Team',
    'Away Team': 'Away Team'
})
# Already normalized? Check if odds use 'Brisbane' and 'GWS Giants' - they do.
odds_match = odds[['year','Home Team','Away Team','total_score_close']].drop_duplicates(subset=['year','Home Team','Away Team'])
print(f"Odds matches available: {len(odds_match)} (years {odds_match['year'].min()}-{odds_match['year'].max()})")

# 3. Merge odds onto fixture matches
matches_with_odds = pd.merge(
    fixture_matches,
    odds_match,
    how='inner',
    left_on=['year','home_team','away_team'],
    right_on=['year','Home Team','Away Team']
)
matches_with_odds = matches_with_odds.drop(columns=['Home Team','Away Team'])
print(f"Matches with odds after merge: {len(matches_with_odds)}")
if len(matches_with_odds) > 0:
    print("Year distribution:")
    print(matches_with_odds['year'].value_counts().sort_index())

# Save match-level odds feature
out_matches = 'data/processed/match_odds.csv'
matches_with_odds.to_csv(out_matches, index=False)
print(f"Saved match odds to {out_matches}")

# 4. Merge into player-game stats
pg = pd.read_csv('data/processed/player_game_stats.csv')
# Standardize names in player-game as well (teams and opponents)
pg['team'] = pg['team'].apply(normalize_team_name)
pg['opponent'] = pg['opponent'].apply(normalize_team_name)
# We also have is_home; create home_team and away_team
pg['home_team'] = pg.apply(lambda r: r['team'] if r['is_home']==1 else r['opponent'], axis=1)
pg['away_team'] = pg.apply(lambda r: r['opponent'] if r['is_home']==1 else r['team'], axis=1)

# Merge left
pg_merged = pd.merge(
    pg,
    matches_with_odds[['year','round','home_team','away_team','total_score_close']],
    how='left',
    on=['year','round','home_team','away_team']
)
missing = pg_merged['total_score_close'].isna().sum()
total = len(pg_merged)
print(f"Player-game rows: {total}, missing total_score_close: {missing} ({missing/total*100:.1f}%)")

# Impute missing with median per team? For now, just fill with overall median.
median_line = pg_merged['total_score_close'].median()
pg_merged['total_score_close'] = pg_merged['total_score_close'].fillna(median_line)
print(f"Imputed missing with median: {median_line:.1f}")

# Save updated dataset
pg_merged.to_csv('data/processed/player_game_wide_with_odds.csv', index=False)
print("Saved player-game with odds feature: data/processed/player_game_wide_with_odds.csv")
