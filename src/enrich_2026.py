"""
Enrich 2026 player-round data with opponent and venue from fixture.
"""
import pandas as pd

# Load data
players = pd.read_csv('data/raw/afltables_2026_full.csv')
fixture = pd.read_csv('data/raw/2026_fixture.csv')

# Normalize team names to match exactly
def normalize_name(s):
    return s.strip().replace('Greater Western Sydney', 'GWS').replace('Western Bulldogs', 'Western Bulldogs').replace('North Melbourne', 'North Melbourne')

players['team_norm'] = players['team'].apply(normalize_name)
fixture['home_norm'] = fixture['home_team'].apply(normalize_name)
fixture['away_norm'] = fixture['away_team'].apply(normalize_name)

# Build lookup: (round, team) -> (opponent, venue)
from collections import defaultdict
lookup = {}
for _, row in fixture.iterrows():
    rn = row['round']
    # Home team
    home = row['home_norm']
    away = row['away_norm']
    venue = row['venue']
    lookup[(rn, home)] = (away, venue)
    lookup[(rn, away)] = (home, venue)

# Map
def get_opponent_venue(row):
    key = (row['round'], row['team_norm'])
    opp_ven = lookup.get(key)
    if opp_ven:
        return pd.Series({'opponent': opp_ven[0], 'venue': opp_ven[1]})
    else:
        return pd.Series({'opponent': None, 'venue': None})

players[['opponent', 'venue']] = players.apply(get_opponent_venue, axis=1)

# Check coverage
missing = players[players['opponent'].isna()]
if len(missing) > 0:
    print(f"Warning: {len(missing)} player-rounds could not be matched to fixture (likely bye or team name mismatch)")
    print("Teams with missing:", missing['team'].unique())
    # Drop unmatched (bye weeks) for now
    players = players.dropna(subset=['opponent'])

# Save enriched dataset
players.to_csv('data/raw/afl2026_matches_enriched.csv', index=False)
print(f"Enriched dataset: {len(players)} player-round records with opponent and venue")
print("Rounds covered:", sorted(players['round'].unique()))
print("Teams:", players['team'].nunique())
