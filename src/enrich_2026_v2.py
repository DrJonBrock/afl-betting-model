"""
Enrich 2026 player-round data with opponent and venue from fixture mapping.
"""
import pandas as pd

players = pd.read_csv('data/raw/afltables_2026_full.csv')
fixture = pd.read_csv('data/raw/2026_fixture_mapping.csv')

# Normalize team names to match between datasets
def normalize(s):
    if pd.isna(s):
        return s
    s = s.strip()
    # The fixture mapping may have slightly different names; ensure consistency
    replacements = {
        'Greater Western Sydney': 'GWS',
        'Brisbane Lions': 'Brisbane',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

players['team_norm'] = players['team'].apply(normalize)
players['round'] = players['round'].astype(int)
fixture['round'] = fixture['round'].astype(int)
fixture['team'] = fixture['team'].apply(normalize)

# Merge on round+team
merged = players.merge(
    fixture[['round','team','opponent','venue','is_home']],
    left_on=['round','team_norm'],
    right_on=['round','team'],
    how='left'
)

# Check unmatched
unmatched = merged[merged['opponent'].isna()]
if len(unmatched) > 0:
    print(f"Unmatched {len(unmatched)} player-rounds (likely bye or team name mismatch)")
    print("Teams with unmatched:", unmatched['team_norm'].unique())
    # Keep only matched
    merged = merged.dropna(subset=['opponent'])

# Save
merged.to_csv('data/raw/afl2026_matches_enriched.csv', index=False)
print(f"Enriched dataset: {len(merged)} player-round records")
print("Rounds:", sorted(merged['round'].unique()))
print("Teams:", merged['team_norm'].nunique())
