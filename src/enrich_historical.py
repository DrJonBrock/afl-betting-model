"""
Enrich historical player_game_stats.csv with round, opponent, venue, is_home using fixture mappings.
"""
import pandas as pd
import os

DATA_PATH = "data/processed/player_game_stats.csv"
OUT_PATH = "data/processed/player_game_stats_with_context.csv"

print("Loading player_game_stats...")
df = pd.read_csv(DATA_PATH)
print(f"Total rows: {len(df)}")
print("Years:", sorted(df['year'].unique()))
print("Sample:", df.head(2).to_string(index=False))

# We'll create a list to collect enriched rows
enriched_rows = []

# Group by year, team
grouped = df.groupby(['year', 'team'])
print(f"Number of (year,team) groups: {len(grouped)}")

for (year, team), group in grouped:
    # Load mapping for this year
    map_path = f"data/raw/fixture_mapping_{int(year)}.csv"
    if not os.path.exists(map_path):
        print(f"Warning: no mapping for {year}, skipping {team}")
        continue
    mapping = pd.read_csv(map_path)
    team_map = mapping[mapping['team'] == team].copy()
    if len(team_map) == 0:
        print(f"Warning: {team} not found in {year} mapping")
        continue
    # Ensure consistent team name normalization? The mapping team names may differ (e.g., "Greater Western Sydney" vs "GWS")
    # We'll attempt to match by standardizing: if team not found, try fuzzy mapping
    if len(team_map) == 0:
        # try to normalize team names in mapping
        def norm(s):
            return s.strip().replace('Greater Western Sydney', 'GWS').replace('Brisbane Lions', 'Brisbane')
        mapping['team_norm'] = mapping['team'].apply(norm)
        team_norm = norm(team)
        team_map = mapping[mapping['team_norm'] == team_norm]
        if len(team_map) == 0:
            print(f"Warning: still no match for {team} in {year}")
            continue
    # Now team_map should have one row per game the team played, with round, opponent, venue, is_home
    # Sort by round
    team_map = team_map.sort_values('round')
    # The group rows represent player-game logs; we assume they are already in order by game_order which corresponds to chronological order
    group_sorted = group.sort_values('game_order').reset_index(drop=True)
    if len(group_sorted) != len(team_map):
        print(f"Warning: {year} {team} has {len(group_sorted)} player-game rows but mapping has {len(team_map)} matches. Skipping.")
        continue
    # Assign round/opponent/venue/is_home from mapping
    for i, row in group_sorted.iterrows():
        assign = team_map.iloc[i]
        row_dict = row.to_dict()
        row_dict['round'] = assign['round']
        row_dict['opponent'] = assign['opponent']
        row_dict['venue'] = assign['venue']
        row_dict['is_home'] = assign['is_home']
        enriched_rows.append(row_dict)

print(f"Enriched rows collected: {len(enriched_rows)}")
enriched_df = pd.DataFrame(enriched_rows)
# Save
enriched_df.to_csv(OUT_PATH, index=False)
print(f"Saved enriched dataset to {OUT_PATH}")
print("Years:", sorted(enriched_df['year'].unique()))
print("Teams:", enriched_df['team'].nunique())
print("Rounds:", sorted(enriched_df['round'].unique())[:10])
