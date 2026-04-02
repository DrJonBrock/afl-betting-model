"""
Merge match total over/under odds from AFL Data Analysis repo to player-game dataset.
Creates feature: total_score_over_close (the closing total points line for the match).
"""
import pandas as pd
import os

# Load odds data
odds_path = 'external/afl-data-analysis/odds_data/odds_data_2009_to_present.xlsx'
print(f"Loading odds from {odds_path}")
odds = pd.read_excel(odds_path)

# Keep relevant columns: Date, Home Team, Away Team, Total Score Over Close
# The column might have a weird name; let's find it
cols = odds.columns.tolist()
total_over_col = [c for c in cols if 'Total Score Over Close' in c][0]
print(f"Using column: {total_over_col}")

odds_match = odds[['Date', 'Home Team', 'Away Team', total_over_col]].copy()
odds_match = odds_match.rename(columns={total_over_col: 'total_score_over_close'})

# Extract year and round from Date? The odds dataset might not have round. Check structure.
print("Sample odds rows:")
print(odds_match.head())

# We need round number. Look at the raw data directory to see if there's a matches file with round info.
# Alternatively, we can match on date+teams to our fixture mapping, which has round.
# Let's load our fixture mappings (2020-2026) to create a match key (year, round, home, away)
fixture_frames = []
for year in range(2020, 2027):
    map_file = f'data/raw/fixture_mapping_{year}.csv'
    if os.path.exists(map_file):
        df_map = pd.read_csv(map_file)
        df_map['year'] = year
        # From mapping we can reconstruct home/away: we have rows for each team with opponent and is_home.
        # Build unique matches: (year, round, home, away) where home is team with is_home=1
        for (yr, rnd), grp in df_map.groupby(['year','round']):
            # Find home team
            home_row = grp[grp['is_home']==1]
            if not home_row.empty:
                home_team = home_row.iloc[0]['team']
                away_team = home_row.iloc[0]['opponent']
                fixture_frames.append({
                    'year': yr,
                    'round': rnd,
                    'home_team': home_team,
                    'away_team': away_team,
                    'venue': home_row.iloc[0]['venue']
                })
        # Note: some rounds may have multiple games; but our grouping yields one per (year,round,team) which is double counting. Actually, mapping has two rows per match. So we need to dedupe by (year,round,team) pairs that represent same match. The above will produce two entries per match if we loop through all teams? Let's think: groupby year, round yields all matches in that round? Actually each (year, round) group contains rows for all teams playing that round. So we will get multiple home/away pairs. Better approach: iterate through each unique game in df_map by creating a composite key. Let's do it properly:
pass

# Let's instead load our pre-built match mapping: from data/processed/player_game_stats.csv we can extract unique match identifiers?
df_pg = pd.read_csv('data/processed/player_game_stats.csv')
# It contains player, team, round, year, opponent, venue. We can create a match_id by combining (year, round, team, opponent) but that is player-specific. Unique matches: each combination of (year, round, team, opponent) corresponds to a match, but both home and away appear. We can define a canonical key: (year, round, sorted_teams). But easier: join directly on year, round, and both team combinations.

# For now, let's just explore the odds dataset more: what years are covered?
odds_match['Date'] = pd.to_datetime(odds_match['Date'])
odds_match['year'] = odds_match['Date'].dt.year
print("Odds years covered:", sorted(odds_match['year'].unique()))
print("Sample rows with teams:")
print(odds_match[['year','Home Team','Away Team','total_score_over_close']].head(10))

# We'll need to map team names to our standard. Let's print unique team names from odds.
home_teams = odds_match['Home Team'].unique()
away_teams = odds_match['Away Team'].unique()
all_odds_teams = set(home_teams) | set(away_teams)
print("Odds teams count:", len(all_odds_teams))
print("Sample teams:", sorted(all_odds_teams)[:20])
