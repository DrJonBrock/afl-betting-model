"""
Compute opponent defensive strength: average disposals allowed by each team to their opponents.
This is derived from historical data: for each team in each season, average the disposals of opposing players.
"""
import pandas as pd
import os

# Load historical player-game data (2020-2025)
df = pd.read_csv('data/processed/player_game_stats_clean.csv')
# Ensure we have year, team, opponent, disposals
df = df[['year','team','opponent','disposals']].copy()

# For each (year, team), compute average disposals allowed (i.e., average of opponent's disposals when playing that team)
# Since each row is a player from the team perspective, we need to pivot: for each team, get the mean of disposals of the opposing players.
# Actually, each row already represents a player on a team. The opponent column is the opposing team.
# So we can group by (year, opponent) and compute mean disposals to get how many disposals opponents typically get against that opponent.
defensive = df.groupby(['year','opponent'])['disposals'].mean().reset_index()
defensive = defensive.rename(columns={'opponent':'team', 'disposals':'opponent_defensive_avg'})

# However, we want this as a feature on the player record. For each row (player, team), we want the defensive rating of the *opponent* team.
# So we merge: df.merge(defensive, left_on=['year','opponent'], right_on=['year','team'], suffixes=('','_opp'))
defensive_merged = df.merge(defensive, left_on=['year','opponent'], right_on=['year','team'], how='left')
# Keep only needed cols
defensive_merged = defensive_merged[['year','team_x','opponent','opponent_defensive_avg']].rename(columns={'team_x':'team'})
# There will be duplicates (multiple players per team/opponent). Drop duplicates.
defensive_merged = defensive_merged.drop_duplicates(subset=['year','team','opponent'])

# Save mapping
out_path = 'data/processed/opponent_defensive_avg.csv'
defensive_merged.to_csv(out_path, index=False)
print(f"Saved opponent defensive average mapping: {len(defensive_merged)} rows to {out_path}")
print(defensive_merged.head())
