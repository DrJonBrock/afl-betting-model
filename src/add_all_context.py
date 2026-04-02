"""
Add contextual features to historical and 2026 data.
- had_bye_last_week
- opponent_avg_disposals (team-level historical average)
- venue_avg_disposals (venue historical average)
"""
import pandas as pd
import numpy as np
import os

def load_clean_hist():
    df = pd.read_csv('data/processed/player_game_stats_clean.csv')
    # Ensure types
    df['year'] = df['year'].astype(int)
    df['round'] = df['round'].astype(int)
    return df

def load_2026():
    df = pd.read_csv('data/raw/afl2026_matches_enriched.csv')
    # Ensure types
    df['year'] = df['year'].astype(int)
    df['round'] = df['round'].astype(int)
    return df

def add_bye_feature(df):
    """Compute had_bye_last_week: 1 if team did not play in previous round."""
    # For each (year,team), sort by round and check gaps
    df = df.copy().sort_values(['year','team','round']).reset_index(drop=True)
    df['had_bye_last_week'] = 0
    for (yr, tm), grp in df.groupby(['year','team']):
        rounds_played = sorted(grp['round'].unique())
        # For each row in group, set flag if (current_round-1) not in rounds_played
        for idx, r in grp['round'].items():
            if (r-1) < min(rounds_played):
                continue
            if (r-1) not in rounds_played:
                df.at[idx, 'had_bye_last_week'] = 1
    return df

def add_team_avg(df, hist_df):
    """Add opponent_avg_disposals from team averages in historical data (2020-2025)."""
    team_avg = hist_df.groupby('team')['disposals'].mean().reset_index()
    team_avg = team_avg.rename(columns={'team':'opponent','disposals':'opponent_avg_disposals'})
    df = df.merge(team_avg, on='opponent', how='left')
    return df

def add_defensive_strength(df, hist_df):
    """Add opponent_defensive_avg: average disposals allowed by opponent to their opponents."""
    # Compute from historical: for each (year, opponent), average disposals of the opposing players.
    # Actually, we need to invert perspective: for each (year, opponent), compute mean of disposals when that opponent is the *team* being faced.
    # In hist_df, each row is a player on a team. The opponent column is the opposing team.
    # So group by (year, opponent) and average disposals to get how many disposals opponents get against that opponent.
    defensive = hist_df.groupby(['year','opponent'])['disposals'].mean().reset_index()
    defensive = defensive.rename(columns={'opponent':'team','disposals':'opponent_defensive_avg'})
    # Merge onto df: for each row, we want the defensive rating of the *opponent* team.
    df = df.merge(defensive, left_on=['year','opponent'], right_on=['year','team'], how='left', suffixes=('','_def'))
    # Drop the extra 'team_def' column
    if 'team_def' in df.columns:
        df = df.drop(columns=['team_def'])
    return df

def add_venue_avg(df, hist_df):
    """Add venue_avg_disposals from venue averages in historical data."""
    venue_avg = hist_df.groupby('venue')['disposals'].mean().reset_index()
    venue_avg = venue_avg.rename(columns={'venue':'venue','disposals':'venue_avg_disposals'})
    df = df.merge(venue_avg, on='venue', how='left')
    return df

def add_venue_climate(df):
    """Add venue climate characteristics (temp, rain, zone)."""
    climate = pd.read_csv('data/raw/venue_climate.csv')
    df = df.merge(climate, on='venue', how='left')
    return df

def main():
    print("Loading clean historical data...")
    hist = load_clean_hist()
    print(f"Historical rows: {len(hist)}")
    print("Loading 2026 data...")
    df2026 = load_2026()
    print(f"2026 rows: {len(df2026)}")
    # Combine for feature computation (use hist to compute aggregates)
    print("Adding bye feature to both sets...")
    hist = add_bye_feature(hist)
    df2026 = add_bye_feature(df2026)
    print("Adding opponent average (from hist)...")
    hist = add_team_avg(hist, hist)  # add to hist itself
    df2026 = add_team_avg(df2026, hist)  # use hist averages on 2026
    print("Adding opponent defensive strength...")
    hist = add_defensive_strength(hist, hist)
    df2026 = add_defensive_strength(df2026, hist)
    print("Adding venue average (from hist)...")
    hist = add_venue_avg(hist, hist)
    df2026 = add_venue_avg(df2026, hist)
    # Save
    hist_out = 'data/processed/hist_with_context.csv'
    df2026_out = 'data/processed/2026_with_context.csv'
    hist.to_csv(hist_out, index=False)
    df2026.to_csv(df2026_out, index=False)
    print(f"Saved historical with context ({len(hist)} rows) to {hist_out}")
    print(f"Saved 2026 with context ({len(df2026)} rows) to {df2026_out}")
    print("Feature columns:", [c for c in hist.columns if c not in ['year','player','team','round','opponent','venue']])

if __name__ == "__main__":
    main()
