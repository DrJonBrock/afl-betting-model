"""
Build contextual features for AFL betting model.
- Bye indicator: did team have a bye in previous round?
- Venue average disposals: historical average total disposals per player at that venue
- Opponent defensive strength: average disposals allowed by opponent to opposing players (complex, simplified as team avg disposals allowed)
"""
import pandas as pd
import numpy as np

def add_bye_feature(df):
    """Add column 'had_bye_last_week' (0/1) for each team-round."""
    # Determine which rounds each team did NOT play (byes)
    # For each team, find rounds where they have no record
    all_teams = df['team'].unique()
    all_rounds = sorted(df['round'].unique())
    # Build set of (team, round) that exist
    existing = set(zip(df['team'], df['round']))
    # For each row, check if (team, round-1) was missing
    df['had_bye_last_week'] = 0
    for idx, row in df.iterrows():
        team = row['team']
        rnd = int(row['round'])
        prev = rnd - 1
        if prev < min(all_rounds):
            continue
        if (team, prev) not in existing:
            df.at[idx, 'had_bye_last_week'] = 1
    return df

def add_venue_averages(df, hist_df):
    """Compute historical average disposals per player at each venue (from hist_df) and add as feature."""
    # Use hist_df (2020-2025) to compute venue means
    venue_stats = hist_df.groupby('venue').agg(
        avg_disposals=('disposals', 'mean'),
        count=('disposals', 'count')
    ).reset_index()
    # Merge to df on venue
    df = df.merge(venue_stats[['venue','avg_disposals']], on='venue', how='left')
    df = df.rename(columns={'avg_disposals': 'venue_avg_disposals'})
    return df

def add_opponent_strength(df, hist_df):
    """Add opponent's historical average disposals (team-level) as a proxy for opponent strength."""
    team_avg = hist_df.groupby('team').agg(
        opponent_avg_disposals=('disposals', 'mean')
    ).reset_index()
    # Convert to dict for fast map
    avg_dict = team_avg.set_index('team')['opponent_avg_disposals'].to_dict()
    df['opponent_avg_disposals'] = df['opponent'].map(avg_dict)
    return df

def main():
    # Load historical data (2020-2025 per-game) and 2026 enriched
    hist = pd.read_csv('data/processed/player_game_stats.csv')
    hist = hist.rename(columns={'game_order': 'round'})[['year','player','team','round','kicks','handballs','marks','disposals','goals','behinds','hitouts','tackles']].copy()
    df2026 = pd.read_csv('data/raw/afl2026_matches_enriched.csv')
    # Rename team_x to team for consistency
    if 'team_x' in df2026.columns:
        df2026 = df2026.rename(columns={'team_x': 'team'})
    # Add bye feature
    df2026 = add_bye_feature(df2026)
    # Add opponent strength (using historical team averages)
    df2026 = add_opponent_strength(df2026, hist)
    # Save
    df2026.to_csv('data/processed/2026_with_context.csv', index=False)
    print("Added bye and opponent strength features.")
    print(df2026[['round','team','had_bye_last_week','opponent_avg_disposals']].head(10))
    print("Total rows:", len(df2026))

if __name__ == "__main__":
    main()
