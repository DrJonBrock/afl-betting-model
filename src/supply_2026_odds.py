"""
Inject current 2026 match total odds into 2026_with_context.csv (and player_game_stats.csv).
Usage: python supply_2026_odds.py <round_number> <odds_csv>
Odds CSV format: year,round,home_team,away_team,total_score_close
Team names must match those in the context data (use same as fixture mapping).
"""
import pandas as pd
import sys
import os

def merge_into_file(path, odds_df, target_round):
    df = pd.read_csv(path)
    df['year'] = df['year'].astype(int)
    df['round'] = df['round'].astype(int)
    # Determine home_team and away_team from team/opponent/is_home
    if 'is_home' in df.columns and 'team' in df.columns and 'opponent' in df.columns:
        df['home_team'] = df.apply(lambda r: r['team'] if r['is_home']==1 else r['opponent'], axis=1)
        df['away_team'] = df.apply(lambda r: r['opponent'] if r['is_home']==1 else r['team'], axis=1)
    else:
        # Already has home_team/away_team?
        pass
    before = df['total_score_close'].isna().sum() if 'total_score_close' in df.columns else len(df)
    # Ensure column exists
    if 'total_score_close' not in df.columns:
        df['total_score_close'] = None
    # Build mapping from odds
    odds_map = {(r.year, r.round, r.home_team, r.away_team): r.total_score_close for r in odds_df.itertuples()}
    # Update
    def fill_odds(row):
        key = (row['year'], row['round'], row['home_team'], row['away_team'])
        if key in odds_map:
            return odds_map[key]
        return row.get('total_score_close', None)
    df['total_score_close'] = df.apply(fill_odds, axis=1)
    after = df['total_score_close'].isna().sum()
    print(f"File {os.path.basename(path)}: NaN odds before {before}, after {after}")
    # Drop helper columns if we added them
    if 'home_team' in df.columns and 'home_team' not in ['home_team','away_team']:
        # Actually these are temporary, drop if they weren't originally present
        orig_cols = pd.read_csv(path, nrows=1).columns.tolist()
        if 'home_team' not in orig_cols:
            df = df.drop(columns=['home_team','away_team'])
    df.to_csv(path, index=False)

def main():
    if len(sys.argv) < 3:
        print("Usage: python supply_2026_odds.py <round_number> <odds_csv>")
        sys.exit(1)
    target_round = int(sys.argv[1])
    odds_csv = sys.argv[2]
    odds = pd.read_csv(odds_csv)
    odds['year'] = odds['year'].astype(int)
    odds['round'] = odds['round'].astype(int)

    # Update 2026_with_context.csv
    ctx_path = 'data/processed/2026_with_context.csv'
    if os.path.exists(ctx_path):
        merge_into_file(ctx_path, odds, target_round)
    else:
        print(f"Warning: {ctx_path} not found")

    # Also update player_game_stats.csv (needed if retraining)
    pg_path = 'data/processed/player_game_stats.csv'
    if os.path.exists(pg_path):
        merge_into_file(pg_path, odds, target_round)
    else:
        print(f"Warning: {pg_path} not found")

    print("Done. You can now run predictions for round", target_round)

if __name__ == "__main__":
    main()
