"""
Transform raw historical full-stats data into player-game dataset with opponent/venue context.
"""
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Load combined full-stats data
    logger.info("Loading afltables_historical_full.csv")
    df_full = pd.read_csv(os.path.join(raw_dir, "afltables_historical_full.csv"))
    # Keep only round stat rows (stat_type like R1, R2, ...) and value = disposals
    df_rounds = df_full[df_full['stat_type'].astype(str).str.startswith('R')].copy()
    df_rounds['round_num'] = df_rounds['stat_type'].str.extract(r'R(\d+)').astype(int)
    # Keep relevant columns
    df_pg = df_rounds[['year', 'player', 'team', 'round_num', 'value']].rename(columns={'value': 'disposals', 'round_num': 'round'})

    # For 2026, we already have a separate gbg dataset that is already in correct format
    df_2026 = pd.read_csv(os.path.join(raw_dir, "afltables_2026_gbg.csv"))
    # Normalize column names to match
    df_2026 = df_2026.rename(columns={'disposals': 'disposals'})[['year', 'player', 'team', 'round', 'disposals']]

    # Combine historical (2000-2025) with 2026
    df_all = pd.concat([df_pg, df_2026], ignore_index=True)

    # Load fixture mappings for 2010-2026
    fixture_frames = []
    for year in range(2010, 2027):
        map_file = os.path.join(raw_dir, f"fixture_mapping_{year}.csv")
        if os.path.exists(map_file):
            df_map = pd.read_csv(map_file)
            df_map['year'] = year
            fixture_frames.append(df_map)
        else:
            logger.warning(f"Missing fixture mapping for {year}")
    df_fixture = pd.concat(fixture_frames, ignore_index=True)

    # Merge player-game with fixture to get opponent, venue, is_home
    df_merged = pd.merge(
        df_all,
        df_fixture,
        how='inner',
        left_on=['year', 'team', 'round'],
        right_on=['year', 'team', 'round']
    )

    # Merge odds total score lines
    # Determine project root (src's parent)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    odds_path = os.path.join(project_root, 'external', 'afl-data-analysis', 'odds_data', 'odds_data_2009_to_present.xlsx')
    if os.path.exists(odds_path):
        odds = pd.read_excel(odds_path)
        odds['year'] = pd.to_datetime(odds['Date']).dt.year
        odds = odds.rename(columns={'Total Score Close': 'total_score_close'})
        # Normalize team names
        def normalize_team(name):
            mapping = {'Brisbane Lions':'Brisbane', 'Greater Western Sydney':'GWS Giants', 'Gold Coast Suns':'Gold Coast'}
            return mapping.get(name, name)
        odds['Home Team'] = odds['Home Team'].apply(normalize_team)
        odds['Away Team'] = odds['Away Team'].apply(normalize_team)
        odds_match = odds[['year','Home Team','Away Team','total_score_close']].drop_duplicates(subset=['year','Home Team','Away Team'])
        # Create match identifiers in df_merged: we need home_team and away_team. Use is_home to derive.
        df_merged['home_team'] = df_merged.apply(lambda r: r['team'] if r['is_home']==1 else r['opponent'], axis=1)
        df_merged['away_team'] = df_merged.apply(lambda r: r['opponent'] if r['is_home']==1 else r['team'], axis=1)
        # Apply normalization to these derived columns
        df_merged['home_team'] = df_merged['home_team'].apply(normalize_team)
        df_merged['away_team'] = df_merged['away_team'].apply(normalize_team)
        # Merge odds
        df_merged = pd.merge(
            df_merged,
            odds_match,
            how='left',
            left_on=['year','home_team','away_team'],
            right_on=['year','Home Team','Away Team']
        )
        # Impute missing odds with median
        median_line = df_merged['total_score_close'].median()
        df_merged['total_score_close'] = df_merged['total_score_close'].fillna(median_line)
        logger.info(f"Odds feature added (merged {len(df_merged[~df_merged['total_score_close'].isna()])} rows, median {median_line:.1f})")
        # Drop temporary columns
        df_merged = df_merged.drop(columns=['home_team','away_team','Home Team','Away Team'], errors='ignore')
    else:
        logger.warning("Odds file not found, skipping odds integration")

    # Save final dataset
    out_path = os.path.join(processed_dir, "player_game_stats.csv")
    df_merged.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df_merged)} player-game records to {out_path}")
    logger.info(f"Years: {sorted(df_merged['year'].unique())}")
    logger.info(f"Teams: {df_merged['team'].nunique()}")
    logger.info(f"Players: {df_merged['player'].nunique()}")

    # Also save a version with only 2010-2025 for training (excluding 2026 live backtest)
    df_train = df_merged[df_merged['year'] < 2026].copy()
    train_path = os.path.join(processed_dir, "player_game_stats_train.csv")
    df_train.to_csv(train_path, index=False)
    logger.info(f"Saved training set (2010-2025): {len(df_train)} records to {train_path}")

if __name__ == "__main__":
    main()
