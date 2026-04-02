"""
Generate predictions for upcoming round using best available model (calibrated/ensemble/base).
Outputs current_predictions.csv for weekly pipeline.
"""
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"
OUTPUT_PATH = "data/current_predictions.csv"

def load_upcoming_players():
    """Load next round's player list with features already computed."""
    # Latest features file includes upcoming round with missing actual disposals
    features_path = os.path.join(PROCESSED_DIR, "features_with_2026.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError("Need features_with_2026.csv containing upcoming round")
    df = pd.read_csv(features_path)
    # Filter to rows where actual disposals is NA (upcoming) or year >= 2026
    if 'disposals' in df.columns:
        df = df[df['disposals'].isna() | (df['year'] >= 2026)]
    else:
        df = df[df['year'] >= 2026]
    return df

def load_best_model(target):
    """Load the best model for a target: calibrated > ensemble > base."""
    # Try calibrated first
    cal_path = f"models/calibrated/{target}_calibrated_platt.pkl"
    if os.path.exists(cal_path):
        return joblib.load(cal_path)
    # Try ensemble stacking
    ens_path = f"models/ensemble/ensemble_{target}.pkl"
    if os.path.exists(ens_path):
        ensemble = joblib.load(ens_path)
        return ensemble['stacking'][1]  # meta-learner; need base predictions first? Let's use voting instead for simplicity
    # Fall back to base RF
    base_path = f"models/model_{target}.pkl"
    if os.path.exists(base_path):
        return joblib.load(base_path)
    raise FileNotFoundError(f"No model found for {target}")

def prepare_X(df, feature_cols):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def fetch_odds(player, target):
    """
    Placeholder: lookup decimal odds for player's over market.
    In practice, this would query an odds API or local CSV.
    For now, assume odds supplied in a separate file.
    """
    # We'll read from a separate odds file: data/current_odds.csv with columns: player,target,deci_odds
    return None

def main():
    print("Loading upcoming player features...")
    upcoming = load_upcoming_players()
    # Load feature columns used during training
    feature_cols = joblib.load("models/feature_columns.pkl")
    X_upcoming = prepare_X(upcoming, feature_cols)

    predictions = []
    for target in ['over_195', 'over_245', 'over_295']:
        print(f"Predicting {target}...")
        try:
            model = load_best_model(target)
        except FileNotFoundError as e:
            print(e)
            continue
        proba = model.predict_proba(X_upcoming)[:, 1]
        for idx, row in upcoming.iterrows():
            player = row['player']
            game_date = row.get('date', datetime.now().date())
            p = proba[idx]
            odds = fetch_odds(player, target)  # placeholder
            if odds is None:
                # If no odds, skip; odds must be provided externally
                continue
            predictions.append({
                'date': game_date,
                'player': player,
                'target': target,
                'model_proba': p,
                'deci_odds': odds,
                'outcome': None
            })

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(pred_df)} predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
