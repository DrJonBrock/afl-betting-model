"""
Generate predictions for the next AFL round.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
from src.models import FEATURE_COLS

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_latest_features():
    """Load feature table and take most recent date/round as upcoming."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
    # If 'player' column exists, sort; otherwise skip sort
    if 'player' in df.columns:
        df = df.sort_values(['player', 'year', 'game_order']).reset_index(drop=True)
    return df

def predict_next_round(df: pd.DataFrame, line: float = 24.5):
    line_str = str(line).replace('.', '_')
    model_path = os.path.join(MODELS_DIR, f"disposals_over_{line_str}.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    model = joblib.load(model_path)
    # For each player, take their last row (most recent stats) and compute rolling features
    # In real system, we'd have a separate feature build for upcoming matchups; here we reuse last row
    players_last = df.groupby('player').tail(1).copy()
    X = players_last[FEATURE_COLS].fillna(0)
    p_over = model.predict_proba(X)[:, 1]
    players_last['p_over'] = p_over
    players_last['fair_odds_over'] = 1 / p_over
    # Add line label
    players_last['line'] = line
    return players_last[['player', 'year', 'game_order', 'p_over', 'fair_odds_over', 'line']]

def main():
    df = load_latest_features()
    predictions = []
    for line in [19.5, 24.5, 29.5]:
        pred = predict_next_round(df, line)
        if pred is not None:
            predictions.append(pred)
    all_preds = pd.concat(predictions, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "predictions_next_round.csv")
    all_preds.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path} ({len(all_preds)} rows).")
    # Print top value candidates (fair odds > 2.0 as example)
    value = all_preds[all_preds['fair_odds_over'] > 2.0]
    print(f"Players with fair odds > 2.0: {len(value)}")
    if not value.empty:
        print(value.head(20).to_string(index=False))

if __name__ == "__main__":
    main()