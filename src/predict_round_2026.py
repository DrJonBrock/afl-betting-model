"""
Generate predictions for a specific 2026 AFL round using the trained context model.
Usage: python predict_round_2026.py <round_number>
Outputs CSV with probabilities for all player-rounds in that round.
"""
import pandas as pd
import numpy as np
import joblib
import sys
import os

def load_data():
    # Load historical (2020-2025) with context and 2026 with context
    hist = pd.read_csv('data/processed/hist_with_context.csv')
    hist['year'] = hist['year'].astype(int)
    hist['round'] = hist['round'].astype(int)
    current = pd.read_csv('data/processed/2026_with_context.csv')
    current['year'] = 2026
    current['round'] = current['round'].astype(int)
    return hist, current

def add_rolling_features(df):
    """Add lagged rolling stats for disposals."""
    df = df.sort_values(['player','year','round']).reset_index(drop=True)
    for window in [5,10,20]:
        rolling_mean = df.groupby('player')['disposals'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'disposals_last_{window}'] = rolling_mean.groupby(df['player']).shift(1).fillna(0)
        rolling_std = df.groupby('player')['disposals'].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
        df[f'disposals_std_{window}'] = rolling_std.groupby(df['player']).shift(1).fillna(0)
    df['disposals_prev'] = df.groupby('player')['disposals'].shift(1).fillna(0)
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_round_2026.py <round_number>")
        sys.exit(1)
    target_round = int(sys.argv[1])

    # Load and combine
    hist, current = load_data()
    all_data = pd.concat([hist, current], ignore_index=True)
    print(f"Combined data: {len(all_data)} rows")

    # Add rolling features
    all_data = add_rolling_features(all_data)

    # Filter to target round in 2026
    round_data = all_data[(all_data['year']==2026) & (all_data['round']==target_round)].copy()
    if round_data.empty:
        print(f"No data found for 2026 round {target_round}")
        sys.exit(1)

    # Load models and feature cols
    models = joblib.load('models/context/models_context.pkl')
    feature_cols = joblib.load('models/context/features.pkl')

    # Build feature matrix
    X = round_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict
    preds = {}
    for target in ['over_195','over_245','over_295']:
        model = models[target]
        probs = model.predict_proba(X)[:,1]
        preds[f'prob_{target}'] = probs
        round_data[f'prob_{target}'] = probs
        round_data[f'pred_{target}'] = model.predict(X)

    # Save selected columns
    out_cols = ['year','round','player','team','opponent','venue','disposals',
                'prob_over_195','pred_over_195',
                'prob_over_245','pred_over_245',
                'prob_over_295','pred_over_295']
    out_df = round_data[out_cols].copy()
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(round_data)} predictions to {out_path}")
    print("Columns:", round_data.columns.tolist())

if __name__ == "__main__":
    main()
