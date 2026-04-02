"""
Backtest on 2026 season: train on 2020-2025, predict on 2026 rounds 1-3, evaluate.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def load_historical():
    """Load 2020-2025 per-game stats."""
    df = pd.read_csv('data/processed/player_game_stats.csv')
    df = df.rename(columns={'game_order': 'round'})[['year','player','team','round','kicks','handballs','marks','disposals','goals','behinds','hitouts','tackles']].copy()
    df['round'] = df['round'].astype(int)
    return df

def load_2026():
    """Load 2026 per-round stats."""
    df = pd.read_csv('data/raw/afltables_2026_full.csv')
    df['round'] = df['round'].astype(int)
    # Add missing columns
    for col in ['behinds','hitouts']:
        if col not in df.columns:
            df[col] = 0
    df = df[['year','player','team','round','kicks','handballs','marks','disposals','goals','behinds','hitouts','tackles']].copy()
    return df

def add_features(df):
    """Compute rolling statistics for each player (lagged, excluding current game)."""
    df = df.copy().sort_values(['player', 'year', 'round']).reset_index(drop=True)
    for window in [5, 10, 20]:
        for stat in ['kicks','handballs','marks','disposals','goals','tackles']:
            rolling = df.groupby('player')[stat].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'{stat}_last_{window}'] = rolling.groupby(df['player']).shift(1).fillna(0)
        rolling_std = df.groupby('player')['disposals'].rolling(window, min_periods=1).std().reset_index(0, drop=True)
        df[f'disposals_std_{window}'] = rolling_std.groupby(df['player']).shift(1).fillna(0)
    df['disposals_prev'] = df.groupby('player')['disposals'].shift(1).fillna(0)
    return df

def create_targets(df):
    thresholds = [19.5, 24.5, 29.5]
    for t in thresholds:
        df[f'over_{int(t*10)}'] = (df['disposals'] > t).astype(int)
    return df

def main():
    logger.info("Loading historical data (2020-2025)...")
    hist = load_historical()
    logger.info(f"Historical rows: {len(hist)}")
    logger.info("Loading 2026 data...")
    new2026 = load_2026()
    logger.info(f"2026 rows: {len(new2026)}")
    # Combine
    all_data = pd.concat([hist, new2026], ignore_index=True)
    logger.info(f"Total rows combined: {len(all_data)}")
    # Features and targets
    logger.info("Adding features...")
    all_data = add_features(all_data)
    logger.info("Creating targets...")
    all_data = create_targets(all_data)
    # Save full feature matrix for inspection
    all_data.to_csv('data/processed/features_with_2026.csv', index=False)
    # Isolate 2026 test set: rounds 1,2,3 only
    df2026 = all_data[all_data['year'] == 2026].copy()
    test_rounds = [1,2,3]
    df_test = df2026[df2026['round'].isin(test_rounds)].copy()
    logger.info(f"Test set: {len(df_test)} player-rounds from rounds {test_rounds}")
    # Load models and feature columns
    models = joblib.load('models/models_all.pkl')
    feature_cols = joblib.load('models/feature_columns.pkl')
    # Prepare X_test (exclude raw stats)
    raw_stats = ['kicks','handballs','disposals','marks','goals','behinds','hitouts','tackles']
    exclude = ['year','player','team','round'] + raw_stats
    feature_cols_clean = [c for c in feature_cols if c not in exclude]
    X_test = df_test[feature_cols_clean].replace([np.inf,-np.inf], np.nan).fillna(0)
    y_true = df_test[['over_195','over_245','over_295']]
    # Evaluate each target
    results = {}
    for target in ['over_195','over_245','over_295']:
        model = models[target]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_true[target], y_proba)
        pr = average_precision_score(y_true[target], y_proba)
        logger.info(f"\n=== {target} (2026 Rounds {test_rounds}) ===")
        logger.info(f"ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}")
        logger.info(f"\n{classification_report(y_true[target], y_pred, digits=4)}")
        results[target] = {'roc_auc': roc, 'pr_auc': pr}
    # Save results and predictions
    joblib.dump(results, 'models/backtest_2026_rounds1-3_results.pkl')
    pred_df = df_test[['year','player','team','round','disposals']].copy()
    for target in ['over_195','over_245','over_295']:
        pred_df[f'prob_{target}'] = models[target].predict_proba(X_test)[:,1]
        pred_df[f'pred_{target}'] = models[target].predict(X_test)
        pred_df[f'actual_{target}'] = y_true[target].values
    pred_df.to_csv('outputs/2026_predictions_rounds1-3.csv', index=False)
    logger.info("Results and predictions saved.")

if __name__ == "__main__":
    main()
