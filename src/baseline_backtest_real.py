"""
Baseline backtest: NO context features, only rolling disposals.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def load_hist():
    df = pd.read_csv('data/processed/player_game_stats_clean.csv')
    df['year'] = df['year'].astype(int)
    df['round'] = df['round'].astype(int)
    return df

def load_2026_raw():
    df = pd.read_csv('data/raw/afl2026_matches_enriched.csv')
    df['year'] = 2026
    df['round'] = df['round'].astype(int)
    return df

def add_rolling_features(df):
    df = df.copy().sort_values(['player','year','round']).reset_index(drop=True)
    for window in [5,10,20]:
        rolling_mean = df.groupby('player')['disposals'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'disposals_last_{window}'] = rolling_mean.groupby(df['player']).shift(1).fillna(0)
        rolling_std = df.groupby('player')['disposals'].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
        df[f'disposals_std_{window}'] = rolling_std.groupby(df['player']).shift(1).fillna(0)
    df['disposals_prev'] = df.groupby('player')['disposals'].shift(1).fillna(0)
    return df

def create_targets(df):
    thresholds = [19.5, 24.5, 29.5]
    for t in thresholds:
        df[f'over_{int(t*10)}'] = (df['disposals'] > t).astype(int)
    return df

def main():
    logger.info("Loading historical (2020-2025) and 2026 raw data...")
    hist = load_hist()
    df2026 = load_2026_raw()
    all_data = pd.concat([hist, df2026], ignore_index=True)
    logger.info(f"Combined rows: {len(all_data)}")
    logger.info("Adding rolling features (disposals only)...")
    all_data = add_rolling_features(all_data)
    all_data = create_targets(all_data)
    # Split
    train = all_data[all_data['year'].isin([2020,2021,2022,2023,2024,2025])].copy()
    test = all_data[(all_data['year']==2026) & (all_data['round'].isin([1,2,3,4]))].copy()
    logger.info(f"Train: {len(train)}, Test: {len(test)}")
    # Features: rolling disposals only, exclude raw
    raw_stats = ['disposals','year','player','team','round','opponent','venue','is_home']
    feature_cols = [c for c in all_data.columns if c not in raw_stats and not c.startswith('over_')]
    logger.info(f"Features: {feature_cols}")
    X_train = train[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
    y_train = train[['over_195','over_245','over_295']]
    X_test = test[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
    y_test = test[['over_195','over_245','over_295']]
    # Train
    from sklearn.ensemble import RandomForestClassifier
    models = {}
    for target in ['over_195','over_245','over_295']:
        logger.info(f"Training {target}")
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train[target])
        y_proba = clf.predict_proba(X_test)[:,1]
        y_pred = clf.predict(X_test)
        roc = roc_auc_score(y_test[target], y_proba)
        pr = average_precision_score(y_test[target], y_proba)
        logger.info(f"  ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}")
        logger.info(f"\n{classification_report(y_test[target], y_pred, digits=4)}")
        models[target] = clf
    # Save
    outdir = "models/baseline_real"
    os.makedirs(outdir, exist_ok=True)
    joblib.dump(models, os.path.join(outdir, 'models_baseline.pkl'))
    preds = test[['year','player','team','round','disposals']].copy()
    for t in ['over_195','over_245','over_295']:
        preds[f'prob_{t}'] = models[t].predict_proba(X_test)[:,1]
        preds[f'pred_{t}'] = models[t].predict(X_test)
        preds[f'actual_{t}'] = y_test[t].values
    preds.to_csv('outputs/2026_predictions_baseline_real.csv', index=False)
    logger.info("Baseline (no context) predictions saved.")

if __name__ == "__main__":
    main()
