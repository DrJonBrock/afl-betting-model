"""
Backtest on 2026 season using clean historical data (2020-2025) with context features.
Train on 2020-2025 (86k rows), test on 2026 rounds 1-3 (15 teams, 249 rows).
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def load_clean_hist():
    df = pd.read_csv('data/processed/hist_with_context.csv')
    df['year'] = df['year'].astype(int)
    df['round'] = df['round'].astype(int)
    return df

def load_2026_context():
    df = pd.read_csv('data/processed/2026_with_context.csv')
    df['year'] = 2026
    df['round'] = df['round'].astype(int)
    return df

def add_features(df):
    """Add lagged rolling stats. Only disposals available in current dataset."""
    df = df.copy().sort_values(['player','year','round']).reset_index(drop=True)
    # Rolling features for disposals only
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
    logger.info("Loading clean historical data (2020-2025)...")
    hist = load_clean_hist()
    logger.info(f"Historical rows: {len(hist)}")
    logger.info("Loading 2026 context data...")
    df2026 = load_2026_context()
    logger.info(f"2026 rows: {len(df2026)}")
    # Combine for feature engineering (so rolling stats can use prior years)
    all_data = pd.concat([hist, df2026], ignore_index=True)
    logger.info(f"Combined rows: {len(all_data)}")
    # Add rolling features
    logger.info("Adding rolling features...")
    all_data = add_features(all_data)
    # Add targets
    all_data = create_targets(all_data)
    # Split train/test: train on 2020-2025, test on 2026 all rounds
    train = all_data[all_data['year'].isin([2020,2021,2022,2023,2024,2025])].copy()
    test = all_data[(all_data['year']==2026)].copy()
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
    # Feature columns: exclude identifiers, raw target, and odds (not beneficial)
    raw_stats = ['disposals','year','player','team','round','opponent','venue','total_score_close']
    feature_cols = [c for c in all_data.columns if c not in raw_stats and not c.startswith('over_')]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    X_train = train[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
    y_train = train[['over_195','over_245','over_295']]
    X_test = test[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
    y_test = test[['over_195','over_245','over_295']]
    # Train models (use simpler to speed up)
    from sklearn.ensemble import RandomForestClassifier
    models = {}
    for target in ['over_195','over_245','over_295']:
        logger.info(f"Training {target} on {len(X_train)} samples")
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train[target])
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test[target], y_proba)
        pr = average_precision_score(y_test[target], y_proba)
        logger.info(f"  ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}")
        logger.info(f"\n{classification_report(y_test[target], y_pred, digits=4)}")
        models[target] = clf
    # Save models and predictions
    outdir = "models/context"
    os.makedirs(outdir, exist_ok=True)
    joblib.dump(models, os.path.join(outdir, 'models_context.pkl'))
    joblib.dump(feature_cols, os.path.join(outdir, 'features.pkl'))
    pred_df = test[['year','player','team','round','disposals']].copy()
    for target in ['over_195','over_245','over_295']:
        pred_df[f'prob_{target}'] = models[target].predict_proba(X_test)[:,1]
        pred_df[f'pred_{target}'] = models[target].predict(X_test)
        pred_df[f'actual_{target}'] = y_test[target].values
    pred_df.to_csv('outputs/2026_predictions_context.csv', index=False)
    logger.info("Context models and predictions saved.")

if __name__ == "__main__":
    main()
