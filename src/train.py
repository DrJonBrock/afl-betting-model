"""
Train AFL betting models on real data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure sorted for rolling calculations
    df = df.sort_values(['player', 'year', 'game_order']).reset_index(drop=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling statistics for each player (lagged, excluding current game)."""
    df = df.copy().sort_values(['player', 'year', 'game_order'])
    # Rolling windows - use shift(1) to exclude current game
    for window in [5, 10, 20]:
        for stat in ['kicks', 'handballs', 'marks', 'disposals', 'goals', 'tackles']:
            # Rolling mean of past N games
            rolling = df.groupby('player')[stat].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'{stat}_last_{window}'] = rolling.groupby(df['player']).shift(1).fillna(0)
        # Disposals std of past N games
        rolling_std = df.groupby('player')['disposals'].rolling(window, min_periods=1).std().reset_index(0, drop=True)
        df[f'disposals_std_{window}'] = rolling_std.groupby(df['player']).shift(1).fillna(0)
    # Previous game value as lag
    df['disposals_prev'] = df.groupby('player')['disposals'].shift(1).fillna(0)
    return df

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary targets for over/under thresholds."""
    thresholds = [19.5, 24.5, 29.5]
    for t in thresholds:
        df[f'over_{int(t*10)}'] = (df['disposals'] > t).astype(int)
    return df

TARGET_COLUMNS = ['over_195', 'over_245', 'over_295']

def train_models(X: pd.DataFrame, y: pd.DataFrame):
    """Train classifiers for each target."""
    models = {}
    # Feature columns are those already in X (caller ensures X excludes raw stats)
    feature_cols = list(X.columns)
    logger.info(f"Using {len(feature_cols)} features")

    for target in TARGET_COLUMNS:
        logger.info(f"Training model for {target}")
        X_train, X_val, y_train, y_val = train_test_split(
            X[feature_cols], y[target], test_size=0.2, random_state=42, stratify=y[target]
        )
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:,1]
        roc = roc_auc_score(y_val, y_proba)
        pr = average_precision_score(y_val, y_proba)
        logger.info(f"  ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}")
        logger.info(f"\n{classification_report(y_val, y_pred, digits=4)}")
        models[target] = clf
        # Save individual model
        joblib.dump(clf, os.path.join(MODELS_DIR, f'model_{target}.pkl'))

    # Also save a combined multi-output model wrapper
    joblib.dump(models, os.path.join(MODELS_DIR, 'models_all.pkl'))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    logger.info("All models saved.")
    return models, feature_cols

def main():
    logger.info("Loading player-game data...")
    df = load_data(os.path.join(PROCESSED_DIR, "player_game_stats.csv"))
    logger.info(f"Loaded {len(df)} rows")
    logger.info("Adding features...")
    df = add_features(df)
    logger.info("Creating targets...")
    df = create_targets(df)
    # Save feature dataset
    df.to_csv(os.path.join(PROCESSED_DIR, "features.csv"), index=False)
    logger.info("Features saved to data/processed/features.csv")
    # Prepare X/y: exclude raw stats and identifiers
    raw_stats = ['kicks', 'handballs', 'disposals', 'marks', 'goals', 'behinds', 'hitouts', 'tackles']
    exclude = ['year', 'player', 'team', 'game_order'] + raw_stats
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('over_')]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[TARGET_COLUMNS]
    logger.info(f"Training on {len(X)} samples, {len(feature_cols)} features")
    models, cols = train_models(X, y)
    # Save summary
    summary = {
        'n_samples': len(X),
        'n_features': len(cols),
        'targets': TARGET_COLUMNS,
        'feature_columns': cols,
    }
    joblib.dump(summary, os.path.join(MODELS_DIR, 'training_summary.pkl'))
    logger.info("Training complete.")

if __name__ == "__main__":
    main()