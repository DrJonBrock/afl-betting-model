"""
Evaluate models on a temporal holdout (train on 2020-2022, test on 2023-2024).
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    logger.info("Loading features...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
    # Load feature columns list
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    # Split temporal
    train = df[df['year'].isin([2020,2021,2022])]
    test = df[df['year'].isin([2023,2024])]
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")

    # Prepare X/y
    raw_stats = ['kicks', 'handballs', 'disposals', 'marks', 'goals', 'behinds', 'hitouts', 'tackles']
    exclude = ['year', 'player', 'team', 'game_order'] + raw_stats
    feature_cols_clean = [c for c in feature_cols if c not in exclude]
    X_train = train[feature_cols_clean].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test[feature_cols_clean].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[['over_195','over_245','over_295']]
    y_test = test[['over_195','over_245','over_295']]

    # Load trained models
    models = joblib.load(os.path.join(MODELS_DIR, 'models_all.pkl'))

    results = {}
    for target in ['over_195','over_245','over_295']:
        model = models[target]
        y_true = y_test[target]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_true, y_proba)
        pr = average_precision_score(y_true, y_proba)
        logger.info(f"\n=== {target} ===")
        logger.info(f"Test ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}")
        logger.info(f"\n{classification_report(y_true, y_pred, digits=4)}")
        results[target] = {'roc_auc': roc, 'pr_auc': pr}

    # Save results
    joblib.dump(results, os.path.join(MODELS_DIR, 'temporal_eval.pkl'))
    logger.info("Temporal evaluation saved.")

if __name__ == "__main__":
    import logging
    main()
