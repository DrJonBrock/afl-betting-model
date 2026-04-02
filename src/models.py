"""
Model training for AFL betting markets.
Trains XGBoost classifiers for binary over/under markets.
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import yaml

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'disposals_last_5', 'disposals_last_10', 'disposals_last_20',
    'disposals_std_20',
    'kicks_last_5', 'handballs_last_5', 'marks_last_5',
    # Add more as available: minutes, home/away, venue_factor, opponent_def_rank, etc.
]

def load_features():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
    return df

def prepare_data(df: pd.DataFrame, target_col: str = 'target_over_24.5'):
    """Prepare train/test split using time-based holdout."""
    # Split by year: train on older, test on most recent
    years = df['year'].unique()
    years = sorted(years)
    # Use last 2 years as test
    test_years = years[-2:]
    train_years = years[:-2]
    train_df = df[df['year'].isin(train_years)].copy()
    test_df = df[df['year'].isin(test_years)].copy()
    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df[target_col]
    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test, train_df, test_df

def train_model(X_train, y_train, params: dict = None):
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)
    print(f"ROC-AUC: {auc_roc:.4f} | PR-AUC: {auc_pr:.4f}")
    return auc_roc, auc_pr

def main():
    df = load_features()
    # Train for each target line
    results = {}
    for line in [19.5, 24.5, 29.5]:
        target = f'target_over_{line}'
        if target not in df.columns:
            continue
        print(f"\n=== Training model for Disposals Over {line} ===")
        X_train, X_test, y_train, y_test, train_df, test_df = prepare_data(df, target)
        model = train_model(X_train, y_train)
        auc_roc, auc_pr = evaluate_model(model, X_test, y_test)
        line_str = str(line).replace('.', '_')
        model_path = os.path.join(MODELS_DIR, f"disposals_over_{line_str}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")
        results[target] = {'roc_auc': auc_roc, 'pr_auc': auc_pr, 'test_size': len(X_test)}
    # Save results summary
    summary = pd.DataFrame.from_dict(results, orient='index')
    summary.to_csv(os.path.join(MODELS_DIR, "model_summary.csv"))
    print("\nModel training complete. Summary:")
    print(summary)

if __name__ == "__main__":
    main()