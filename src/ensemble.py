"""
Ensemble model using stacking: RF + XGBoost + LogisticRegression meta-learner.
Creates a robust probability estimator for betting value detection.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import os

MODELS_DIR = "models"
ENSEMBLE_DIR = "models/ensemble"
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv("data/processed/features.csv")
    return df

def prepare_X_y(df, target_col):
    raw_stats = ['kicks', 'handballs', 'disposals', 'marks', 'goals', 'behinds', 'hitouts', 'tackles']
    exclude = ['year', 'player', 'team', 'game_order'] + raw_stats
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('over_')]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col]
    return X, y, feature_cols

def create_base_models():
    """Define diverse base learners."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    lr = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    return [('rf', rf), ('xgb', xgb_model), ('lr', lr)]

def train_stacking_ensemble(X_train, y_train, X_val, y_val):
    """Two-level stacking: base models predict, meta-learner combines."""
    base_models = create_base_models()
    # Train base models on training set
    base_fitted = []
    val_preds = np.zeros((len(X_val), len(base_models)))
    for idx, (name, model) in enumerate(base_models):
        print(f"Training base model: {name}")
        model.fit(X_train, y_train)
        base_fitted.append((name, model))
        val_preds[:, idx] = model.predict_proba(X_val)[:, 1]

    # Train meta-learner on base predictions (cross-validated to avoid leakage)
    meta = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    meta.fit(val_preds, y_val)

    # Evaluate ensemble on validation
    meta_proba = meta.predict_proba(val_preds)[:, 1]
    auc = roc_auc_score(y_val, meta_proba)
    pr = average_precision_score(y_val, meta_proba)
    print(f"Stacking Ensemble: ROC-AUC={auc:.4f}, PR-AUC={pr:.4f}")
    print("Meta coefficients:", meta.coef_)

    # Also test simple voting for comparison
    voting = VotingClassifier(
        estimators=base_models,
        voting='soft'
    )
    voting.fit(X_train, y_train)
    vote_proba = voting.predict_proba(X_val)[:, 1]
    vote_auc = roc_auc_score(y_val, vote_proba)
    vote_pr = average_precision_score(y_val, vote_proba)
    print(f"Voting Ensemble: ROC-AUC={vote_auc:.4f}, PR-AUC={vote_pr:.4f}")

    return {
        'stacking': (base_fitted, meta),
        'voting': voting,
        'metrics': {
            'stacking_auc': auc,
            'stacking_pr': pr,
            'voting_auc': vote_auc,
            'voting_pr': vote_pr
        }
    }

def main():
    print("Loading data...")
    df = load_data()
    results = {}

    for target in ['over_195', 'over_245', 'over_295']:
        print(f"\n=== Building Ensemble for {target} ===")
        X, y, feature_cols = prepare_X_y(df, target)
        # Time-based split: last 2 years validation
        years = sorted(df['year'].unique())
        val_years = years[-2:]
        train_mask = ~df['year'].isin(val_years)
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[~train_mask], y[~train_mask]

        ensemble = train_stacking_ensemble(X_train, y_train, X_val, y_val)
        results[target] = ensemble['metrics']

        # Save ensemble
        out_path = os.path.join(ENSEMBLE_DIR, f'ensemble_{target}.pkl')
        joblib.dump(ensemble, out_path)
        print(f"Saved ensemble to {out_path}")

    # Summary
    summary = pd.DataFrame(results).T
    print("\n=== Ensemble Summary ===")
    print(summary)
    summary.to_csv(os.path.join(ENSEMBLE_DIR, 'ensemble_summary.csv'))

if __name__ == "__main__":
    main()
