"""
Model calibration for probability predictions.
Ensures predicted probabilities match true frequencies (critical for betting EV calculations).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import os

MODELS_DIR = "models"
CALIBRATED_DIR = "models/calibrated"
os.makedirs(CALIBRATED_DIR, exist_ok=True)

class PlattCalibrator:
    """Wraps a base classifier and applies Platt scaling (logistic) on its probabilities."""
    def __init__(self, base_classifier, logistic_model):
        self.base = base_classifier
        self.lr = logistic_model
    def predict_proba(self, X):
        base_proba = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        pos = self.lr.predict_proba(base_proba)[:, 1]
        # return 2-class array (n_samples, 2)
        return np.vstack([1 - pos, pos]).T

class IsotonicCalibrator:
    """Wraps a base classifier and applies isotonic regression calibration."""
    def __init__(self, base_classifier, isotonic_regressor):
        self.base = base_classifier
        self.ir = isotonic_regressor
    def predict_proba(self, X):
        base_proba = self.base.predict_proba(X)[:, 1]
        pos = self.ir.transform(base_proba)
        pos = np.clip(pos, 0, 1)
        return np.vstack([1 - pos, pos]).T

def load_models():
    """Load base models."""
    models = {}
    for target in ['over_195', 'over_245', 'over_295']:
        path = os.path.join(MODELS_DIR, f'model_{target}.pkl')
        if os.path.exists(path):
            models[target] = joblib.load(path)
    return models

def load_data():
    """Load feature dataset."""
    df = pd.read_csv("data/processed/features.csv")
    return df

def prepare_X_y(df, target_col):
    """Prepare features and target."""
    raw_stats = ['kicks', 'handballs', 'disposals', 'marks', 'goals', 'behinds', 'hitouts', 'tackles']
    exclude = ['year', 'player', 'team', 'game_order'] + raw_stats
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('over_')]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col]
    return X, y, feature_cols

def calibrate_platt(base_model, X_val, y_val):
    """Platt scaling: fit logistic regression on base model's validation predictions."""
    val_proba = base_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(val_proba, y_val)
    return PlattCalibrator(base_model, lr)

def calibrate_isotonic(base_model, X_val, y_val):
    """Isotonic regression calibration (non-parametric)."""
    val_proba = base_model.predict_proba(X_val)[:, 1]
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_proba, y_val)
    return IsotonicCalibrator(base_model, ir)

def main():
    print("Loading data and base models...")
    models = load_models()
    df = load_data()
    results = {}

    for target, model in models.items():
        print(f"\n=== Calibrating {target} ===")
        X, y, feature_cols = prepare_X_y(df, target)
        # Split: train on historical older years, calibrate on recent years
        years = sorted(df['year'].unique())
        calib_years = years[-2:]  # use last 2 years for calibration
        train_mask = ~df['year'].isin(calib_years)
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[~train_mask]
        y_val = y[~train_mask]

        # Platt scaling (sigmoid)
        calibrated_platt = calibrate_platt(model, X_val, y_val)
        # Isotonic (optional, more flexible)
        # calibrated_iso = calibrate_isotonic(model, X_val, y_val)

        # Evaluate
        val_proba_base = model.predict_proba(X_val)[:,1]
        base_auc = roc_auc_score(y_val, val_proba_base)
        base_brier = brier_score_loss(y_val, val_proba_base)
        calib_proba = calibrated_platt.predict_proba(X_val)[:,1]
        calib_auc = roc_auc_score(y_val, calib_proba)
        calib_brier = brier_score_loss(y_val, calib_proba)
        print(f"Base:    AUC={base_auc:.4f}, Brier={base_brier:.4f}")
        print(f"Calibrated (Platt): AUC={calib_auc:.4f}, Brier={calib_brier:.4f}")

        # Save
        out_path = os.path.join(CALIBRATED_DIR, f'{target}_calibrated_platt.pkl')
        joblib.dump(calibrated_platt, out_path)
        print(f"Saved calibrated model to {out_path}")
        results[target] = {
            'base_auc': base_auc,
            'base_brier': base_brier,
            'calibrated_auc': calib_auc,
            'calibrated_brier': calib_brier,
        }

    # Save summary
    summary = pd.DataFrame(results).T
    summary.to_csv(os.path.join(CALIBRATED_DIR, 'calibration_summary.csv'))
    print("\nCalibration complete. Summary:")
    print(summary)

if __name__ == "__main__":
    main()
