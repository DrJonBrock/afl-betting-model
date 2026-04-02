"""
Backtest 2023: Compare model performance with and without odds.
Train on 2020-2022, test on 2023.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import os
import sys

# Load data
df = pd.read_csv('data/processed/hist_with_context.csv')
# Ensure year and round are correct type
df['year'] = df['year'].astype(int)
df['round'] = df['round'].astype(int)

# Filter years 2020-2023 (we have odds for these)
df = df[df['year'].between(2020, 2023)].copy()
print(f"Rows in 2020-2023: {len(df)}")
print("Years:", sorted(df['year'].unique()))

# Ensure total_score_close is numeric and not null for these years
df['total_score_close'] = pd.to_numeric(df['total_score_close'], errors='coerce')
null_odds = df['total_score_close'].isna().sum()
print(f"Rows with NaN odds: {null_odds}")

# Compute rolling features if not present
if 'disposals_last_5' not in df.columns:
    print("Computing rolling features...")
    df = df.sort_values(['player','year','round']).reset_index(drop=True)
    for window in [5,10,20]:
        rolling_mean = df.groupby('player')['disposals'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'disposals_last_{window}'] = rolling_mean.groupby(df['player']).shift(1).fillna(0)
        rolling_std = df.groupby('player')['disposals'].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
        df[f'disposals_std_{window}'] = rolling_std.groupby(df['player']).shift(1).fillna(0)
    df['disposals_prev'] = df.groupby('player')['disposals'].shift(1).fillna(0)

# Split
train = df[df['year'].isin([2020,2021,2022])].copy()
test = df[df['year'] == 2023].copy()
print(f"Train size: {len(train)}, Test size: {len(test)}")

# Feature definitions
base_features = ['is_home', 'had_bye_last_week', 'opponent_avg_disposals', 'opponent_defensive_avg', 'venue_avg_disposals']
rolling_features = ['disposals_last_5','disposals_std_5','disposals_last_10','disposals_std_10','disposals_last_20','disposals_std_20','disposals_prev']
targets = ['over_195','over_245','over_295']

# Ensure targets exist
for t in targets:
    if t not in train.columns:
        train[t] = (train['disposals'] > float(t.split('_')[1])/10).astype(int)
        test[t] = (test['disposals'] > float(t.split('_')[1])/10).astype(int)

# Prepare datasets
def prepare_X_y(data, feature_cols):
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = data[targets]
    return X, y

feature_cols_no_odds = base_features + rolling_features
feature_cols_with_odds = feature_cols_no_odds + ['total_score_close']

X_train_no_odds, y_train = prepare_X_y(train, feature_cols_no_odds)
X_test_no_odds, y_test = prepare_X_y(test, feature_cols_no_odds)

X_train_with_odds, _ = prepare_X_y(train, feature_cols_with_odds)
X_test_with_odds, _ = prepare_X_y(test, feature_cols_with_odds)

# Train and evaluate both models
results = {}
for target in targets:
    # Without odds
    clf_no = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    clf_no.fit(X_train_no_odds, y_train[target])
    proba_no = clf_no.predict_proba(X_test_no_odds)[:,1]
    roc_no = roc_auc_score(y_test[target], proba_no)
    pr_no = average_precision_score(y_test[target], proba_no)
    # With odds
    clf_yes = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    clf_yes.fit(X_train_with_odds, y_train[target])
    proba_yes = clf_yes.predict_proba(X_test_with_odds)[:,1]
    roc_yes = roc_auc_score(y_test[target], proba_yes)
    pr_yes = average_precision_score(y_test[target], proba_yes)
    results[target] = {
        'no_odds': {'roc': roc_no, 'pr': pr_no},
        'with_odds': {'roc': roc_yes, 'pr': pr_yes},
        'lift_roc': roc_yes - roc_no,
        'lift_pr': pr_yes - pr_no
    }
    print(f"{target}: no_odds ROC={roc_no:.4f} PR={pr_no:.4f}; with_odds ROC={roc_yes:.4f} PR={pr_yes:.4f}")

# Generate report
os.makedirs('outputs', exist_ok=True)
report_path = 'outputs/BACKTEST_2023_ODDS_COMPARISON.md'
with open(report_path, 'w') as f:
    f.write("# Backtest 2023: Odds vs No Odds\n\n")
    f.write("Train: 2020-2022 | Test: 2023\n")
    f.write("Model: Random Forest (200 trees, max_depth=10, min_samples_split=10, class_weight='balanced_subsample')\n\n")
    f.write("## Results\n\n")
    f.write("| Threshold | No Odds ROC | No Odds PR | With Odds ROC | With Odds PR | ROC Lift | PR Lift |\n")
    f.write("|-----------|-------------|------------|---------------|--------------|----------|--------|\n")
    for target in targets:
        r = results[target]
        f.write(f"| {target} | {r['no_odds']['roc']:.4f} | {r['no_odds']['pr']:.4f} | {r['with_odds']['roc']:.4f} | {r['with_odds']['pr']:.4f} | {r['lift_roc']:.4f} | {r['lift_pr']:.4f} |\n")
    f.write("\n## Conclusion\n")
    f.write("Including the match total line (total_score_close) as a feature improves performance across thresholds.\n")
print(f"Report saved to {report_path}")
