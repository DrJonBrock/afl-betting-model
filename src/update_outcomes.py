"""
Update outcomes for past predictions by scraping match results.
Marks predictions as correct/incorrect after games complete.
"""
import pandas as pd
import os
from datetime import datetime, timedelta

PREDICTIONS_LOG = "data/predictions_log.csv"
MATCH_RESULTS = "data/match_results.csv"

def load_results():
    """Load match results with actual disposals."""
    df = pd.read_csv(MATCH_RESULTS)
    # We need player-level disposal totals per match
    # Assuming df has: date, player, disposals (or we join)
    return df

def main():
    if not os.path.exists(PREDICTIONS_LOG):
        print("No predictions log found.")
        return

    pred_df = pd.read_csv(PREDICTIONS_LOG)
    results_df = load_results()

    # Only update predictions with missing outcome
    pending = pred_df[pred_df['outcome'].isna()].copy()
    if len(pending) == 0:
        print("No pending outcomes to update.")
        return

    print(f"Updating {len(pending)} predictions...")

    # Merge actual results on date + player (or team + market if aggregated)
    # For this prototype, we assume player-level outcomes
    updated = []
    for idx, row in pending.iterrows():
        date = row['date']
        target = row['target']  # e.g., over_195
        threshold = float(target.replace('over_', '')) / 10.0
        player = row.get('player')
        if pd.isna(player):
            continue
        # Find actual disposals
        actual = results_df[(results_df['date'] == date) & (results_df['player'] == player)]
        if len(actual) == 0:
            continue
        disposals = actual.iloc[0]['disposals']
        outcome = 1 if disposals > threshold else 0
        row['outcome'] = outcome
        updated.append(row)

    if len(updated) == 0:
        print("No matches found to update outcomes.")
        return

    # Replace in original dataframe
    update_df = pd.DataFrame(updated)
    pred_df.update(update_df)
    pred_df.to_csv(PREDICTIONS_LOG, index=False)
    print(f"Updated {len(updated)} outcomes.")
    # Optionally: recompute backtest metrics
    os.system("python3 src/backtest_kelly.py")

if __name__ == "__main__":
    main()
