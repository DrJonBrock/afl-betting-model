"""
Log predictions to central history file with timestamps.
Appends new predictions and ensures schema consistency.
"""
import pandas as pd
import os
from datetime import datetime

PREDICTIONS_LOG = "data/predictions_log.csv"
NEW_PREDICTIONS = "data/current_predictions.csv"

def main():
    if not os.path.exists(NEW_PREDICTIONS):
        print(f"No new predictions file at {NEW_PREDICTIONS}")
        return

    new_df = pd.read_csv(NEW_PREDICTIONS)
    # Ensure required columns
    required = ['date', 'target', 'model_proba', 'deci_odds', 'outcome']
    for col in required:
        if col not in new_df.columns:
            if col == 'outcome':
                new_df[col] = None
            else:
                print(f"Missing column {col} in predictions; abort.")
                return

    # Add timestamp
    new_df['logged_at'] = datetime.now().isoformat()

    # Load existing log if present
    if os.path.exists(PREDICTIONS_LOG):
        old_df = pd.read_csv(PREDICTIONS_LOG)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Deduplicate by (date, target, player) if exists
    if 'player' in combined.columns:
        combined = combined.drop_duplicates(subset=['date', 'target', 'player'], keep='last')
    else:
        combined = combined.drop_duplicates(subset=['date', 'target'], keep='last')

    combined.to_csv(PREDICTIONS_LOG, index=False)
    print(f"Logged {len(new_df)} predictions to {PREDICTIONS_LOG} (total {len(combined)})")

if __name__ == "__main__":
    main()
