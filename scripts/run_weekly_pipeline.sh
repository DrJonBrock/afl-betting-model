#!/bin/bash
# AFL Betting Model - Weekly Automation Pipeline
# Runs every Monday (or manually) to generate predictions and update performance tracking

set -euo pipefail

PROJECT_DIR="/home/lukef/.openclaw/workspace/projects/afl-betting-model"
cd "$PROJECT_DIR"

echo "=== Starting weekly pipeline: $(date) ==="

# 1. Scrape latest player stats (AFL Data API)
echo "[1/5] Fetching latest player statistics..."
python3 src/scrape_2026_full.py || { echo "Scrape failed"; exit 1; }

# 2. Fetch current week's odds from bookmakers
echo "[2/5] Fetching current odds..."
python3 src/supply_2026_odds.py || { echo "Odds fetch failed"; exit 1; }

# 3. Create features and run predictions
echo "[3/5] Generating predictions..."
python3 src/predict_round_2026.py || { echo "Prediction failed"; exit 1; }

# 4. Append to predictions log with timestamps
echo "[4/5] Updating predictions history..."
python3 src/log_predictions.py || { echo "Logging failed"; exit 1; }

# 5. Update performance metrics (once outcomes known)
# This step will run after games complete; separate cron job
echo "[5/5] Checking for pending outcome updates..."
python3 src/update_outcomes.py || echo "No new outcomes or update script not needed yet"

echo "=== Pipeline complete: $(date) ==="

# Optional: Git commit history for transparency
git add data/predictions_log.csv data/backtest_*.csv models/ 2>/dev/null || true
git commit -m "Weekly update: $(date +%Y-%m-%d) predictions and performance" 2>/dev/null || echo "No changes to commit"
