#!/bin/bash
# Full training pipeline: base models -> ensemble -> calibration -> backtest

set -euo pipefail

cd /home/lukef/.openclaw/workspace/projects/afl-betting-model

echo "=== Starting full training pipeline ==="

# Step 1: Train base Random Forest models (existing train.py)
echo "[1/4] Training base Random Forest models..."
python3 src/train.py

# Step 2: Train ensemble stacking models
echo "[2/4] Training stacking ensemble..."
python3 src/ensemble.py

# Step 3: Calibrate probabilities
echo "[3/4] Calibrating model probabilities..."
python3 src/calibrate.py

# Step 4: Backtest with Kelly criterion
echo "[4/4] Running backtest simulation..."
python3 src/backtest_kelly.py

echo "=== Training pipeline complete ==="
echo "Models saved in models/, models/ensemble/, models/calibrated/"
echo "Backtest results: data/backtest_kelly_bets.csv, data/backtest_bankroll_curve.csv"
