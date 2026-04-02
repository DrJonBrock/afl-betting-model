# ABM Competitive Edge: Implementation Summary (2026-04-02)

**Scope:** Research-driven enhancements to match professional sports betting model standards.

## Implemented Features

### 1. Model Calibration (`src/calibrate.py`)
- Platt scaling (logistic) on existing RF models
- Isotonic regression option (commented, data-hungry)
- Splits data: train on older years, calibrate on recent 2 years
- Saves calibrated models to `models/calibrated/`
- Metrics: ROC-AUC, Brier score

### 2. Ensemble Architecture (`src/ensemble.py`)
- Stacking ensemble: base learners = RF + XGBoost + Logistic Regression
- Meta-learner: Logistic Regression on base probabilities
- Also implements soft VotingClassifier for comparison
- Saves ensemble objects to `models/ensemble/`
- Per-target models for over_195, over_245, over_295

### 3. Kelly Backtesting (`src/backtest_kelly.py`)
- Fractional Kelly Criterion for optimal bet sizing
- Bankroll simulation with 5% max cap
- Portfolio metrics: total return, Sharpe ratio, max drawdown
- Breakdown by target market
- Outputs: `data/backtest_kelly_bets.csv`, `data/backtest_bankroll_curve.csv`

### 4. Weekly Automation Pipeline (`scripts/run_weekly_pipeline.sh`)
- End-to-end weekly steps: scrape → odds → predict → log → update outcomes
- Requires: Python src scripts and external APIs
- Designed for cron execution (Monday mornings)

### 5. Prediction Logging (`src/log_predictions.py`)
- Appends current predictions to central `data/predictions_log.csv`
- Deduplication by (date, target, player)
- Adds `logged_at` timestamp

### 6. Outcome Updater (`src/update_outcomes.py`)
- Scrapes match results to mark predictions correct/incorrect
- Enables ongoing backtest accuracy after games complete

### 7. Dashboard Generator (`src/generate_dashboard.py`)
- Produces `docs/performance_dashboard.html` with:
  - Ensemble validation metrics
  - Calibration impact (Brier, AUC)
  - Backtest Kelly portfolio results
- Intended as shareable performance report

### 8. Predict Week (`src/predict_week.py`)
- Generates predictions for upcoming round using best available model
- Looks for calibrated > ensemble > base
- Outputs `data/current_predictions.csv`

### 9. Requirements Update
- Added `xgboost==2.0.3` for ensemble base models

## Documentation Added
- `ENHANCEMENTS_PLAN.md` — rationale and roadmap
- `OPENCLAW_BEST_PRACTICES.md` — compiled community setup guide (your Task 1 output)

## Data Source Additions (planned/integration)
- AFL Champion Data API (official stats)
- Live odds (Betfair/Sportsbet)
- BOM weather API
- Injury lists (AFL website)
- Advanced features: rest days, travel distance, venue factors, weather impact

## Professional Practices Ingested
- Calibration > raw accuracy for betting probabilities (ScienceDirect 2024)
- Ensemble methods consistently outperform single models in sports betting (MDPI 2024)
- Player prop markets are high-value differentiators (Sportpunter since 2006)
- Transparent performance tracking builds subscriber trust
- Portfolio-style risk management (Kelly, Sharpe ratio)
- Avoid overfitting; use time-based validation splits

## Current Status
- Code complete for enhancement components
- Awaits integration testing: need to run ensemble.py and calibrate.py on existing features
- Backtest requires predictions_log with known outcomes (to be populated weekly)
- Weekly pipeline scaffolded; needs live odds API and outcome scraper wiring

## Next Steps
1. Run `scripts/run_full_training.sh` to train ensemble and calibrate models
2. Populate `data/predictions_log.csv` with historical predictions for backtest validation
3. Implement live odds feed (API keys)
4. Deploy API with match + player market endpoints
5. Build subscriber web dashboard (react/vue)
6. Add CI/CD and automated weekly runs
7. Publish transparent performance history

## Notes
- All new code is Python 3.12 compatible
- Uses existing `data/processed/features.csv` as single source of truth
- Models saved with joblib for fast loading
- Modular design for independent component testing
