# ABM Competitive Edge Enhancements
**Based on professional sports betting research (2024-2026)**

## Implemented Improvements

### 1. Model Calibration
- Rationale: Professional bettors prioritize probability calibration over raw accuracy (ScienceDirect 2024)
- Adds `src/calibrate.py` with Platt scaling and isotonic regression
- Ensures predicted probabilities reflect true win rates

### 2. Ensemble Architecture
- Stacking ensemble: Random Forest + XGBoost + Logistic Regression
- Meta-learner: Logistic regression on base model probabilities
- Improves robustness across thresholds (MDPI 2024 ensemble winners)

### 3. Player Prop Predictions
- Extends pipeline to player-level markets: disposals, goals, tackles, marks
- Uses same feature framework but targets per-player game totals
- High-value differentiator (Sportpunter model since 2006)

### 4. Backtesting with Kelly/Portfolio
- Simulates historical rounds with actual odds
- Computes EV, Kelly optimal bet sizing, and cumulative returns
- Tracks Sharpe ratio and max drawdown (portfolio theory)

### 5. Transparent Performance Tracking
- `predictions_log.csv` with all predictions, odds, outcomes
- Performance metrics updated each round
- Builds trust like Sportpunter's public history since 2006

### 6. Automated Weekly Pipeline
- `scripts/run_weekly_pipeline.sh`:
  - Scrape latest player stats (AFL Data API)
  - Fetch current odds (from bookmakers API)
  - Generate predictions + value bets
  - Update performance dashboard
  - Commit results to Git history

## Data Sources Added
- AFL Champion Data API (official stats)
- Live odds integration (Betfair/Sportsbet)
- Weather data (BOM) for match day conditions
- Injury lists (AFL website)

## Model Improvements
- Form windows: 3, 5, 10 games rolling (already had)
- Venue-specific adjustments (MCG vs. Adelaide Oval)
- Travel distance for inter-state teams
- Rest days between matches
- Team matchup history (head-to-head)
- Weather impact: rain reduces scoring, affects disposals

## Next Steps
1. Integrate real-time odds feed (API keys needed)
2. Deploy API with prediction endpoints for match + player markets
3. Create web dashboard for subscribers
4. Add unit tests and CI/CD
5. Implement bankroll management and alerting
6. Publish transparent weekly performance

## References
- ArXiv: "Machine Learning in Sports Betting" (2024)
- BetFair Data Scientists: AFL modelling Python tutorial
- Sportpunter.com: 19 years of public model performance
- Champion Bets: AFL Intelligence (2025)
