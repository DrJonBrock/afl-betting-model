# AFL Betting Model
**Goal:** Build a data-driven system to generate fair odds for AFL player and match markets, identify value bets against bookmaker lines.
**Started:** 2025-03-30
**Lead:** Luke (owner), Marc Radaic (partner)
**Architect:** BobBot (autonomous AI assistant)

## Data Sources
- AFL Tables (player-game stats, match results, venues)
- Footywire (advanced metrics, rankings)
- Footyinfo (betting lines)
- AFL.com.au (injury reports, team lists)
- Fantasy SuperCoach (projections)

## Modelling Approach
- Primary: XGBoost for binary/multi-class markets
- Secondary: Bayesian shrinkage for small-sample players
- Ensemble options later

## Quick Start
```bash
cd /home/lukef/.openclaw/workspace/projects/afl-betting-model
pip install -r requirements.txt
# Run scraper to collect data
python src/scrape.py
# Preprocess and build features
python src/preprocess.py
# Train model
python src/models.py
# Backtest
python src/backtest.py
# Predict next round
python src/predict.py
```

## Project Status (2025-03-30)
- [x] Project scaffold created
- [ ] Scraper for AFL Tables (in progress)
- [ ] Feature engineering pipeline
- [ ] MVP model (disposals over/under)
- [ ] Backtesting framework
- [ ] Value bet generator