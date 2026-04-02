"""
Advanced backtesting with Kelly Criterion and portfolio risk management.
Simulates historical betting performance with optimal bankroll allocation.
"""
import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import gmean

def load_predictions():
    """Load predictions log with probabilities and odds."""
    df = pd.read_csv("data/predictions_log.csv")
    return df

def kelly_fraction(prob, odds, fractions=[1.0]):
    """
    Compute Kelly Criterion fractional bet sizes.
    prob: model's predicted probability
    odds: decimal odds (e.g., 2.0)
    Returns optimal fraction of bankroll.
    b = odds - 1 (net odds)
    f* = (b*p - q) / b = (p*(odds-1) - (1-p)) / (odds-1)
    """
    if odds <= 1.0:
        return 0.0
    b = odds - 1
    q = 1 - prob
    f_star = (b * prob - q) / b
    # Cap at conservative fractions
    return max(0, min(f_star, 0.05))  # never bet >5% of bankroll per Kelly

def simulate_bankroll(df, initial_bankroll=10000.0):
    """
    Simulate betting results through time.
    df must contain: date, target_market, model_proba, deci_odds, outcome (1/0)
    Returns bankroll curve and metrics.
    """
    df = df.copy().sort_values('date')
    bankroll = [initial_bankroll]
    bets = []
    for _, row in df.iterrows():
        bet_fraction = kelly_fraction(row['model_proba'], row['deci_odds'])
        stake = bankroll[-1] * bet_fraction
        if row['outcome'] == 1:
            win = stake * (row['deci_odds'] - 1)
            new_bankroll = bankroll[-1] + win
        else:
            new_bankroll = bankroll[-1] - stake
        bankroll.append(new_bankroll)
        bets.append({
            'date': row['date'],
            'target': row['target'],
            'proba': row['model_proba'],
            'odds': row['deci_odds'],
            'fraction': bet_fraction,
            'stake': stake,
            'outcome': row['outcome'],
            'bankroll_before': bankroll[-1],
            'bankroll_after': new_bankroll
        })
    bets_df = pd.DataFrame(bets)
    final = bankroll[-1]
    total_return = (final - initial_bankroll) / initial_bankroll
    n_bets = len(bets)
    win_rate = bets_df['outcome'].mean()
    avg_stake = bets_df['stake'].mean() / initial_bankroll
    # Sharpe ratio (simplified, assuming daily steps)
    returns = bets_df['bankroll_after'].pct_change().fillna(0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    # Max drawdown
    running_max = bets_df['bankroll_after'].cummax()
    drawdown = (bets_df['bankroll_after'] - running_max) / running_max
    max_dd = drawdown.min()
    metrics = {
        'final_bankroll': final,
        'total_return_pct': total_return * 100,
        'n_bets': n_bets,
        'win_rate': win_rate,
        'avg_stake_pct': avg_stake * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd * 100,
        'gmean_daily': gmean(1 + returns) - 1 if n_bets > 0 else 0
    }
    return metrics, bets_df, bankroll

def main():
    print("Loading predictions history...")
    df = load_predictions()
    # Filter to over_ markets only, with known outcomes and decimal odds
    df = df.dropna(subset=['model_proba', 'deci_odds', 'outcome'])
    df = df[df['target'].str.startswith('over_')]
    print(f"Evaluating {len(df)} historical predictions")

    metrics, bets_df, bankroll_curve = simulate_bankroll(df, initial_bankroll=10000.0)

    print("\n=== Backtest Metrics (Kelly + Portfolio) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save detailed bets
    bets_df.to_csv("data/backtest_kelly_bets.csv", index=False)
    pd.DataFrame({'bankroll': bankroll_curve}).to_csv("data/backtest_bankroll_curve.csv", index=False)
    print("\nSaved detailed bets to data/backtest_kelly_bets.csv")
    print("Saved bankroll curve to data/backtest_bankroll_curve.csv")

    # Compute by target market
    print("\n=== Performance by Market ===")
    by_target = []
    for target in df['target'].unique():
        sub = df[df['target'] == target]
        m, _, _ = simulate_bankroll(sub, initial_bankroll=10000.0)
        m['target'] = target
        by_target.append(m)
    pd.DataFrame(by_target).to_csv("data/backtest_by_target.csv", index=False)
    print(pd.DataFrame(by_target)[['target','total_return_pct','n_bets','win_rate','sharpe_ratio']])

if __name__ == "__main__":
    main()
