"""
Backtest the AFL betting models: simulate bets and track returns.
Assumes we have models and fair odds; compares to bookmaker odds.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from src.models import FEATURE_COLS, load_features

MODELS_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(line: float):
    line_str = str(line).replace('.', '_')
    filename = f"disposals_over_{line_str}.pkl"
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)

def fair_odds_from_prob(p: float) -> float:
    """Convert model probability to fair decimal odds."""
    if p <= 0 or p >= 1:
        return np.nan
    return 1.0 / p

def kelly_fraction(p_fair: float, o_bookmaker: float) -> float:
    """Kelly criterion: f* = (bp - q) / b, where b = o-1, p = fair prob, q = 1-p."""
    b = o_bookmaker - 1
    if b <= 0:
        return 0.0
    p = 1 / p_fair
    q = 1 - p
    edge = b * p - q
    if edge <= 0:
        return 0.0
    f_star = edge / b
    return max(0.0, f_star)

def simulate_bets(df: pd.DataFrame, line: float = 24.5, min_edge: float = 0.05):
    """
    For each row in test set, generate model probability and decide to bet.
    Assume bookmaker_odds_over column exists (we'll mock with fair odds minus margin for now).
    """
    model = load_model(line)
    X = df[FEATURE_COLS].fillna(0)
    p_over = model.predict_proba(X)[:, 1]
    df = df.copy()
    df['p_model_over'] = p_over
    df['fair_odds_over'] = df['p_model_over'].apply(fair_odds_from_prob)
    # Mock bookmaker odds: assume bookmakers shade odds by 5% (overround)
    df['bookmaker_odds_over'] = df['fair_odds_over'] * 0.95
    # Determine value: we bet if fair odds exceed bookmaker odds by threshold (min_edge)
    df['edge'] = df['fair_odds_over'] / df['bookmaker_odds_over'] - 1
    df['bet'] = df['edge'] >= min_edge
    # Actual outcome
    target_col = f'target_over_{line}'
    df['actual_over'] = df[target_col]
    # Simulated P&L: if we bet OVER, we win (odds-1)*stake if actual_over else lose stake
    # Use Kelly fractional staking (fraction of bankroll)
    df['kelly'] = df.apply(lambda row: kelly_fraction(row['p_model_over'], row['bookmaker_odds_over']), axis=1)
    # Cap Kelly to avoid overexposure; e.g., half-Kelly
    df['stake'] = df['kelly'] * 0.5
    df['pnl'] = np.where(df['bet'], np.where(df['actual_over'] == 1, df['stake'] * (df['bookmaker_odds_over'] - 1), -df['stake']), 0)
    return df

def backtest(df: pd.DataFrame, line: float = 24.5):
    results = simulate_bets(df, line)
    bets = results[results['bet']]
    total_bets = len(bets)
    total_pnl = bets['pnl'].sum()
    win_rate = bets['actual_over'].mean()
    avg_odds = bets['bookmaker_odds_over'].mean()
    avg_edge = bets['edge'].mean()
    print(f"Backtest results (Disposals > {line}):")
    print(f"  Total bets placed: {total_bets}")
    print(f"  Total P&L: {total_pnl:.2f} (per unit stake)")
    print(f"  Win rate: {win_rate:.2%}")
    print(f"  Avg odds: {avg_odds:.2f}")
    print(f"  Avg edge: {avg_edge:.2%}")
    # Save results
    results.to_csv(os.path.join(OUTPUT_DIR, f"backtest_results_{line}.csv"), index=False)
    # Plot cumulative P&L
    if total_bets > 0:
        bets['cumulative_pnl'] = bets['pnl'].cumsum()
        plt.figure(figsize=(10, 4))
        plt.plot(bets['cumulative_pnl'].values)
        plt.title(f'Cumulative P&L – Disposals Over {line}')
        plt.xlabel('Bet number')
        plt.ylabel('P&L')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"cumulative_pnl_{line}.png"))
        plt.close()
    return results

def main():
    from src.models import load_features
    df = load_features()
    # Identify test set years: use last 2 years (as in models.py split)
    years = sorted(df['year'].unique())
    test_years = years[-2:]
    test_df = df[df['year'].isin(test_years)].copy()
    # Backtest each market
    for line in [19.5, 24.5, 29.5]:
        target = f'target_over_{line}'
        if target not in test_df.columns:
            continue
        backtest(test_df, line)

if __name__ == "__main__":
    main()