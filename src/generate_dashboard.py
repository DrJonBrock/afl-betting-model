"""
Generate a simple HTML dashboard of model performance and backtest results.
Run after each weekly update to produce a shareable report.
"""
import pandas as pd
import os
from datetime import datetime

def load_backtest_summary():
    """Load aggregated backtest metrics."""
    path = "data/backtest_by_target.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None

def load_ensemble_summary():
    path = "models/ensemble/ensemble_summary.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        return df
    return None

def load_calibration_summary():
    path = "models/calibrated/calibration_summary.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        return df
    return None

def generate_html():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>ABM Performance Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1, h2 {{ color: #2c3e50; }}
    table {{ border-collapse: collapse; margin: 20px 0; }}
    th, td {{ border: 1px solid #bdc3c7; padding: 8px 12px; text-align: left; }}
    th {{ background-color: #34495e; color: white; }}
    tr:nth-child(even) {{ background-color: #ecf0f1; }}
    .metric {{ font-weight: bold; }}
  </style>
</head>
<body>
  <h1>AFL Betting Model — Performance Dashboard</h1>
  <p>Generated: {timestamp}</p>
"""
    # Ensemble summary
    ens = load_ensemble_summary()
    if ens is not None:
        html += "<h2>Ensemble Model Performance (Validation)</h2>"
        html += ens.to_html(classes="data-table", border=0)
    else:
        html += "<p><i>Ensemble training results not available yet.</i></p>"

    # Calibration summary
    cal = load_calibration_summary()
    if cal is not None:
        html += "<h2>Calibration Impact</h2>"
        html += cal.to_html(classes="data-table", border=0)
    else:
        html += "<p><i>Calibration results not available yet.</i></p>"

    # Backtest summary
    back = load_backtest_summary()
    if back is not None:
        html += "<h2>Backtest: Kelly Portfolio Simulation</h2>"
        html += back.to_html(classes="data-table", border=0)
    else:
        html += "<p><i>Backtest results not available yet.</i></p>"

    html += """
  <h2>Notes</h2>
  <ul>
    <li>Models trained on historical data up to 2025, validated on 2024-2025.</li>
    <li>Calibration ensures predicted probabilities reflect true frequencies (Brier score).</li>
    <li>Backtest uses decimal odds and Kelly Criterion for fractional bet sizing.</li>
    <li>Simulated bankroll starts at $10,000 with 5% max bet cap.</li>
  </ul>
</body>
</html>
"""
    out_path = "docs/performance_dashboard.html"
    os.makedirs("docs", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Dashboard generated: {out_path}")

if __name__ == "__main__":
    generate_html()
