"""
stress_test.py
--------------
Applies discrete, named market-shock scenarios to the portfolio to estimate
tail losses outside the normal VaR framework (VaR only tells you about
"normal" bad days - stress tests answer "what if 2008 happens again?").

Shocks are expressed as one-day return shocks per asset class. In a real
desk these would be sourced from historical episodes or a scenario
committee; here they are illustrative and clearly labeled as such.
"""

import numpy as np
import pandas as pd


# Illustrative shock scenarios: {scenario_name: {ticker_shock}}
# Values are one-day (or scenario-window) return shocks, e.g. -0.20 = -20%.
# These are calibrated for real ETF tickers covering major asset classes.
DEFAULT_SCENARIOS = {
    "2008_GFC_equity_crash": {
        "SPY": -0.20, "QQQ": -0.25, "TLT": 0.03, "GLD": 0.05, "USO": -0.30,
        "HYG": -0.15, "EEM": -0.25, "VNQ": -0.35,
    },
    "2020_COVID_shock": {
        "SPY": -0.12, "QQQ": -0.15, "TLT": 0.02, "GLD": 0.03, "USO": -0.40,
        "HYG": -0.10, "EEM": -0.18, "VNQ": -0.20,
    },
    "rates_up_100bp_shock": {
        "SPY": -0.03, "QQQ": -0.05, "TLT": -0.04, "GLD": -0.02, "USO": 0.01,
        "HYG": -0.035, "EEM": -0.06, "VNQ": -0.05,
    },
    "emerging_market_crisis": {
        "SPY": -0.08, "QQQ": -0.10, "TLT": 0.01, "GLD": 0.02, "USO": -0.05,
        "HYG": -0.08, "EEM": -0.25, "VNQ": -0.12,
    },
    "commodity_crash": {
        "SPY": -0.05, "QQQ": -0.07, "TLT": 0.02, "GLD": -0.10, "USO": -0.35,
        "HYG": -0.04, "EEM": -0.12, "VNQ": -0.08,
    },
}


def run_stress_tests(asset_names, weights, scenarios=None, portfolio_value=1_000_000):
    """
    Apply each scenario's per-asset shock to the portfolio and report the
    resulting P&L in both percentage and currency terms.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    weights = np.asarray(weights)
    rows = []
    for name, shock_map in scenarios.items():
        shocks = np.array([shock_map.get(a, 0.0) for a in asset_names])
        port_shock_pct = weights @ shocks
        rows.append({
            "scenario": name,
            "portfolio_return_pct": port_shock_pct * 100,
            "pnl_currency": port_shock_pct * portfolio_value,
        })

    df = pd.DataFrame(rows).sort_values("portfolio_return_pct")
    return df


def historical_scenario_replay(returns_df, weights, portfolio_value=1_000_000):
    """
    Replay actual historical returns from major crisis periods to estimate
    portfolio P&L under those conditions.
    
    This is more realistic than hypothetical shocks as it uses actual market
    data including correlation breakdown and volatility clustering effects.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Historical asset returns with DatetimeIndex
    weights : array-like
        Portfolio weights
    portfolio_value : float
        Portfolio value in base currency
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with scenario name, date range, portfolio return, and P&L
    """
    # Define historical crisis periods (date ranges)
    historical_scenarios = {
        "2008_GFC": ("2008-09-01", "2008-11-30"),
        "2011_EU_Debt_Crisis": ("2011-07-01", "2011-09-30"),
        "2015_China_Slowdown": ("2015-06-01", "2015-09-30"),
        "2018_Q4_Selloff": ("2018-10-01", "2018-12-31"),
        "2020_COVID_Crash": ("2020-02-15", "2020-03-31"),
        "2022_Rate_Hike_Cycle": ("2022-01-01", "2022-06-30"),
    }
    
    weights = np.asarray(weights)
    rows = []
    
    for scenario_name, (start_date, end_date) in historical_scenarios.items():
        # Filter returns for the crisis period
        period_returns = returns_df.loc[start_date:end_date]
        
        if len(period_returns) == 0:
            # Skip if no data available for this period
            continue
        
        # Compute cumulative portfolio return over the period
        # Using geometric compounding: (1 + r1) * (1 + r2) * ... - 1
        period_returns_values = period_returns.values
        daily_port_returns = period_returns_values @ weights
        cumulative_return = (1 + daily_port_returns).prod() - 1
        
        rows.append({
            "scenario": scenario_name,
            "start_date": start_date,
            "end_date": end_date,
            "n_days": len(period_returns),
            "portfolio_return_pct": cumulative_return * 100,
            "pnl_currency": cumulative_return * portfolio_value,
        })
    
    df = pd.DataFrame(rows).sort_values("portfolio_return_pct")
    return df
