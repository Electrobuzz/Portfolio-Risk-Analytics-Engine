"""
data.py
-------
Provides portfolio price/return data for the VaR engine.

Two modes:
1. load_from_csv(path)  -> use your own historical price data (recommended for real use).
   CSV format: first column = Date, remaining columns = one per asset, values = adjusted close prices.
2. generate_synthetic_prices(...) -> generates realistic correlated multi-asset price paths
   (correlated Geometric Brownian Motion) so the engine can be demoed without a live market
   data feed. Swap this out for real data (e.g. via yfinance, Bloomberg, or your firm's data
   warehouse) in production.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def generate_synthetic_prices(
    tickers=("EQUITY_IDX", "GOVT_BOND", "CORP_BOND", "FX_USDINR"),
    n_days=1000,
    annual_vols=(0.18, 0.06, 0.09, 0.08),
    annual_drift=(0.09, 0.03, 0.04, 0.01),
    correlation=None,
    start_price=100.0,
    seed=42,
):
    """
    Simulate correlated daily prices for a small multi-asset portfolio using
    correlated Geometric Brownian Motion. Meant to stand in for a real
    historical price feed during development/demo.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    if correlation is None:
        # Reasonable default correlation structure:
        # equity vs govt bond slightly negative, equity vs corp bond positive,
        # FX roughly uncorrelated with the rest.
        correlation = np.array([
            [1.00, -0.25, 0.35, 0.10],
            [-0.25, 1.00, 0.55, -0.05],
            [0.35, 0.55, 1.00, 0.05],
            [0.10, -0.05, 0.05, 1.00],
        ])[:n_assets, :n_assets]

    daily_vols = np.array(annual_vols) / np.sqrt(252)
    daily_drift = np.array(annual_drift) / 252

    cov = np.outer(daily_vols, daily_vols) * correlation
    chol = np.linalg.cholesky(cov)

    z = rng.standard_normal(size=(n_days, n_assets))
    correlated_shocks = z @ chol.T
    log_returns = daily_drift + correlated_shocks

    log_prices = np.log(start_price) + np.cumsum(log_returns, axis=0)
    prices = np.exp(log_prices)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df = pd.DataFrame(prices, index=dates, columns=list(tickers))
    df.index.name = "Date"
    return df


def load_from_yfinance(tickers, start_date, end_date=None):
    """
    Load historical price data from Yahoo Finance using yfinance.
    
    Parameters:
    -----------
    tickers : list of str
        List of ticker symbols (e.g., ["SPY", "QQQ", "TLT"])
    start_date : str
        Start date in "YYYY-MM-DD" format
    end_date : str, optional
        End date in "YYYY-MM-DD" format. If None, uses today.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date index and one column per ticker with adjusted close prices.
    """
    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )["Close"]
    
    # Ensure we have a DataFrame even if only one ticker
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    return prices.sort_index()


def load_from_csv(path):
    """Load a price history CSV: Date index, one column per asset (adjusted close)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def compute_returns(price_df):
    """Simple (arithmetic) daily returns, used for historical/parametric/MC VaR."""
    return price_df.pct_change().dropna()
