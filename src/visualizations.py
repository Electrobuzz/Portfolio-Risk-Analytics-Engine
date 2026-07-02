"""
visualizations.py
-----------------
Visualization functions for the VaR engine including:
- Rolling VaR charts with multiple models
- VaR breach heatmaps by month/year
- Return distribution plots
- Stress test impact charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rolling_var_backtest(returns_series, forecasts_historical, forecasts_student_t=None, 
                               breaches=None, confidence=0.95, output_path="rolling_var_backtest.png"):
    """
    Generate a chart showing rolling VaR forecasts vs actual returns with breaches highlighted.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Actual portfolio returns
    forecasts_historical : array-like
        Rolling historical VaR forecasts
    forecasts_student_t : array-like, optional
        Rolling Student-t VaR forecasts
    breaches : array-like, optional
        Boolean array indicating VaR breaches
    confidence : float
        Confidence level used for VaR
    output_path : str
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Align forecasts with actual returns (forecasts start after window period)
    n_forecasts = len(forecasts_historical)
    returns_aligned = returns_series.iloc[-n_forecasts:]
    
    idx = np.arange(n_forecasts)
    
    # Plot actual returns
    ax.plot(idx, returns_aligned.values * 100, color="#4C72B0", linewidth=0.8, 
            label="Actual daily return", alpha=0.7)
    
    # Plot historical VaR
    ax.plot(idx, -forecasts_historical * 100, color="#DD8452", linewidth=1.2, 
            label=f"{int(confidence*100)}% Historical VaR")
    
    # Plot Student-t VaR if provided
    if forecasts_student_t is not None:
        ax.plot(idx, -forecasts_student_t * 100, color="#55A868", linewidth=1.2, 
                label=f"{int(confidence*100)}% Student-t VaR", linestyle='--')
    
    # Highlight breaches
    if breaches is not None:
        breach_idx = idx[breaches]
        ax.scatter(breach_idx, returns_aligned.values[breaches] * 100, color="#C44E52", 
                   zorder=5, s=30, label="VaR breach", marker='x')
    
    ax.set_title(f"Rolling {int(confidence*100)}% VaR Backtest: Forecast vs. Actual Returns")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Return (%)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_breach_heatmap(returns_series, forecasts, confidence=0.95, output_path="breach_heatmap.png"):
    """
    Generate a calendar heatmap showing VaR breaches by month/year.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Actual portfolio returns with DatetimeIndex
    forecasts : array-like
        Rolling VaR forecasts
    confidence : float
        Confidence level used for VaR
    output_path : str
        Path to save the figure
    """
    # Ensure returns_series has DatetimeIndex
    if not isinstance(returns_series.index, pd.DatetimeIndex):
        returns_series.index = pd.to_datetime(returns_series.index)
    
    # Align returns with forecasts (forecasts start after window period)
    n_forecasts = len(forecasts)
    returns_aligned = returns_series.iloc[-n_forecasts:]
    
    # Calculate breaches
    breaches = returns_aligned < -forecasts
    
    # Create DataFrame with year and month columns
    breach_df = pd.DataFrame({
        'breach': breaches,
        'year': returns_aligned.index.year,
        'month': returns_aligned.index.month
    })
    
    # Count breaches by year and month
    breach_counts = breach_df.groupby(['year', 'month'])['breach'].sum().unstack(fill_value=0)
    
    # Ensure all months are present
    for month in range(1, 13):
        if month not in breach_counts.columns:
            breach_counts[month] = 0
    breach_counts = breach_counts[sorted(breach_counts.columns)]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    sns.heatmap(breach_counts, annot=True, fmt='d', cmap='Reds', 
                cbar_kws={'label': 'Number of Breaches'}, ax=ax)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_title(f'VaR Breach Heatmap ({int(confidence*100)}% VaR)')
    ax.set_xticklabels(month_names, rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_return_distribution(port_returns, var_95, var_99, output_path="return_distribution_var.png"):
    """
    Plot portfolio return distribution with VaR thresholds marked.
    
    Parameters:
    -----------
    port_returns : array-like
        Portfolio returns
    var_95 : float
        95% VaR threshold
    var_99 : float
        99% VaR threshold
    output_path : str
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(port_returns * 100, bins=60, color="#4C72B0", alpha=0.75, edgecolor="white")
    ax.axvline(-var_95, color="#DD8452", linestyle="--", linewidth=2, 
               label=f"95% VaR = {var_95:.2f}%")
    ax.axvline(-var_99, color="#C44E52", linestyle="--", linewidth=2, 
               label=f"99% VaR = {var_99:.2f}%")
    ax.set_title("Portfolio Daily Return Distribution with VaR Thresholds")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_stress_test_impact(stress_df, output_path="stress_test_impact.png"):
    """
    Plot stress test scenario impact as a horizontal bar chart.
    
    Parameters:
    -----------
    stress_df : pd.DataFrame
        DataFrame with scenario and portfolio_return_pct columns
    output_path : str
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52" if v < 0 else "#55A868" for v in stress_df["portfolio_return_pct"]]
    ax.barh(stress_df["scenario"], stress_df["portfolio_return_pct"], color=colors)
    ax.set_title("Stress Test: Portfolio Impact by Scenario")
    ax.set_xlabel("Portfolio Return (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
