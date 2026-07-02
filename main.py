"""
main.py
-------
End-to-end demo of the Enhanced Portfolio VaR & Stress-Testing Engine:

1. Load real market data from Yahoo Finance for a multi-asset portfolio
2. Compute 1-day VaR at 95% / 99% via multiple methods:
   - Historical Simulation
   - Parametric Normal
   - Parametric Student-t
   - Monte Carlo Normal
   - Monte Carlo Student-t
   - Filtered Historical Simulation (GARCH-based)
3. Backtest using Kupiec POF test and Christoffersen Conditional Coverage test
4. Run hypothetical stress scenarios and historical scenario replay
5. Compute diversification analysis, component VaR, marginal VaR, incremental VaR
6. Generate comprehensive visualizations and report

Run:
    python main.py
Outputs land in ./output/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_from_yfinance, compute_returns
from src.var_models import (summarize_var, portfolio_returns, student_t_parametric_var,
                        filtered_historical_var, diversification_analysis, component_var,
                        marginal_var, incremental_var, historical_var, fit_student_t_distribution,
                        compare_distribution_fit)
from src.backtest import rolling_var_forecast, rolling_fhs_forecast, kupiec_pof_test, christoffersen_test
from src.stress_test import run_stress_tests, DEFAULT_SCENARIOS, historical_scenario_replay
from src.visualizations import (plot_rolling_var_backtest, plot_breach_heatmap,
                           plot_return_distribution, plot_stress_test_impact)

OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
TICKERS = [
    "SPY",   # S&P 500 ETF (US Large Cap Equity)
    "QQQ",   # Nasdaq 100 ETF (US Tech Equity)
    "TLT",   # 20+ Year Treasury Bond ETF (Long-term Govt Bonds)
    "GLD",   # Gold ETF (Commodity/Precious Metal)
    "USO",   # US Oil Fund ETF (Energy Commodity)
    "HYG",   # High Yield Corporate Bond ETF (Corporate Bonds)
    "EEM",   # Emerging Markets ETF (EM Equity)
    "VNQ"    # US Real Estate ETF (REITs)
]
WEIGHTS = [0.25, 0.15, 0.15, 0.10, 0.05, 0.10, 0.10, 0.10]  # portfolio allocation, must sum to 1
PORTFOLIO_VALUE = 1_000_000  # in USD for currency-denominated results

# Load 15 years of historical data from Yahoo Finance (covers major crises: 2008 GFC, 2011 EU debt crisis, 2015-16 China slowdown, 2018 Q4 selloff, 2020 COVID, 2022 rate hike cycle)
prices = load_from_yfinance(tickers=TICKERS, start_date="2010-01-01")
returns = compute_returns(prices)

print(f"Loaded {len(prices)} days of price history for {len(TICKERS)} assets.")
print(f"Portfolio weights: {dict(zip(TICKERS, WEIGHTS))}\n")

# ---------------------------------------------------------------------------
# 2. VaR Summary (All Methods)
# ---------------------------------------------------------------------------
var_summary, port_ret = summarize_var(returns, WEIGHTS, confidences=(0.95, 0.99))
var_summary_display = var_summary.copy()
for col in var_summary_display.columns:
    if col != "confidence":
        var_summary_display[col] = var_summary_display[col].round(3)

print("=== 1-Day VaR Summary (% of portfolio value) ===")
print(var_summary_display.to_string(index=False))
print()

# Add Filtered Historical Simulation VaR
print("=== Filtered Historical Simulation VaR (GARCH-based) ===")
for conf in [0.95, 0.99]:
    fhs_var = filtered_historical_var(port_ret, confidence=conf)
    print(f"At {int(conf*100)}% confidence: FHS VaR = {fhs_var*100:.3f}%")
print()

# ---------------------------------------------------------------------------
# Student-t Distribution Fit
# ---------------------------------------------------------------------------
print("=== Student-t Distribution Fit ===")
student_t_fit = fit_student_t_distribution(port_ret)

# Validation warnings
if student_t_fit["df"] <= 2:
    print("WARNING: Estimated degrees of freedom <= 2 - variance does not exist. Model invalid.")
elif student_t_fit["df"] < 4:
    print("WARNING: Estimated degrees of freedom indicate extremely heavy tails.")
elif student_t_fit["df"] > 30:
    print("NOTE: Student-t distribution is very close to Gaussian.")

print(f"Estimated Degrees of Freedom (ν): {student_t_fit['df']:.2f}")
print(f"Estimated Mean (μ):              {student_t_fit['mu']*100:.4f}%")
print(f"Estimated Scale Parameter:       {student_t_fit['scale']*100:.3f}%")
print(f"Sample Standard Deviation:      {student_t_fit['sample_std']*100:.3f}%")
print(f"Optimization Status:             {'SUCCESS' if student_t_fit['optimization_success'] else 'FAILED'}")
if student_t_fit["excess_kurtosis"] == np.inf:
    print(f"Implied Excess Kurtosis:       Infinite (ν <= 4)")
else:
    print(f"Implied Excess Kurtosis:       {student_t_fit['excess_kurtosis']:.2f}")
print()

# Distribution Fit Comparison
print("=== Distribution Fit Comparison ===")
fit_comparison = compare_distribution_fit(port_ret)
print(f"{'Distribution':<15} {'LogLik':<12} {'AIC':<12} {'BIC':<12}")
print("-" * 51)
print(f"{'Normal':<15} {fit_comparison['normal']['log_likelihood']:<12.1f} {fit_comparison['normal']['aic']:<12.1f} {fit_comparison['normal']['bic']:<12.1f}")
print(f"{'Student-t':<15} {fit_comparison['student_t']['log_likelihood']:<12.1f} {fit_comparison['student_t']['aic']:<12.1f} {fit_comparison['student_t']['bic']:<12.1f}")
print()
print(f"Best Fit: {fit_comparison['best_fit'].upper()}")
print()

# Currency-denominated view for the 95% / 99% historical VaR
for _, row in var_summary.iterrows():
    conf_pct = int(row["confidence"] * 100)
    usd_var = row["historical_var_pct"] / 100 * PORTFOLIO_VALUE
    print(f"At {conf_pct}% confidence: 1-day historical VaR ≈ ${usd_var:,.0f} "
          f"(on a ${PORTFOLIO_VALUE:,.0f} portfolio)")
print()

# ---------------------------------------------------------------------------
# 3. Backtesting (Kupiec POF and Christoffersen tests)
# ---------------------------------------------------------------------------
port_ret_series = pd.Series(port_ret, index=returns.index)

# Historical VaR backtesting
forecasts_hist, actuals_hist, breaches_hist = rolling_var_forecast(port_ret_series, window=250, confidence=0.95)
kupiec_result_hist = kupiec_pof_test(breaches_hist, confidence=0.95)
christoffersen_result_hist = christoffersen_test(breaches_hist, confidence=0.95)

print("=== Historical VaR Backtest (95% VaR, 250-day rolling window) ===")
print("KUPIEC POF TEST")
for k, v in kupiec_result_hist.items():
    print(f"  {k}: {v}")
print("CHRISTOFFERSEN CONDITIONAL COVERAGE TEST")
for k, v in christoffersen_result_hist.items():
    if k != "transition_matrix":
        print(f"  {k}: {v}")
print()

# FHS VaR backtesting (GARCH-based)
print("=== FHS VaR Backtest (95% VaR, 250-day rolling window, GARCH-based) ===")
print("Computing rolling FHS forecasts (this may take a few minutes)...")
forecasts_fhs, actuals_fhs, breaches_fhs = rolling_fhs_forecast(port_ret_series, window=250, confidence=0.95, n_bootstrap=500)
kupiec_result_fhs = kupiec_pof_test(breaches_fhs, confidence=0.95)
christoffersen_result_fhs = christoffersen_test(breaches_fhs, confidence=0.95)

print("KUPIEC POF TEST")
for k, v in kupiec_result_fhs.items():
    print(f"  {k}: {v}")
print("CHRISTOFFERSEN CONDITIONAL COVERAGE TEST")
for k, v in christoffersen_result_fhs.items():
    if k != "transition_matrix":
        print(f"  {k}: {v}")
print()

# Use FHS forecasts for visualizations (more accurate)
forecasts = forecasts_fhs
breaches = breaches_fhs

# ---------------------------------------------------------------------------
# 4. Stress testing (Hypothetical and Historical Replay)
# ---------------------------------------------------------------------------
stress_df = run_stress_tests(TICKERS, WEIGHTS, DEFAULT_SCENARIOS, PORTFOLIO_VALUE)
print("=== Hypothetical Stress Test Scenarios (worst to best) ===")
print(stress_df.to_string(index=False))
print()

historical_stress_df = historical_scenario_replay(returns, WEIGHTS, PORTFOLIO_VALUE)
print("=== Historical Scenario Replay (worst to best) ===")
print(historical_stress_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 5. Diversification Analysis
# ---------------------------------------------------------------------------
div_analysis = diversification_analysis(returns, WEIGHTS, confidence=0.95)
print("=== Diversification Analysis (95% VaR) ===")
print(f"Sum of Individual VaRs: {div_analysis['sum_individual_vars_pct']:.3f}%")
print(f"Portfolio VaR: {div_analysis['portfolio_var_pct']:.3f}%")
print(f"Diversification Benefit: {div_analysis['diversification_benefit_pct']:.3f}%")
print(f"Diversification Efficiency: {div_analysis['diversification_efficiency']:.4f}")
print()

# ---------------------------------------------------------------------------
# 6. Risk Contributions (Component VaR)
# ---------------------------------------------------------------------------
# Use dict weights for proper alignment
weights_dict = dict(zip(TICKERS, WEIGHTS))
comp_var_df, portfolio_var = component_var(returns, weights_dict, confidence=0.95)
print("=== Component VaR (95%, Euler Allocation) ===")
print(comp_var_df.to_string(index=False))
sum_comp_var = comp_var_df['component_var_pct'].sum()
print(f"Sum of Component VaRs: {sum_comp_var:.3f}% (vs Portfolio VaR: {portfolio_var*100:.3f}%)")
print(f"Validation: {'PASS' if np.isclose(sum_comp_var, portfolio_var*100, atol=0.01) else 'FAIL'} - sum(Component VaR) ≈ Portfolio VaR")
print()

# ---------------------------------------------------------------------------
# 7. Marginal VaR (Analytical)
# ---------------------------------------------------------------------------
marg_var_df = marginal_var(returns, weights_dict, confidence=0.95, delta=0.01, method="analytical")
print("=== Marginal VaR (95%, Analytical Formula) ===")
print(marg_var_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 8. Incremental VaR
# ---------------------------------------------------------------------------
inc_var_df = incremental_var(returns, weights_dict, confidence=0.95)
print("=== Incremental VaR (95%, position removal) ===")
print(inc_var_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 9. Audited Risk Attribution DataFrame
# ---------------------------------------------------------------------------
print("=== Audited Risk Attribution (Asset-Weight Aligned) ===")
audit_df = pd.DataFrame({
    "asset": comp_var_df["asset"],
    "weight": comp_var_df["weight"],
    "marginal_var_pct": marg_var_df["marginal_var_pct"],
    "component_var_pct": comp_var_df["component_var_pct"],
    "incremental_var_pct": inc_var_df["incremental_var_pct"],
})
print(audit_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 9. Charts
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# Chart 1: Portfolio return distribution with VaR thresholds marked
var95 = var_summary.loc[var_summary.confidence == 0.95, "historical_var_pct"].values[0]
var99 = var_summary.loc[var_summary.confidence == 0.99, "historical_var_pct"].values[0]
plot_return_distribution(port_ret, var95, var99, 
                        output_path=os.path.join(OUTDIR, "return_distribution_var.png"))

# Chart 2: Rolling VaR forecast vs. actual returns, with breaches marked
plot_rolling_var_backtest(port_ret_series, forecasts, breaches=breaches, confidence=0.95,
                         output_path=os.path.join(OUTDIR, "rolling_var_backtest.png"))

# Chart 3: VaR Breach Heatmap
plot_breach_heatmap(port_ret_series, forecasts, confidence=0.95,
                    output_path=os.path.join(OUTDIR, "breach_heatmap.png"))

# Chart 4: Stress test scenario impact
plot_stress_test_impact(stress_df, output_path=os.path.join(OUTDIR, "stress_test_impact.png"))

print(f"Charts saved to ./{OUTDIR}/")

# ---------------------------------------------------------------------------
# 10. Text summary report
# ---------------------------------------------------------------------------
report_lines = []
report_lines.append("PORTFOLIO VAR & STRESS TESTING - COMPREHENSIVE REPORT")
report_lines.append("=" * 70)
report_lines.append(f"Portfolio value: ${PORTFOLIO_VALUE:,.0f}")
report_lines.append(f"Assets & weights: {dict(zip(TICKERS, WEIGHTS))}")
report_lines.append(f"History used: {len(prices)} trading days\n")

report_lines.append("=" * 70)
report_lines.append("PORTFOLIO SUMMARY")
report_lines.append("=" * 70)
report_lines.append(f"Number of assets: {len(TICKERS)}")
report_lines.append(f"Data period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("VaR COMPARISON (All Methods)")
report_lines.append("=" * 70)
report_lines.append(var_summary_display.to_string(index=False))
report_lines.append("")
report_lines.append("Filtered Historical Simulation VaR (GARCH-based):")
for conf in [0.95, 0.99]:
    fhs_var = filtered_historical_var(port_ret, confidence=conf)
    report_lines.append(f"  {int(conf*100)}% confidence: {fhs_var*100:.3f}%")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("STUDENT-T DISTRIBUTION FIT")
report_lines.append("=" * 70)
report_lines.append(f"Estimated Degrees of Freedom (ν): {student_t_fit['df']:.2f}")
report_lines.append(f"Estimated Mean (μ):              {student_t_fit['mu']*100:.4f}%")
report_lines.append(f"Estimated Scale Parameter:       {student_t_fit['scale']*100:.3f}%")
report_lines.append(f"Sample Standard Deviation:      {student_t_fit['sample_std']*100:.3f}%")
report_lines.append(f"Optimization Status:             {'SUCCESS' if student_t_fit['optimization_success'] else 'FAILED'}")
if student_t_fit["excess_kurtosis"] == np.inf:
    report_lines.append("Implied Excess Kurtosis:       Infinite (ν <= 4)")
else:
    report_lines.append(f"Implied Excess Kurtosis:       {student_t_fit['excess_kurtosis']:.2f}")
report_lines.append("")
report_lines.append("Interpretation:")
if student_t_fit["df"] > 30:
    report_lines.append("  - ν > 30: approximately Gaussian")
elif student_t_fit["df"] >= 5:
    report_lines.append("  - ν between 5 and 15: realistic financial fat tails")
elif student_t_fit["df"] > 2:
    report_lines.append("  - ν between 2 and 5: extremely heavy-tailed returns")
else:
    report_lines.append("  - ν <= 2: variance does not exist (model invalid)")
report_lines.append("")
report_lines.append("Distribution Fit Comparison:")
report_lines.append(f"  Normal:    LogLik={fit_comparison['normal']['log_likelihood']:.1f}, AIC={fit_comparison['normal']['aic']:.1f}, BIC={fit_comparison['normal']['bic']:.1f}")
report_lines.append(f"  Student-t: LogLik={fit_comparison['student_t']['log_likelihood']:.1f}, AIC={fit_comparison['student_t']['aic']:.1f}, BIC={fit_comparison['student_t']['bic']:.1f}")
report_lines.append(f"  Best Fit: {fit_comparison['best_fit'].upper()}")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("EXPECTED SHORTFALL")
report_lines.append("=" * 70)
for _, row in var_summary.iterrows():
    conf_pct = int(row["confidence"] * 100)
    es = row["expected_shortfall_pct"]
    report_lines.append(f"At {conf_pct}% confidence: ES = {es:.3f}%")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("BACKTESTING")
report_lines.append("=" * 70)
report_lines.append("HISTORICAL VaR (95%, 250-day rolling window)")
report_lines.append("KUPIEC POF TEST")
for k, v in kupiec_result_hist.items():
    report_lines.append(f"  {k}: {v}")
verdict_kupiec_hist = "PASS" if kupiec_result_hist["model_adequate"] else "FAIL"
report_lines.append(f"  Verdict: {verdict_kupiec_hist}")
report_lines.append("")
report_lines.append("CHRISTOFFERSEN CONDITIONAL COVERAGE TEST")
for k, v in christoffersen_result_hist.items():
    if k != "transition_matrix":
        report_lines.append(f"  {k}: {v}")
verdict_christ_hist = "PASS" if christoffersen_result_hist["model_adequate"] else "FAIL"
report_lines.append(f"  Verdict: {verdict_christ_hist}")
report_lines.append("")
report_lines.append("FHS VaR (GARCH-based, 95%, 250-day rolling window)")
report_lines.append("KUPIEC POF TEST")
for k, v in kupiec_result_fhs.items():
    report_lines.append(f"  {k}: {v}")
verdict_kupiec_fhs = "PASS" if kupiec_result_fhs["model_adequate"] else "FAIL"
report_lines.append(f"  Verdict: {verdict_kupiec_fhs}")
report_lines.append("")
report_lines.append("CHRISTOFFERSEN CONDITIONAL COVERAGE TEST")
for k, v in christoffersen_result_fhs.items():
    if k != "transition_matrix":
        report_lines.append(f"  {k}: {v}")
verdict_christ_fhs = "PASS" if christoffersen_result_fhs["model_adequate"] else "FAIL"
report_lines.append(f"  Verdict: {verdict_christ_fhs}")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("DIVERSIFICATION ANALYSIS")
report_lines.append("=" * 70)
report_lines.append(f"Sum of Individual VaRs (95%): {div_analysis['sum_individual_vars_pct']:.3f}%")
report_lines.append(f"Portfolio VaR (95%): {div_analysis['portfolio_var_pct']:.3f}%")
report_lines.append(f"Diversification Benefit: {div_analysis['diversification_benefit_pct']:.3f}%")
report_lines.append(f"Diversification Efficiency: {div_analysis['diversification_efficiency']:.4f}")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("RISK CONTRIBUTIONS (Component VaR)")
report_lines.append("=" * 70)
report_lines.append(comp_var_df.to_string(index=False))
report_lines.append(f"Sum of Component VaRs: {comp_var_df['component_var_pct'].sum():.3f}%")
report_lines.append(f"Portfolio VaR: {portfolio_var*100:.3f}%")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("MARGINAL VaR (1% weight increase)")
report_lines.append("=" * 70)
report_lines.append(marg_var_df.to_string(index=False))
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("INCREMENTAL VaR (Position Removal)")
report_lines.append("=" * 70)
report_lines.append(inc_var_df.to_string(index=False))
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("STRESS TESTS")
report_lines.append("=" * 70)
report_lines.append("HYPOTHETICAL SCENARIOS")
report_lines.append(stress_df.to_string(index=False))
report_lines.append("")
report_lines.append("HISTORICAL SCENARIO REPLAY")
report_lines.append(historical_stress_df.to_string(index=False))
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("CHARTS GENERATED")
report_lines.append("=" * 70)
report_lines.append("  - return_distribution_var.png: Portfolio return distribution with VaR thresholds")
report_lines.append("  - rolling_var_backtest.png: Rolling VaR forecast vs actual returns")
report_lines.append("  - breach_heatmap.png: VaR breaches by month/year")
report_lines.append("  - stress_test_impact.png: Stress test scenario impact")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("NOTES")
report_lines.append("=" * 70)
report_lines.append("- Price data is sourced from Yahoo Finance (yfinance) using real historical")
report_lines.append("  adjusted close prices for the specified ETF tickers from 2010-01-01 onwards.")
report_lines.append("  This covers major market crises including 2008 GFC aftermath, 2011 EU debt crisis,")
report_lines.append("  2015-16 China slowdown, 2018 Q4 selloff, 2020 COVID, and 2022 rate hike cycle.")
report_lines.append("- Stress scenarios include both hypothetical shocks and historical replay.")
report_lines.append("- VaR methods include Historical, Parametric (Normal & Student-t), Monte Carlo (Normal & Student-t),")
report_lines.append("  and Filtered Historical Simulation (GARCH-based).")
report_lines.append("- Backtesting includes both Kupiec POF test and Christoffersen Conditional Coverage test.")
report_lines.append("- Risk attribution includes Component VaR (Euler), Marginal VaR, and Incremental VaR.")

report_text = "\n".join(report_lines)
with open(os.path.join(OUTDIR, "summary_report.txt"), "w") as f:
    f.write(report_text)

print(f"\nFull summary report written to ./{OUTDIR}/summary_report.txt")
