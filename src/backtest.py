"""
backtest.py
-----------
Kupiec Proportion-of-Failures (POF) backtest: checks whether the observed
rate of VaR breaches over a historical window is statistically consistent
with the VaR model's stated confidence level.

This is the standard first-line model-validation test regulators and risk
teams use to sanity-check a VaR model before relying on it.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def rolling_var_forecast(returns_series, window=250, confidence=0.95):
    """
    Produce a rolling 1-day-ahead historical VaR forecast: for each day t,
    use the trailing `window` days of returns (up to t-1) to forecast VaR
    for day t, then compare against the actual realized return on day t.
    """
    alpha = 1 - confidence
    forecasts, actuals, breaches = [], [], []

    values = returns_series.values
    for t in range(window, len(values)):
        hist_window = values[t - window: t]
        var_forecast = -np.percentile(hist_window, alpha * 100)
        actual_return = values[t]
        breach = actual_return < -var_forecast

        forecasts.append(var_forecast)
        actuals.append(actual_return)
        breaches.append(breach)

    return np.array(forecasts), np.array(actuals), np.array(breaches)


def rolling_fhs_forecast(returns_series, window=250, confidence=0.95, n_bootstrap=1000, seed=42):
    """
    Produce a rolling 1-day-ahead Filtered Historical Simulation (FHS) VaR forecast.
    
    For each day t, fit a GARCH(1,1) model to the trailing `window` days of returns,
    compute the conditional volatility, and use FHS to forecast VaR for day t.
    This produces dynamic VaR limits that account for volatility clustering.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Portfolio returns with DatetimeIndex
    window : int
        Rolling window size for GARCH fitting
    confidence : float
        Confidence level for VaR
    n_bootstrap : int
        Number of bootstrap samples for FHS
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (forecasts, actuals, breaches) arrays for the forecast period
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("arch package is required for FHS backtesting. Install with: pip install arch")
    
    import numpy as np
    
    alpha = 1 - confidence
    forecasts, actuals, breaches = [], [], []
    
    values = returns_series.values
    rng = np.random.default_rng(seed)
    
    for t in range(window, len(values)):
        # Fit GARCH(1,1) to trailing window
        hist_window = values[t - window: t]
        
        try:
            # Fit GARCH model (use mean='Zero' for volatility focus)
            model = arch_model(hist_window * 100, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
            res = model.fit(disp='off')
            
            # Get conditional volatility (convert back from percentage)
            conditional_vol = res.conditional_volatility / 100
            
            # Standardize returns by conditional volatility
            epsilon = 1e-12
            standardized_returns = hist_window / (conditional_vol + epsilon)
            
            # Get today's conditional volatility (last value)
            current_vol = conditional_vol[-1]
            
            # Bootstrap standardized residuals
            n_obs = len(standardized_returns)
            bootstrapped_residuals = rng.choice(standardized_returns, size=(n_bootstrap, n_obs), replace=True)
            
            # Scale by today's volatility to get filtered returns
            filtered_returns = bootstrapped_residuals * current_vol
            
            # Compute VaR from filtered returns
            var_forecast = -np.percentile(filtered_returns, alpha * 100)
            
            actual_return = values[t]
            breach = actual_return < -var_forecast
            
            forecasts.append(var_forecast)
            actuals.append(actual_return)
            breaches.append(breach)
            
        except Exception as e:
            # Fallback to historical VaR if GARCH fails
            var_forecast = -np.percentile(hist_window, alpha * 100)
            actual_return = values[t]
            breach = actual_return < -var_forecast
            
            forecasts.append(var_forecast)
            actuals.append(actual_return)
            breaches.append(breach)
    
    return np.array(forecasts), np.array(actuals), np.array(breaches)


def kupiec_pof_test(breaches, confidence=0.95):
    """
    Kupiec's Proportion-of-Failures likelihood-ratio test.

    H0: the observed breach rate equals the expected breach rate (1 - confidence).
    Returns the LR statistic, p-value, and a pass/fail flag at the 95% test level.
    
    Uses log-likelihoods to avoid floating-point underflow for large datasets.
    """
    n = len(breaches)
    x = breaches.sum()  # number of breaches observed
    p = 1 - confidence  # expected breach probability
    
    # Handle edge cases with epsilon clipping
    epsilon = 1e-12
    
    if x == 0:
        # No breaches observed - use tiny epsilon for observed rate
        pi_hat = epsilon
    elif x == n:
        # All observations are breaches - use 1 - epsilon
        pi_hat = 1.0 - epsilon
    else:
        pi_hat = x / n
    
    # Clip probabilities to avoid log(0)
    p_clipped = np.clip(p, epsilon, 1.0 - epsilon)
    pi_hat_clipped = np.clip(pi_hat, epsilon, 1.0 - epsilon)
    
    # Compute log-likelihoods to avoid underflow
    # LL_H0: log-likelihood under null hypothesis (breach rate = p)
    LL_H0 = x * np.log(p_clipped) + (n - x) * np.log(1.0 - p_clipped)
    
    # LL_H1: log-likelihood under alternative (breach rate = pi_hat)
    LL_H1 = x * np.log(pi_hat_clipped) + (n - x) * np.log(1.0 - pi_hat_clipped)
    
    # Likelihood ratio statistic
    lr_stat = -2 * (LL_H0 - LL_H1)
    
    # Ensure lr_stat is non-negative (numerical stability)
    lr_stat = max(lr_stat, 0.0)
    
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    model_ok = p_value > 0.05  # fail to reject H0 at 5% significance -> model is adequate

    return {
        "n_observations": n,
        "n_breaches": int(x),
        "expected_breaches": round(n * p, 2),
        "breach_rate_pct": round(pi_hat * 100, 3),
        "expected_rate_pct": round(p * 100, 3),
        "lr_statistic": round(lr_stat, 4),
        "p_value": round(p_value, 4),
        "model_adequate": bool(model_ok),
    }


def christoffersen_test(breaches, confidence=0.95):
    """
    Christoffersen's Conditional Coverage test for independence of VaR breaches.
    
    Unlike Kupiec which only checks the correct number of breaches, Christoffersen
    tests whether breaches occur independently (i.e., not clustered in time).
    This is important because clustered breaches indicate the model fails during
    stress periods - exactly when it's needed most.
    
    The test uses a first-order Markov chain with two states: breach (1) and no breach (0).
    It tests whether the transition probabilities differ from the unconditional breach rate.
    
    H0: Breaches are independent (transition matrix has equal probabilities)
    Returns LR statistics for independence and conditional coverage, with p-values.
    """
    n = len(breaches)
    
    if n < 2:
        return {
            "n_observations": n,
            "lr_independence": np.nan,
            "p_value_independence": np.nan,
            "lr_conditional": np.nan,
            "p_value_conditional": np.nan,
            "model_adequate": False,
            "note": "Insufficient observations for Christoffersen test"
        }
    
    # Build transition matrix
    # n00: transitions from 0 to 0, n01: from 0 to 1, n10: from 1 to 0, n11: from 1 to 1
    n00 = n01 = n10 = n11 = 0
    
    for i in range(1, n):
        if breaches[i-1] == 0 and breaches[i] == 0:
            n00 += 1
        elif breaches[i-1] == 0 and breaches[i] == 1:
            n01 += 1
        elif breaches[i-1] == 1 and breaches[i] == 0:
            n10 += 1
        elif breaches[i-1] == 1 and breaches[i] == 1:
            n11 += 1
    
    n0_ = n00 + n01  # total transitions from state 0
    n1_ = n10 + n11  # total transitions from state 1
    n_0 = n00 + n10  # total transitions to state 0
    n_1 = n01 + n11  # total transitions to state 1
    
    # Handle edge cases with epsilon
    epsilon = 1e-12
    
    # Unconditional probabilities (under independence)
    pi_0 = n_0 / (n_0 + n_1) if (n_0 + n_1) > 0 else 0.5
    pi_1 = n_1 / (n_0 + n_1) if (n_0 + n_1) > 0 else 0.5
    
    # Conditional probabilities (observed)
    pi_01 = n01 / n0_ if n0_ > 0 else epsilon
    pi_11 = n11 / n1_ if n1_ > 0 else epsilon
    pi_00 = 1 - pi_01
    pi_10 = 1 - pi_11
    
    # Clip to avoid log(0)
    pi_0 = np.clip(pi_0, epsilon, 1.0 - epsilon)
    pi_1 = np.clip(pi_1, epsilon, 1.0 - epsilon)
    pi_00 = np.clip(pi_00, epsilon, 1.0 - epsilon)
    pi_01 = np.clip(pi_01, epsilon, 1.0 - epsilon)
    pi_10 = np.clip(pi_10, epsilon, 1.0 - epsilon)
    pi_11 = np.clip(pi_11, epsilon, 1.0 - epsilon)
    
    # Log-likelihood under independence (transitions follow unconditional probabilities)
    LL_ind = (n00 * np.log(pi_0) + n01 * np.log(pi_1) +
               n10 * np.log(pi_0) + n11 * np.log(pi_1))
    
    # Log-likelihood under dependence (transitions follow conditional probabilities)
    LL_dep = (n00 * np.log(pi_00) + n01 * np.log(pi_01) +
               n10 * np.log(pi_10) + n11 * np.log(pi_11))
    
    # LR independence test
    LR_ind = -2 * (LL_ind - LL_dep)
    LR_ind = max(LR_ind, 0.0)
    p_ind = 1 - chi2.cdf(LR_ind, df=1)
    
    # LR conditional coverage test (combines independence with Kupiec)
    # This tests both correct frequency AND independence
    x = breaches.sum()
    p = 1 - confidence
    
    if x == 0:
        pi_hat = epsilon
    elif x == n:
        pi_hat = 1.0 - epsilon
    else:
        pi_hat = x / n
    
    p_clipped = np.clip(p, epsilon, 1.0 - epsilon)
    pi_hat_clipped = np.clip(pi_hat, epsilon, 1.0 - epsilon)
    
    LL_uncond = x * np.log(p_clipped) + (n - x) * np.log(1.0 - p_clipped)
    LL_cond = x * np.log(pi_hat_clipped) + (n - x) * np.log(1.0 - pi_hat_clipped)
    LR_cc = -2 * (LL_uncond - LL_cond) + LR_ind
    LR_cc = max(LR_cc, 0.0)
    p_cc = 1 - chi2.cdf(LR_cc, df=2)
    
    model_ok = p_ind > 0.05 and p_cc > 0.05
    
    return {
        "n_observations": n,
        "n_breaches": int(x),
        "transition_matrix": {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "pi_01": round(pi_01, 4), "pi_11": round(pi_11, 4)
        },
        "lr_independence": round(LR_ind, 4),
        "p_value_independence": round(p_ind, 4),
        "lr_conditional": round(LR_cc, 4),
        "p_value_conditional": round(p_cc, 4),
        "model_adequate": bool(model_ok),
    }
