"""
var_models.py
-------------
Three standard Value-at-Risk methodologies applied at the portfolio level:

1. Historical Simulation VaR   - non-parametric, uses actual historical return distribution
2. Parametric (Variance-Covariance) VaR - assumes normally distributed returns
3. Monte Carlo VaR              - simulates returns from the estimated covariance structure

All VaR figures are reported as a POSITIVE number representing a loss
(e.g. VaR_95 = 0.023 means "a 1-day loss of 2.3% of portfolio value or worse
is expected with 5% probability").
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t


def portfolio_returns(returns_df, weights):
    """Combine asset returns into a single portfolio return series given weights."""
    weights = np.asarray(weights)
    assert np.isclose(weights.sum(), 1.0), "Portfolio weights must sum to 1"
    return returns_df.values @ weights


def historical_var(port_returns, confidence=0.95):
    """
    Historical simulation VaR: take the empirical quantile of realized portfolio
    returns. No distributional assumption - captures real fat tails / skew.
    """
    alpha = 1 - confidence
    var_return = np.percentile(port_returns, alpha * 100)
    return -var_return  # convert to positive loss figure


def historical_es(port_returns, confidence=0.95):
    """Expected Shortfall (CVaR): average loss beyond the VaR threshold."""
    alpha = 1 - confidence
    threshold = np.percentile(port_returns, alpha * 100)
    tail_losses = port_returns[port_returns <= threshold]
    return -tail_losses.mean()


def parametric_var(returns_df, weights, confidence=0.95, horizon_days=1):
    """
    Variance-Covariance (delta-normal) VaR. Assumes portfolio returns are
    normally distributed. Fast, closed-form, but understates tail risk for
    non-normal (fat-tailed) return series.
    """
    weights = np.asarray(weights)
    mu = returns_df.mean().values
    cov = returns_df.cov().values

    # port_mu = weights @ mu
    port_sigma = np.sqrt(weights @ cov @ weights)

    z = norm.ppf(1 - confidence)
    # var_1day = -(port_mu + z * port_sigma)
    var_1day = -(z * port_sigma)  # since the daily expected return is tiny we can ignore port_mu from VaR calculation
    return var_1day * np.sqrt(horizon_days)  # scale by sqrt(time) for multi-day horizon


def fit_student_t_distribution(portfolio_returns, df_bounds=(2.1, 30)):
    """
    Fit a Student-t distribution to portfolio returns via maximum likelihood estimation.
    
    This function estimates the degrees of freedom (ν), location (μ), and scale parameters
    of a Student-t distribution that best fits the observed returns. The estimated parameters
    are returned along with diagnostic information for model validation.
    
    Financial Interpretation:
    - ν (degrees of freedom): Controls tail heaviness. Lower ν = fatter tails.
      ν > 30: approximately Gaussian
      ν between 5-15: realistic financial fat tails
      ν between 2-4: extremely heavy-tailed returns
      ν <= 2: variance does not exist (model invalid)
    - μ (location): Mean of the distribution (expected daily return)
    - scale: Scale parameter related to volatility. For Student-t: variance = scale^2 * ν / (ν - 2)
    
    Parameters:
    -----------
    portfolio_returns : array-like
        Portfolio return series
    df_bounds : tuple
        (lower, upper) bounds for degrees of freedom optimization
    
    Returns:
    --------
    dict
        Dictionary containing fitted parameters and diagnostics:
        - df: estimated degrees of freedom (ν)
        - mu: estimated location parameter (μ)
        - scale: estimated scale parameter
        - sample_std: sample standard deviation of returns
        - log_likelihood: log-likelihood of fitted model
        - optimization_success: whether optimization converged
        - excess_kurtosis: implied excess kurtosis (6/(ν-4) if ν > 4, else infinite)
    """
    from scipy.optimize import minimize_scalar
    
    port_ret = np.asarray(portfolio_returns)
    sample_std = np.std(port_ret)
    sample_mean = np.mean(port_ret)
    
    def neg_log_likelihood(df_val):
        """
        Negative log-likelihood for Student-t distribution with MLE location and scale.
        
        For a given df, the MLE of location is the sample mean and the MLE of scale
        is derived from the sample standard deviation adjusted for the Student-t variance formula.
        """
        # df must be > 2 for variance to exist
        if df_val <= 2:
            return 1e10
        
        # MLE of location is sample mean
        mu = sample_mean
        
        # MLE of scale: relationship between sample std and scale for Student-t
        # variance = scale^2 * df / (df - 2)
        # Therefore: scale = sample_std * sqrt((df - 2) / df)
        sigma = sample_std * np.sqrt((df_val - 2) / df_val)
        
        # Compute log-likelihood
        log_lik = np.sum(t.logpdf(port_ret, df=df_val, loc=mu, scale=sigma))
        return -log_lik
    
    # Optimize degrees of freedom
    result = minimize_scalar(neg_log_likelihood, bounds=df_bounds, method='bounded')
    
    df = result.x
    optimization_success = result.success
    
    # Compute final parameters at optimal df
    mu = sample_mean
    scale = sample_std * np.sqrt((df - 2) / df)
    
    # Compute log-likelihood at optimal parameters
    log_likelihood = -neg_log_likelihood(df)
    
    # Compute implied excess kurtosis
    # For Student-t: excess kurtosis = 6 / (ν - 4) when ν > 4
    if df > 4:
        excess_kurtosis = 6 / (df - 4)
    else:
        excess_kurtosis = np.inf
    
    return {
        "df": df,
        "mu": mu,
        "scale": scale,
        "sample_std": sample_std,
        "log_likelihood": log_likelihood,
        "optimization_success": optimization_success,
        "excess_kurtosis": excess_kurtosis,
    }


def student_t_parametric_var(returns_df, weights, confidence=0.95, horizon_days=1, df=None):
    """
    Parametric VaR assuming Student's t-distribution for portfolio returns.
    
    The Student-t distribution has fatter tails than the normal distribution,
    making it more suitable for financial returns which exhibit excess kurtosis.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Asset returns
    weights : array-like
        Portfolio weights
    confidence : float
        Confidence level (e.g., 0.95 for 95% VaR)
    horizon_days : int
        VaR horizon in days
    df : float or None
        Degrees of freedom for t-distribution. If None, estimated from data
        using maximum likelihood via fit_student_t_distribution().
    
    Returns:
    --------
    float
        VaR as a positive loss percentage
    """
    weights = np.asarray(weights)
    port_ret = portfolio_returns(returns_df, weights)
    
    # Fit Student-t distribution (estimates df, mu, scale)
    fit_result = fit_student_t_distribution(port_ret)
    
    # Use estimated df if not provided, otherwise use provided df
    if df is None:
        df = fit_result["df"]
    
    # Get location and scale from fit
    mu = fit_result["mu"]
    scale = fit_result["scale"]
    
    # Validation: df must be > 2 for variance to exist
    if df <= 2:
        raise ValueError(f"Degrees of freedom ({df:.2f}) must be > 2 for variance to exist. Student-t model invalid.")
    
    # Student-t quantile
    t_quantile = t.ppf(1 - confidence, df=df)
    
    # VaR calculation (using t-distribution quantile)
    var_1day = -(mu + t_quantile * scale)
    
    return var_1day * np.sqrt(horizon_days)


def compare_distribution_fit(portfolio_returns):
    """
    Compare Normal and Student-t distribution fits to portfolio returns.
    
    Computes AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)
    for both distributions to determine which provides a better statistical fit.
    
    Lower AIC/BIC indicates better fit (penalizes model complexity).
    
    Parameters:
    -----------
    portfolio_returns : array-like
        Portfolio return series
    
    Returns:
    --------
    dict
        Dictionary containing fit comparison:
        - normal: dict with log_likelihood, AIC, BIC for normal distribution
        - student_t: dict with log_likelihood, AIC, BIC for Student-t distribution
        - best_fit: "normal" or "student_t" based on lowest AIC
    """
    port_ret = np.asarray(portfolio_returns)
    n = len(port_ret)
    
    # Fit Normal distribution
    mu_normal = np.mean(port_ret)
    sigma_normal = np.std(port_ret)
    log_lik_normal = np.sum(norm.logpdf(port_ret, loc=mu_normal, scale=sigma_normal))
    
    # Normal: 2 parameters (mean, std)
    k_normal = 2
    aic_normal = 2 * k_normal - 2 * log_lik_normal
    bic_normal = k_normal * np.log(n) - 2 * log_lik_normal
    
    # Fit Student-t distribution
    student_t_fit = fit_student_t_distribution(port_ret)
    log_lik_student_t = student_t_fit["log_likelihood"]
    
    # Student-t: 3 parameters (df, location, scale)
    k_student_t = 3
    aic_student_t = 2 * k_student_t - 2 * log_lik_student_t
    bic_student_t = k_student_t * np.log(n) - 2 * log_lik_student_t
    
    # Determine best fit (lower AIC is better)
    best_fit = "student_t" if aic_student_t < aic_normal else "normal"
    
    return {
        "normal": {
            "log_likelihood": log_lik_normal,
            "aic": aic_normal,
            "bic": bic_normal,
            "n_params": k_normal,
        },
        "student_t": {
            "log_likelihood": log_lik_student_t,
            "aic": aic_student_t,
            "bic": bic_student_t,
            "n_params": k_student_t,
            "df": student_t_fit["df"],
        },
        "best_fit": best_fit,
    }


def monte_carlo_var(returns_df, weights, confidence=0.95, n_sims=50_000, horizon_days=1, seed=7, distribution="normal"):
    """
    Monte Carlo VaR: simulate n_sims portfolio return paths by sampling from a
    multivariate distribution fitted to historical mean/covariance, then take the
    empirical quantile of the simulated portfolio returns.
    
    Parameters:
    -----------
    distribution : str
        "normal" for multivariate normal (default)
        "student" for multivariate Student-t with fatter tails
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights)
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    
    n_assets = len(weights)
    
    if distribution == "normal":
        sims = rng.multivariate_normal(mu, cov, size=n_sims)
        sim_port_returns = sims @ weights
    elif distribution == "student":
        # Multivariate Student-t using the method:
        # t = mu + (X / sqrt(Z/df)) where X ~ N(0, cov), Z ~ Chi2(df)
        # Estimate df from portfolio returns
        port_ret = portfolio_returns(returns_df, weights)
        from scipy.optimize import minimize_scalar
        
        def neg_log_likelihood(df_val):
            if df_val <= 2:
                return 1e10
            sigma = np.std(port_ret) * np.sqrt((df_val - 2) / df_val)
            log_lik = np.sum(t.logpdf(port_ret, df=df_val, loc=0, scale=sigma))
            return -log_lik
        
        result = minimize_scalar(neg_log_likelihood, bounds=(2.1, 30), method='bounded')
        df = result.x
        
        # Generate multivariate Student-t
        # Step 1: Generate chi-squared random variable
        Z = rng.chisquare(df, size=n_sims)
        # Step 2: Generate multivariate normal
        X = rng.multivariate_normal(np.zeros(n_assets), cov, size=n_sims)
        # Step 3: Combine to get Student-t
        scaling_factor = np.sqrt(df / Z)[:, np.newaxis]
        sims = mu + X * scaling_factor
        sim_port_returns = sims @ weights
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Use 'normal' or 'student'.")

    alpha = 1 - confidence
    var_1day = -np.percentile(sim_port_returns, alpha * 100)
    return var_1day * np.sqrt(horizon_days), sim_port_returns


def summarize_var(returns_df, weights, confidences=(0.95, 0.99)):
    """
    Produce a summary table comparing all VaR methods at given confidence levels.
    Includes: Historical, Parametric Normal, Parametric Student-t, MC Normal, MC Student-t.
    """
    port_ret = portfolio_returns(returns_df, weights)
    rows = []
    for c in confidences:
        hist = historical_var(port_ret, c)
        es = historical_es(port_ret, c)
        param = parametric_var(returns_df, weights, c)
        param_t = student_t_parametric_var(returns_df, weights, c)
        mc_normal, _ = monte_carlo_var(returns_df, weights, c, distribution="normal")
        mc_student, _ = monte_carlo_var(returns_df, weights, c, distribution="student")
        rows.append({
            "confidence": c,
            "historical_var_pct": hist * 100,
            "expected_shortfall_pct": es * 100,
            "parametric_normal_pct": param * 100,
            "parametric_student_t_pct": param_t * 100,
            "mc_normal_pct": mc_normal * 100,
            "mc_student_t_pct": mc_student * 100,
        })
    return pd.DataFrame(rows), port_ret


def filtered_historical_var(port_returns, confidence=0.95, horizon_days=1, n_bootstrap=1000, seed=42):
    """
    GARCH-based Filtered Historical Simulation (FHS) VaR.
    
    This method accounts for time-varying volatility by:
    1. Fitting a GARCH(1,1) model to portfolio returns
    2. Computing conditional volatility series
    3. Standardizing returns by conditional volatility
    4. Bootstrapping standardized residuals
    5. Scaling by today's conditional volatility
    6. Computing VaR from the filtered returns
    
    FHS captures volatility clustering and provides more accurate VaR during
    turbulent periods compared to standard historical simulation.
    
    Parameters:
    -----------
    port_returns : array-like
        Portfolio return series
    confidence : float
        Confidence level (e.g., 0.95 for 95% VaR)
    horizon_days : int
        VaR horizon in days
    n_bootstrap : int
        Number of bootstrap samples for FHS
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    float
        FHS VaR as a positive loss percentage
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("arch package is required for FHS. Install with: pip install arch")
    
    rng = np.random.default_rng(seed)
    returns = np.asarray(port_returns)
    
    # Fit GARCH(1,1) model to returns
    # Use mean='Zero' since we're interested in volatility, not mean
    model = arch_model(returns * 100, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
    res = model.fit(disp='off')
    
    # Get conditional volatility (convert back from percentage)
    conditional_vol = res.conditional_volatility / 100
    
    # Standardize returns by conditional volatility
    # Add small epsilon to avoid division by zero
    epsilon = 1e-12
    standardized_returns = returns / (conditional_vol + epsilon)
    
    # Get today's conditional volatility (last value)
    current_vol = conditional_vol[-1]
    
    # Bootstrap standardized residuals
    n_obs = len(standardized_returns)
    bootstrapped_residuals = rng.choice(standardized_returns, size=(n_bootstrap, n_obs), replace=True)
    
    # Scale by today's volatility to get filtered returns
    filtered_returns = bootstrapped_residuals * current_vol
    
    # Compute portfolio returns from filtered returns (sum across horizon)
    if horizon_days > 1:
        # For multi-day horizon, sum independent daily returns
        filtered_returns = np.sum(filtered_returns[:, :horizon_days], axis=1)
    else:
        filtered_returns = filtered_returns[:, 0]
    
    # Compute VaR from filtered returns
    alpha = 1 - confidence
    var = -np.percentile(filtered_returns, alpha * 100)
    
    return var


def diversification_analysis(returns_df, weights, confidence=0.95):
    """
    Compute diversification metrics for the portfolio.
    
    Diversification benefit arises from imperfect correlations between assets.
    This function computes:
    - Standalone VaR for each asset
    - Sum of individual VaRs (undiversified VaR)
    - Portfolio VaR (diversified VaR)
    - Diversification benefit (reduction in VaR due to diversification)
    - Diversification Efficiency (portfolio VaR / sum of individual VaRs)
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Asset returns
    weights : array-like
        Portfolio weights
    confidence : float
        Confidence level for VaR calculation
    
    Returns:
    --------
    dict
        Dictionary containing diversification metrics
    """
    weights = np.asarray(weights)
    n_assets = len(weights)
    
    # Compute standalone VaR for each asset (using historical simulation)
    standalone_vars = []
    for i in range(n_assets):
        asset_returns = returns_df.iloc[:, i].values
        var_i = historical_var(asset_returns, confidence)
        standalone_vars.append(var_i)
    
    standalone_vars = np.array(standalone_vars)
    
    # Sum of individual VaRs (undiversified)
    sum_individual_vars = np.sum(standalone_vars)
    
    # Portfolio VaR (diversified)
    port_ret = portfolio_returns(returns_df, weights)
    portfolio_var = historical_var(port_ret, confidence)
    
    # Diversification benefit
    div_benefit = sum_individual_vars - portfolio_var
    
    # Diversification ratio (lower is better - more diversification)
    div_ratio = portfolio_var / sum_individual_vars if sum_individual_vars > 0 else np.nan
    
    return {
        "standalone_vars": standalone_vars * 100,  # as percentages
        "sum_individual_vars_pct": sum_individual_vars * 100,
        "portfolio_var_pct": portfolio_var * 100,
        "diversification_benefit_pct": div_benefit * 100,
        "diversification_efficiency": div_ratio,
    }


def component_var(returns_df, weights, confidence=0.95):
    """
    Compute Component VaR using Euler allocation.
    
    Component VaR decomposes portfolio VaR into contributions from each asset.
    It satisfies the property that sum(Component VaR) = Portfolio VaR.
    
    Component VaR_i = weight_i * Marginal VaR_i
    Marginal VaR_i = ∂VaR/∂weight_i
    
    For parametric normal VaR:
    Marginal VaR_i = -z * (cov_matrix @ weights)_i / portfolio_sigma
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Asset returns
    weights : array-like or dict
        Portfolio weights (array in same order as returns_df.columns, or dict mapping asset->weight)
    confidence : float
        Confidence level for VaR calculation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Component VaR for each asset
    float
        Portfolio VaR
    """
    # Handle weights as dict or array - enforce strict alignment
    if isinstance(weights, dict):
        # Convert dict to Series aligned with DataFrame columns
        weights_series = pd.Series(weights)
        # Reindex to match returns_df columns (strict alignment)
        weights_aligned = weights_series.reindex(returns_df.columns)
        if weights_aligned.isna().any():
            missing = weights_series.index[~weights_series.index.isin(returns_df.columns)]
            raise ValueError(f"Weights dict contains assets not in returns_df: {missing.tolist()}")
        weights = weights_aligned.values
    else:
        weights = np.asarray(weights)
        # Validate length matches columns
        if len(weights) != len(returns_df.columns):
            raise ValueError(f"Weights array length ({len(weights)}) does not match number of assets ({len(returns_df.columns)})")
    
    # Compute covariance matrix (preserves column order)
    cov = returns_df.cov().values
    
    # Portfolio volatility: sigma_p = sqrt(w^T * Sigma * w)
    port_sigma = np.sqrt(weights @ cov @ weights)
    
    # Z-score for confidence level
    z = norm.ppf(1 - confidence)
    
    # Marginal VaR vector: MVaR = z * (Sigma * w) / sigma_p
    marginal_vars = -z * (cov @ weights) / port_sigma
    
    # Component VaR: CVaR = w ⊙ MVaR (element-wise multiplication)
    component_vars = weights * marginal_vars
    
    # Portfolio VaR (parametric normal for consistency)
    portfolio_var = parametric_var(returns_df, weights, confidence)
    
    # Percentage contribution
    pct_contributions = (component_vars / portfolio_var) * 100
    
    # Create result DataFrame with strict asset alignment
    result = pd.DataFrame({
        "asset": returns_df.columns.tolist(),
        "weight": weights,
        "marginal_var_pct": marginal_vars * 100,
        "component_var_pct": component_vars * 100,
        "pct_contribution": pct_contributions,
    })
    
    return result, portfolio_var


def marginal_var(returns_df, weights, confidence=0.95, delta=0.01, method="analytical"):
    """
    Compute Marginal VaR using analytical formula or finite differences.
    
    Marginal VaR measures the change in portfolio VaR when an asset's weight
    is increased by a small amount (holding other weights proportional).
    
    Analytical formula (recommended for stability):
    MVaR_i = -z * (Sigma @ w)_i / sigma_p
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Asset returns
    weights : array-like or dict
        Portfolio weights (array in same order as returns_df.columns, or dict mapping asset->weight)
    confidence : float
        Confidence level for VaR calculation
    delta : float
        Small change in weight (only used if method="finite_difference")
    method : str
        "analytical" (default, stable) or "finite_difference" (numerical)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Marginal VaR for each asset
    """
    # Handle weights as dict or array - enforce strict alignment
    if isinstance(weights, dict):
        weights_series = pd.Series(weights)
        weights_aligned = weights_series.reindex(returns_df.columns)
        if weights_aligned.isna().any():
            missing = weights_series.index[~weights_series.index.isin(returns_df.columns)]
            raise ValueError(f"Weights dict contains assets not in returns_df: {missing.tolist()}")
        weights = weights_aligned.values
    else:
        weights = np.asarray(weights)
        if len(weights) != len(returns_df.columns):
            raise ValueError(f"Weights array length ({len(weights)}) does not match number of assets ({len(returns_df.columns)})")
    
    assets = returns_df.columns.tolist()
    
    if method == "analytical":
        # Use analytical formula for numerical stability
        cov = returns_df.cov().values
        port_sigma = np.sqrt(weights @ cov @ weights)
        z = norm.ppf(1 - confidence)
        
        # Marginal VaR vector: MVaR = -z * (Sigma @ w) / sigma_p
        marginal_vars = -z * (cov @ weights) / port_sigma
        
        result = pd.DataFrame({
            "asset": assets,
            "weight": weights,
            "marginal_var_pct": marginal_vars * 100,
            "var_impact_1pct_increase_pct": marginal_vars * delta * 100,
        })
    else:
        # Finite difference method (less stable, for validation)
        n_assets = len(weights)
        port_ret = portfolio_returns(returns_df, weights)
        base_var = historical_var(port_ret, confidence)
        
        marginal_vars = []
        
        for i in range(n_assets):
            # Increase weight of asset i by delta, decrease others proportionally
            new_weights = weights.copy()
            new_weights[i] += delta
            
            # Decrease other weights proportionally to maintain sum = 1
            other_indices = [j for j in range(n_assets) if j != i]
            total_other = weights[other_indices].sum()
            if total_other > 0:
                new_weights[other_indices] = weights[other_indices] * (1 - delta) / total_other
            
            # Explicitly normalize to ensure sum = 1
            new_weights = new_weights / new_weights.sum()
            
            # Compute new VaR
            new_port_ret = portfolio_returns(returns_df, new_weights)
            new_var = historical_var(new_port_ret, confidence)
            
            # Marginal VaR = change in VaR / change in weight
            marg_var = (new_var - base_var) / delta
            marginal_vars.append(marg_var)
        
        result = pd.DataFrame({
            "asset": assets,
            "weight": weights,
            "marginal_var_pct": np.array(marginal_vars) * 100,
            "var_impact_1pct_increase_pct": np.array(marginal_vars) * delta * 100,
        })
    
    return result


def incremental_var(returns_df, weights, confidence=0.95):
    """
    Compute Incremental VaR for removing each position.
    
    Incremental VaR measures the change in portfolio VaR when an asset is
    completely removed from the portfolio (weight set to 0, others rescaled).
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Asset returns
    weights : array-like or dict
        Portfolio weights (array in same order as returns_df.columns, or dict mapping asset->weight)
    confidence : float
        Confidence level for VaR calculation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Incremental VaR for each asset
    """
    # Handle weights as dict or array - enforce strict alignment
    if isinstance(weights, dict):
        weights_series = pd.Series(weights)
        weights_aligned = weights_series.reindex(returns_df.columns)
        if weights_aligned.isna().any():
            missing = weights_series.index[~weights_series.index.isin(returns_df.columns)]
            raise ValueError(f"Weights dict contains assets not in returns_df: {missing.tolist()}")
        weights = weights_aligned.values
    else:
        weights = np.asarray(weights)
        if len(weights) != len(returns_df.columns):
            raise ValueError(f"Weights array length ({len(weights)}) does not match number of assets ({len(returns_df.columns)})")
    
    n_assets = len(weights)
    assets = returns_df.columns.tolist()
    
    # Base portfolio VaR
    port_ret = portfolio_returns(returns_df, weights)
    base_var = historical_var(port_ret, confidence)
    
    incremental_vars = []
    new_vars = []
    
    for i in range(n_assets):
        # Remove asset i (set weight to 0)
        new_weights = weights.copy()
        new_weights[i] = 0
        
        # Rescale remaining weights to sum to 1
        total_remaining = new_weights.sum()
        if total_remaining > 0:
            new_weights = new_weights / total_remaining
        
        # Compute new VaR
        new_port_ret = portfolio_returns(returns_df, new_weights)
        new_var = historical_var(new_port_ret, confidence)
        
        # Incremental VaR = base VaR - new VaR (positive means risk reduction)
        inc_var = base_var - new_var
        incremental_vars.append(inc_var)
        new_vars.append(new_var)
    
    result = pd.DataFrame({
        "asset": assets,
        "current_weight": weights,
        "portfolio_var_pct": base_var * 100,
        "portfolio_var_without_asset_pct": np.array(new_vars) * 100,
        "incremental_var_pct": np.array(incremental_vars) * 100,
    })
    
    return result
