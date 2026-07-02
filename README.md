📊 Portfolio Risk Analytics Engine
==================================

### A production-inspired market risk analytics engine for portfolio Value-at-Risk, Expected Shortfall, stress testing, statistical backtesting, and risk attribution using real financial market data.

![Python](https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Market Risk](https://img.shields.io/badge/Domain-Market%20Risk-darkgreen?style=for-the-badge) ![Quantitative Finance](https://img.shields.io/badge/Quantitative-Finance-blueviolet?style=for-the-badge) ![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

* * * * *

**Historical Simulation - Parametric VaR - Student-t VaR - Monte Carlo Simulation - GARCH-based Filtered Historical Simulation - Expected Shortfall - Kupiec & Christoffersen Backtesting - Historical Scenario Replay - Portfolio Risk Attribution**

📑 Table of Contents
--------------------

-   [Overview](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#overview)
-   [Features](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-features)
-   [Example Portfolio & Key Results](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-example-portfolio--key-results)
-   [Visualizations](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-visualizations)
-   [Methodology](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-methodology)
    -   [Portfolio Return Calculation](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#1-portfolio-return-calculation)
    -   [Historical Simulation VaR](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#2-historical-simulation-var)
    -   [Expected Shortfall](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#3-expected-shortfall-conditional-var)
    -   [Parametric VaR](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#4-parametric-variance-covariance-var)
    -   [Monte Carlo Simulation](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#5-monte-carlo-simulation)
    -   [Filtered Historical Simulation](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#6-filtered-historical-simulation-fhs)
    -   [Statistical Backtesting](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-statistical-backtesting)
    -   [Stress Testing](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-stress-testing)
    -   [Portfolio Risk Attribution](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-portfolio-risk-attribution)
-   [Architecture](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-architecture)
-   [Project Structure](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-project-structure)
-   [Installation](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#%EF%B8%8F-installation)
-   [Running the Project](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-running-the-project)
-   [References](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-references)
-   [Future Roadmap](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-future-roadmap)
-   [License](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/README.md#-license)

🔍 Quick Navigation
-------------------

-   📊 **Risk Models:** Historical, Parametric (Gaussian & Student-t), Monte Carlo, GARCH-FHS
-   📉 **Tail Risk:** Expected Shortfall (CVaR)
-   ✅ **Backtesting:** Kupiec POF & Christoffersen Conditional Coverage
-   🌍 **Stress Testing:** Historical Replay & Hypothetical Scenarios
-   🧮 **Risk Attribution:** Component, Marginal & Incremental VaR
-   📈 **Visualizations:** Rolling VaR, Return Distribution, Breach Heatmap, Stress Impact

Overview
--------

Modern financial institutions rely on **Value-at-Risk (VaR)** and **stress testing** to quantify potential portfolio losses under normal and extreme market conditions. While many academic implementations stop at a single VaR methodology, this project implements a **comprehensive market risk analytics framework** inspired by industry practices.

Using **real historical market data** from Yahoo Finance, the engine supports multiple VaR models, volatility-adjusted risk estimation, statistical model validation, historical crisis replay, and portfolio risk decomposition.

The project is designed to demonstrate both **quantitative finance concepts** and **software engineering practices**, making it suitable for quantitative risk, market risk, and financial engineering applications.

✨ Highlights
------------

-   📈 **6 portfolio risk models**

    -   Historical Simulation
    -   Parametric (Gaussian)
    -   Parametric (Student-t)
    -   Monte Carlo (Gaussian)
    -   Monte Carlo (Student-t)
    -   GARCH-based Filtered Historical Simulation (FHS)
-   📉 **Expected Shortfall (CVaR)** for tail risk measurement

-   📊 **Statistical backtesting**

    -   Kupiec Proportion of Failures Test
    -   Christoffersen Conditional Coverage Test
-   🌍 **Historical crisis replay**

    -   COVID-19 Crash
    -   2018 Q4 Selloff
    -   2015 China Slowdown
    -   2011 European Debt Crisis
    -   2022 Rate Hike Cycle
-   ⚠️ **Hypothetical stress testing** across multiple market scenarios

-   🧮 **Portfolio risk attribution**

    -   Component VaR
    -   Marginal VaR
    -   Incremental VaR
    -   Diversification Analysis
-   📈 **Automatic visualization**

    -   Return distributions
    -   Rolling VaR forecasts
    -   VaR breach heatmaps
    -   Stress testing charts
-   📝 **Comprehensive report generation** summarizing all risk metrics

🚀 Features
-----------

| Category | Capability | Description |
| --- | --- | --- |
| 📊 **Risk Models** | Historical Simulation VaR | Non-parametric VaR using empirical return distributions. |
|  | Parametric (Gaussian) VaR | Variance-Covariance VaR under multivariate normal assumptions. |
|  | Parametric (Student-t) VaR | Heavy-tailed VaR with maximum likelihood estimation of Student-t parameters. |
|  | Monte Carlo (Gaussian) | Simulates correlated portfolio returns using a multivariate Gaussian distribution. |
|  | Monte Carlo (Student-t) | Heavy-tailed Monte Carlo simulation preserving portfolio covariance structure. |
|  | Filtered Historical Simulation (FHS) | GARCH(1,1)-based volatility-adjusted Historical Simulation. |
| 📉 **Tail Risk** | Expected Shortfall (CVaR) | Computes average loss beyond the VaR threshold. |
| 📈 **Backtesting** | Kupiec Proportion of Failures Test | Validates whether observed VaR breaches match expected breach frequency. |
|  | Christoffersen Conditional Coverage Test | Tests both unconditional coverage and independence of VaR breaches. |
| 🌍 **Stress Testing** | Hypothetical Scenarios | User-defined market shocks (e.g., GFC, COVID, commodity crash, interest-rate shocks). |
|  | Historical Scenario Replay | Replays real historical market crises using actual ETF returns. |
| 🧮 **Risk Attribution** | Component VaR | Euler decomposition of portfolio VaR into asset-level contributions. |
|  | Marginal VaR | Measures sensitivity of portfolio VaR to a small increase in asset weight. |
|  | Incremental VaR | Measures the impact of removing an asset from the portfolio. |
| 📦 **Portfolio Analytics** | Diversification Analysis | Quantifies diversification benefit and portfolio risk efficiency. |
| 📊 **Visualization** | Return Distribution | Histogram with Historical, Parametric, and Monte Carlo VaR thresholds. |
|  | Rolling VaR Backtest | Rolling VaR forecast with realized returns and breach indicators. |
|  | Breach Heatmap | Monthly visualization of VaR exceptions. |
|  | Stress Test Impact | Comparative visualization of hypothetical and historical stress scenarios. |
| 📄 **Reporting** | Comprehensive Risk Report | Automatically generates a structured report containing portfolio summary, VaR comparison, Student-t diagnostics, Expected Shortfall, backtesting, risk attribution, stress testing, historical replay, and visualization summaries. |
| 📡 **Market Data** | Yahoo Finance Integration | Downloads real historical ETF prices using `yfinance` for reproducible analysis. |

### 📌 At a Glance

-   ✅ **6** Value-at-Risk methodologies
-   ✅ **2** statistical backtesting procedures
-   ✅ **2** Monte Carlo simulation engines
-   ✅ **2** stress testing frameworks
-   ✅ **3** portfolio risk attribution techniques
-   ✅ **4** automatically generated visualizations
-   ✅ Real market data from **Yahoo Finance**
-   ✅ End-to-end report generation

📈 Example Portfolio & Key Results
==================================

The engine was evaluated on a diversified multi-asset ETF portfolio using **real historical daily market data** downloaded from **Yahoo Finance**.

Example Portfolio
-----------------

| Asset | Description | Weight |
| --- | --- | --- |
| SPY | S&P 500 ETF | 25% |
| QQQ | Nasdaq-100 ETF | 15% |
| TLT | 20+ Year Treasury Bond ETF | 15% |
| GLD | Gold ETF | 10% |
| USO | United States Oil Fund | 5% |
| HYG | High Yield Corporate Bond ETF | 10% |
| EEM | Emerging Markets ETF | 10% |
| VNQ | Real Estate ETF | 10% |

**Portfolio Value:** **$1,000,000**

**Historical Data:** 4,149 trading days (2010 -- Present)

* * * * *

Key Risk Metrics
================

| Metric | Result |
| --- | --- |
| Historical VaR (95%) | **1.168%** |
| Historical VaR (99%) | **2.061%** |
| Expected Shortfall (95%) | **1.823%** |
| Expected Shortfall (99%) | **3.147%** |
| Best Backtested Model | **Filtered Historical Simulation (GARCH)** |
| Student-t Degrees of Freedom | **ν = 3.72** |
| Worst Historical Scenario | **COVID Crash** |
| Maximum Historical Loss | **-19.65%** |

* * * * *

VaR Model Comparison
--------------------

| Model | 95% VaR | 99% VaR |
| --- | --- | --- |
| Historical Simulation | **1.168%** | **2.061%** |
| Parametric (Gaussian) | 1.282% | 1.813% |
| Parametric (Student-t) | 1.123% | 2.039% |
| Monte Carlo (Gaussian) | 1.248% | 1.766% |
| Monte Carlo (Student-t) | 1.642% | 2.999% |
| Filtered Historical Simulation | **1.700%** | **2.788%** |

* * * * *

Model Validation
----------------

| Model | Kupiec Test | Christoffersen Test |
| --- | --- | --- |
| Historical Simulation | ✅ PASS | ❌ FAIL |
| Filtered Historical Simulation | ✅ PASS | ✅ PASS |

> **Observation:** Historical Simulation captures unconditional coverage but fails to model volatility clustering. The GARCH-based Filtered Historical Simulation passes both statistical backtests, demonstrating improved responsiveness to changing market regimes.

* * * * *

Portfolio Risk Attribution
--------------------------

Largest contributors to portfolio risk:

| Asset | Contribution |
| --- | --- |
| SPY | **33.3%** |
| QQQ | **23.0%** |
| EEM | **15.1%** |
| VNQ | **13.3%** |

Portfolio hedges:

| Asset | Contribution |
| --- | --- |
| TLT | **-0.81%** |

Negative Component VaR indicates that long-duration Treasury exposure provides diversification benefits and reduces overall portfolio risk.

* * * * *

Historical Stress Testing
-------------------------

| Historical Event | Portfolio Return |
| --- | --- |
| COVID-19 Crash (2020) | **-19.65%** |
| China Slowdown (2015) | -10.85% |
| Rate Hike Cycle (2022) | -10.85% |
| European Debt Crisis (2011) | -8.85% |
| Q4 Equity Selloff (2018) | -8.28% |

* * * * *

> **Key Takeaways**
>
> -   Student-t provides a statistically superior fit than the Gaussian distribution (**lower AIC/BIC**).
> -   Filtered Historical Simulation (GARCH) delivers the most reliable VaR forecasts based on statistical backtesting.
> -   Treasury exposure (TLT) acts as a portfolio hedge by reducing overall market risk.
> -   The COVID-19 market crash represents the largest historical drawdown for the example portfolio.

* * * * *

📊 Visualizations
=================

The following figures are automatically generated by the engine as part of the risk analysis workflow.

![Breach Heatmap](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/output/breach_heatmap.png)

Breach Heatmap showing VaR violations across different confidence levels and time periods.

![Return Distribution](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/output/return_distribution_var.png)

Return Distribution with fitted parametric distributions (Gaussian and Student-t).

![Rolling VaR](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/output/rolling_var_backtest.png)

Rolling VaR backtest showing actual vs expected violations.

![Stress Test](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/output/stress_test_impact.png)

Stress Test Impact showing portfolio performance under historical stress events.

* * * * *

* * * * *

🧠 Methodology
==============

The **Portfolio Risk Analytics Engine** combines multiple complementary approaches to estimate market risk under different assumptions and market conditions.

Instead of relying on a single Value-at-Risk methodology, the engine compares **historical, parametric, simulation-based, and volatility-adjusted models**, validates them using statistical backtesting, and complements them with stress testing and portfolio risk attribution.

The following sections summarize the mathematical foundations behind each implemented model.

* * * * *

1\. Portfolio Return Calculation
--------------------------------

Individual asset returns are computed using daily log returns:

rt=ln⁡(PtPt-1)rt​=ln(Pt-1​Pt​​)

where

-   PtPt​ = asset price at time tt
-   rtrt​ = daily log return

Portfolio returns are obtained as the weighted sum of individual asset returns:

rp=wTrrp​=wTr

where

-   ww = portfolio weight vector
-   rr = vector of asset returns

* * * * *

2\. Historical Simulation VaR
-----------------------------

Historical Simulation estimates Value-at-Risk directly from observed historical returns without assuming any probability distribution.

The VaR at confidence level αα is the empirical lower-tail quantile:

VaRα=-Q1-α(rp)VaRα​=-Q1-α​(rp​)

### Advantages

-   No distributional assumptions
-   Naturally captures skewness and fat tails
-   Simple and intuitive

### Limitations

-   Assumes the future resembles the past
-   Does not adapt to changing market volatility

* * * * *

3\. Expected Shortfall (Conditional VaR)
----------------------------------------

Expected Shortfall measures the **average loss beyond the VaR threshold**, providing a more informative measure of tail risk.

ESα=-E[rp∣rp≤VaRα]ESα​=-E[rp​∣rp​≤VaRα​]

Unlike VaR, Expected Shortfall satisfies the properties of a coherent risk measure and captures the severity of extreme losses.

* * * * *

4\. Parametric (Variance-Covariance) VaR
----------------------------------------

### Gaussian VaR

The variance-covariance approach assumes portfolio returns follow a multivariate normal distribution.

Portfolio volatility is

σp=wTΣwσp​=wTΣw​

where

-   ΣΣ is the covariance matrix of asset returns.

The one-day VaR is

VaRα=-zασpVaRα​=-zα​σp​

where

-   zαzα​ is the standard normal quantile.

### Student-t VaR

Financial returns often exhibit **fat tails** that are poorly captured by the Gaussian assumption.

The engine therefore also models returns using a Student's t-distribution:

VaRα=-tα,ν⋅σν-2νVaRα​=-tα,ν​⋅σνν-2​​

where

-   νν = estimated degrees of freedom
-   tα,νtα,ν​ = Student-t quantile

The Student-t parameters are estimated using **Maximum Likelihood Estimation (MLE)**.

Compared to the Gaussian model, Student-t VaR better captures extreme market events.

* * * * *

5\. Monte Carlo Simulation
--------------------------

Monte Carlo VaR estimates portfolio risk by generating thousands of possible future return scenarios.

The engine supports

-   Multivariate Gaussian simulation
-   Multivariate Student-t simulation

The simulated portfolio return is

rp(i)=wTr(i)rp(i)​=wTr(i)

where

r(i)r(i)

is the simulated asset return vector.

The empirical lower-tail quantile of the simulated portfolio distribution provides the VaR estimate.

### Advantages

-   Flexible
-   Handles nonlinear portfolios
-   Easily extended to alternative distributions

* * * * *

6\. Filtered Historical Simulation (FHS)
----------------------------------------

Historical Simulation assumes constant volatility.

Filtered Historical Simulation improves this by incorporating a **GARCH(1,1)** volatility model.

The workflow is

1.  Fit a GARCH(1,1) model to portfolio returns.
2.  Estimate conditional volatility.
3.  Standardize historical returns.
4.  Bootstrap standardized residuals.
5.  Rescale using today's volatility.
6.  Compute Historical VaR on the filtered returns.

The conditional variance evolves according to

σt2=ω+αϵt-12+βσt-12σt2​=ω+αϵt-12​+βσt-12​

where

-   ωω = long-run variance
-   αα = ARCH parameter
-   ββ = GARCH parameter

FHS captures volatility clustering and typically produces more reliable VaR forecasts during turbulent market conditions.

* * * * *

📉 Statistical Backtesting
==========================

VaR models should not only estimate risk---they should also be validated.

The engine implements two industry-standard statistical backtests.

* * * * *

7\. Kupiec Proportion of Failures Test
--------------------------------------

Kupiec's test verifies whether the observed number of VaR breaches matches the expected breach probability.

The likelihood ratio statistic is

LRPOF=-2(ln⁡L0-ln⁡L1)LRPOF​=-2(lnL0​-lnL1​)

where

-   L0L0​ = likelihood under the expected breach rate
-   L1L1​ = likelihood under the observed breach rate

A high p-value indicates that the VaR model correctly estimates the expected frequency of losses.

* * * * *

8\. Christoffersen Conditional Coverage Test
--------------------------------------------

A correct VaR model should also produce **independent breaches**.

The Christoffersen test evaluates

-   Correct unconditional coverage
-   Independence of breaches

using a first-order Markov chain.

Passing both Kupiec and Christoffersen tests indicates a statistically reliable VaR model.

* * * * *

🌍 Stress Testing
=================

Value-at-Risk estimates losses under normal market conditions.

Stress testing evaluates portfolio performance under extreme market events.

The engine supports two complementary approaches.

### Hypothetical Scenarios

User-defined shocks representing events such as

-   Global Financial Crisis
-   COVID-19 Shock
-   Commodity Crash
-   Interest Rate Shock
-   Emerging Market Crisis

### Historical Scenario Replay

Historical ETF returns are replayed over major market crises including

-   COVID Crash (2020)
-   China Slowdown (2015)
-   European Debt Crisis (2011)
-   Q4 Selloff (2018)
-   Rate Hike Cycle (2022)

Unlike hypothetical stress tests, historical replay preserves actual cross-asset correlations observed during each event.

* * * * *

📊 Portfolio Risk Attribution
=============================

Beyond estimating total portfolio risk, the engine decomposes VaR into asset-level contributions.

Component VaR
-------------

Euler allocation decomposes portfolio VaR as

CVaRi=wi∂VaR∂wiCVaRi​=wi​∂wi​∂VaR​

The sum of Component VaRs equals total portfolio VaR.

* * * * *

Marginal VaR
------------

Marginal VaR measures the sensitivity of portfolio VaR to a small increase in an asset's portfolio weight.

MVaRi=∂VaR∂wiMVaRi​=∂wi​∂VaR​

This metric identifies which assets contribute most strongly to incremental portfolio risk.

* * * * *

Incremental VaR
---------------

Incremental VaR measures the change in portfolio VaR when a position is removed from the portfolio.

IVaRi=VaRportfolio-VaRwithoutiIVaRi​=VaRportfolio​-VaRwithout  i​

Negative Incremental VaR indicates that an asset acts as a **portfolio hedge**, reducing overall portfolio risk.

* * * * *

Summary
-------

By combining multiple complementary risk models, statistical validation techniques, stress testing, and portfolio risk attribution, the engine provides a practical end-to-end framework for market risk analysis inspired by methodologies used in modern financial institutions.

| Method | Distribution Assumption | Captures Fat Tails | Captures Volatility Clustering | Backtested |
| --- | --- | --- | --- | --- |
| Historical Simulation | None | ✅ | ❌ | ✅ |
| Parametric Gaussian | Normal | ❌ | ❌ | ❌ |
| Parametric Student-t | Student-t | ✅ | ❌ | ❌ |
| Monte Carlo Gaussian | Normal | ❌ | ❌ | ❌ |
| Monte Carlo Student-t | Student-t | ✅ | ❌ | ❌ |
| Filtered Historical Simulation | GARCH + Historical | ✅ | ✅ | ✅ |

* * * * *

🏗️ Architecture
================

The engine follows a modular pipeline where each component performs a single responsibility, making the project easy to extend with additional risk models or data sources.

![Project Architecture](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/output/architecture_diagram.png)

The modular design allows additional risk models, asset classes, or scenario generators to be integrated with minimal changes to the existing codebase.

* * * * *

📂 Project Structure
====================

```
Portfolio-Risk-Analytics-Engine/
│
├── output/
│   ├── architecture_diagram.png
│   ├── return_distribution_var.png
│   ├── rolling_var_backtest.png
│   ├── breach_heatmap.png
│   ├── stress_test_impact.png
│   └── summary_report.txt
│
├── src/
│   ├── backtest.py
│   ├── data.py
│   ├── stress_test.py
│   ├── var_models.py
│   ├── visualizations.py
│
├── main.py
├── requirements.txt
├── README.md
└── LICENSE

```

### Module Overview

| File | Purpose |
| --- | --- |
| `main.py` | Entry point for the complete market risk workflow. |
| `data_loader.py` | Downloads and preprocesses historical ETF price data. |
| `var_models.py` | Implements all VaR methodologies and portfolio risk attribution. |
| `backtest.py` | Kupiec and Christoffersen statistical backtesting. |
| `stress_test.py` | Hypothetical and historical scenario analysis. |
| `visualization.py` | Generates charts and graphical summaries. |
| `report_generator.py` | Produces the comprehensive risk report. |

* * * * *

⚙️ Installation
===============

Clone the repository
--------------------

```
git clone https://github.com/<your-username>/portfolio-risk-analytics-engine.git

cd portfolio-risk-analytics-engine

```

* * * * *

Create a virtual environment
----------------------------

```
python -m venv .venv

```

### Linux / macOS

```
source .venv/bin/activate

```

### Windows

```
.venv\Scripts\activate

```

* * * * *

Install dependencies
--------------------

```
pip install -r requirements.txt

```

* * * * *

Requirements
------------

-   Python **3.14+**

Required libraries

```
numpy
pandas
scipy
matplotlib
yfinance
arch

```

* * * * *

🚀 Running the Project
======================

Execute

```
python main.py

```

The engine will automatically

-   Download historical ETF prices from Yahoo Finance
-   Compute portfolio returns
-   Estimate multiple VaR models
-   Perform statistical backtesting
-   Run stress testing
-   Compute portfolio risk attribution
-   Generate visualizations
-   Export a comprehensive summary report

* * * * *

Customizing the Portfolio
-------------------------

The current implementation defines the portfolio directly inside `main.py`.

You can modify

-   Asset universe
-   Portfolio weights
-   Historical data period
-   Confidence levels
-   Portfolio value

by editing the configuration variables.

> **Future Update:** An interactive dashboard is planned, allowing users to configure portfolios through a graphical interface without modifying the source code.

* * * * *

📚 References
=============

The implementation is based on established quantitative finance literature and market risk methodologies.

### Books

-   Philippe Jorion --- *Value at Risk: The New Benchmark for Managing Financial Risk*
-   John C. Hull --- *Risk Management and Financial Institutions*
-   Kevin Dowd --- *Measuring Market Risk*
-   McNeil, Frey & Embrechts --- *Quantitative Risk Management*

### Research Papers

-   Kupiec, P. (1995). *Techniques for Verifying the Accuracy of Risk Measurement Models.*
-   Christoffersen, P. (1998). *Evaluating Interval Forecasts.*
-   Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999). *Filtered Historical Simulation.*
-   Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.*

### Data Source

Historical ETF price data is obtained from **Yahoo Finance** through the `yfinance` Python package.

* * * * *

🚀 Future Roadmap
=================

Planned enhancements include

-   Interactive Streamlit dashboard
-   Portfolio optimization (Mean-Variance / Black-Litterman)
-   EVT (Extreme Value Theory) VaR
-   Cornish-Fisher VaR
-   Multi-day VaR forecasting
-   Risk factor decomposition
-   Interactive scenario builder
-   Fixed Income and FX portfolio support
-   Docker deployment
-   REST API for portfolio analytics
-   CI/CD pipeline with automated testing
-   Unit and integration test suite

Contributions and suggestions are always welcome.

* * * * *

📄 License
==========

This project is licensed under the **MIT License**.

Feel free to use, modify, and distribute this project for educational and research purposes.

See the [LICENSE](https://file+.vscode-resource.vscode-cdn.net/home/raman/Projects/portfolio-risk-analytics-engine/LICENSE) file for details.

* * * * *

⭐ If you found this project interesting or useful, consider giving it a star!
