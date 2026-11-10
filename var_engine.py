import numpy as np
import pandas as pd
from scipy.stats import norm
from risk_metrics import historical_var, historical_cvar

def parametric_var(returns, confidence=0.95, horizon=1):
    mu = returns.mean() * horizon
    sigma = returns.std() * np.sqrt(horizon)
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))

def parametric_es(returns, confidence=0.95, horizon=1):
    mu = returns.mean() * horizon
    sigma = returns.std() * np.sqrt(horizon)
    z = norm.ppf(1 - confidence)
    es_factor = norm.pdf(z) / (1 - confidence)
    return float(-(mu - sigma * es_factor))

def monte_carlo_var(asset_returns, weights, confidence=0.95, horizon=1, n_sims=10000, seed=42):
    rng = np.random.default_rng(seed)
    mean_vec = asset_returns.mean().values * horizon
    cov_matrix = asset_returns.cov().values * horizon
    w = weights.reindex(asset_returns.columns).values
    sims = rng.multivariate_normal(mean_vec, cov_matrix, size=n_sims)
    port_sim_returns = sims @ w
    var = -np.percentile(port_sim_returns, (1 - confidence) * 100)
    tail = port_sim_returns[port_sim_returns <= -var]
    es = -tail.mean() if len(tail) > 0 else var
    return float(var), float(es), port_sim_returns

def var_comparison(returns, asset_returns, weights, confidence=0.95, horizon=1, n_sims=10000):
    hist_var = historical_var(returns, confidence)
    hist_es = historical_cvar(returns, confidence)
    param_var = parametric_var(returns, confidence, horizon)
    param_es = parametric_es(returns, confidence, horizon)
    mc_var, mc_es, _ = monte_carlo_var(asset_returns, weights, confidence, horizon, n_sims)
    return {
        "historical": {"VaR": float(hist_var), "ES": float(hist_es)},
        "parametric": {"VaR": param_var, "ES": param_es},
        "monte_carlo": {"VaR": mc_var, "ES": mc_es}
    }
