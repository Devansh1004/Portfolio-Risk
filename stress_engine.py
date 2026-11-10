import numpy as np
import pandas as pd
from covariance import portfolio_volatility

def apply_market_crash(asset_returns, weights, shock_pct=-0.20):
    w = weights.reindex(asset_returns.columns)
    loss = float((w.values * shock_pct).sum())
    return {"scenario": f"Market crash ({shock_pct:.0%} shock to all assets)", "portfolio_loss": loss}

def apply_asymmetric_shock(asset_returns, weights, shocks):
    w = weights.reindex(asset_returns.columns).fillna(0)
    shock_series = pd.Series(shocks).reindex(asset_returns.columns).fillna(0)
    loss = float((w.values * shock_series.values).sum())
    return {"scenario": "Custom asset-level shock", "shocks": shocks, "portfolio_loss": loss}

def apply_volatility_shock(asset_returns, weights, multiplier=2.0, periods_per_year=252):
    cov = asset_returns.cov() * periods_per_year
    shocked_cov = cov * (multiplier ** 2)
    base_vol = portfolio_volatility(weights, cov)
    shocked_vol = portfolio_volatility(weights, shocked_cov)
    return {"scenario": f"Volatility shock ({multiplier}x)", "base_volatility": base_vol, "shocked_volatility": shocked_vol}

def apply_correlation_shock(asset_returns, weights, target_corr=0.9, periods_per_year=252):
    std = asset_returns.std().values
    n = len(std)
    shocked_corr = np.full((n, n), target_corr)
    np.fill_diagonal(shocked_corr, 1.0)
    shocked_cov = pd.DataFrame(shocked_corr * np.outer(std, std) * periods_per_year, index=asset_returns.columns, columns=asset_returns.columns)
    base_cov = asset_returns.cov() * periods_per_year
    base_vol = portfolio_volatility(weights, base_cov)
    shocked_vol = portfolio_volatility(weights, shocked_cov)
    return {"scenario": f"Correlation shock (all pairs to {target_corr})", "base_volatility": base_vol, "shocked_volatility": shocked_vol}

def run_stress_tests(asset_returns, weights):
    results = []
    results.append(apply_market_crash(asset_returns, weights, -0.20))
    results.append(apply_market_crash(asset_returns, weights, -0.10))
    results.append(apply_volatility_shock(asset_returns, weights, 2.0))
    results.append(apply_correlation_shock(asset_returns, weights, 0.9))
    return results
