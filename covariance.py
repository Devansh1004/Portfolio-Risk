import numpy as np
import pandas as pd

def compute_covariance_matrix(asset_returns, periods_per_year=252):
    return asset_returns.cov() * periods_per_year

def compute_correlation_matrix(asset_returns):
    return asset_returns.corr()

def portfolio_volatility(weights, cov_matrix):
    w = weights.reindex(cov_matrix.columns).values
    return float(np.sqrt(w @ cov_matrix.values @ w))

def portfolio_variance_contribution(weights, cov_matrix):
    w = weights.reindex(cov_matrix.columns)
    port_var = w.values @ cov_matrix.values @ w.values
    return port_var
