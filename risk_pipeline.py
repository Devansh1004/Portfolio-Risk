import pandas as pd
from data import fetch_prices, compute_asset_returns, compute_portfolio_returns
from covariance import compute_covariance_matrix, compute_correlation_matrix
from var_engine import var_comparison
from stress_engine import run_stress_tests
from risk_decomposition import decompose_risk
from risk_metrics import compute_all_metrics, rolling_var_series, kupiec_test, exception_analysis

def run_full_risk_analysis(tickers, weights, start, end, confidence=0.95, var_window=100):
    prices = fetch_prices(tickers, start, end)
    asset_returns = compute_asset_returns(prices)
    weights_series = pd.Series(weights)
    weights_series = weights_series / weights_series.sum()
    portfolio_returns = compute_portfolio_returns(asset_returns, weights_series)
    cov_matrix = compute_covariance_matrix(asset_returns)
    corr_matrix = compute_correlation_matrix(asset_returns)
    risk_metrics = compute_all_metrics(portfolio_returns, confidence)
    var_results = var_comparison(portfolio_returns, asset_returns, weights_series, confidence)
    stress_results = run_stress_tests(asset_returns, weights_series)
    decomposition_df = decompose_risk(weights_series, cov_matrix)
    var_series = rolling_var_series(portfolio_returns, window=var_window, confidence=confidence)
    backtest = kupiec_test(portfolio_returns, var_series, confidence)
    exceptions = exception_analysis(portfolio_returns, var_series)
    return {
        "asset_returns": asset_returns,
        "portfolio_returns": portfolio_returns,
        "weights": weights_series,
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
        "risk_metrics": risk_metrics,
        "var_results": var_results,
        "stress_results": stress_results,
        "decomposition_df": decomposition_df,
        "var_series": var_series,
        "backtest": backtest,
        "exceptions": exceptions
    }
