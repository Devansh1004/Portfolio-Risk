import numpy as np
import pandas as pd
from risk_metrics import annualized_volatility, historical_var, historical_cvar, max_drawdown, compute_all_metrics
from features import build_feature_frame, chronological_split
from data import compute_portfolio_returns

def sample_returns():
    np.random.seed(0)
    return pd.Series(np.random.normal(0, 0.01, 500), index=pd.date_range("2020-01-01", periods=500))

def test_annualized_volatility_positive():
    r = sample_returns()
    assert annualized_volatility(r) > 0

def test_var_less_than_cvar():
    r = sample_returns()
    var = historical_var(r)
    cvar = historical_cvar(r)
    assert cvar >= var

def test_max_drawdown_negative_or_zero():
    r = sample_returns()
    assert max_drawdown(r) <= 0

def test_compute_all_metrics_keys():
    r = sample_returns()
    metrics = compute_all_metrics(r)
    for key in ["volatility_annualized", "VaR_95", "CVaR_95", "max_drawdown"]:
        assert key in metrics

def test_feature_frame_no_lookahead():
    r = sample_returns()
    df, feature_cols = build_feature_frame(r)
    assert "target_vol" not in feature_cols
    assert len(df) > 0

def test_chronological_split_order():
    r = sample_returns()
    df, _ = build_feature_frame(r)
    train, val, test = chronological_split(df)
    assert train.index[-1] < val.index[0]
    assert val.index[-1] < test.index[0]

def test_portfolio_returns_weights_sum():
    idx = pd.date_range("2020-01-01", periods=10)
    asset_returns = pd.DataFrame({"A": np.random.normal(0, 0.01, 10), "B": np.random.normal(0, 0.01, 10)}, index=idx)
    weights = {"A": 1, "B": 1}
    port = compute_portfolio_returns(asset_returns, weights)
    expected = asset_returns["A"] * 0.5 + asset_returns["B"] * 0.5
    assert np.allclose(port.values, expected.values)
