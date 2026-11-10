import numpy as np
import pandas as pd
from scipy.stats import chi2

def annualized_volatility(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def historical_var(returns, confidence=0.95):
    return -np.percentile(returns, (1 - confidence) * 100)

def historical_cvar(returns, confidence=0.95):
    var = historical_var(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return -tail.mean()

def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)

def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)

def calmar_ratio(returns, periods_per_year=252):
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    annual_return = returns.mean() * periods_per_year
    return annual_return / abs(mdd)

def rolling_realized_volatility(returns, window):
    return returns.rolling(window).std()

def rolling_var_series(returns, window=252, confidence=0.95):
    return returns.rolling(window).apply(lambda x: historical_var(x, confidence), raw=False)

def exception_analysis(returns, var_series):
    aligned = pd.concat([returns, var_series], axis=1).dropna()
    aligned.columns = ["ret", "var"]
    breach_mask = aligned["ret"] < -aligned["var"]
    breach_dates = aligned.index[breach_mask].strftime("%Y-%m-%d").tolist()
    breach_flags = breach_mask.astype(int).values
    clusters = 0
    for i in range(1, len(breach_flags)):
        if breach_flags[i] == 1 and breach_flags[i - 1] == 1:
            clusters += 1
    return {"breach_dates": breach_dates, "total_breaches": int(breach_mask.sum()), "consecutive_breach_pairs": int(clusters)}

def kupiec_test(returns, var_series, confidence=0.95):
    aligned = pd.concat([returns, var_series], axis=1).dropna()
    aligned.columns = ["ret", "var"]
    breaches = (aligned["ret"] < -aligned["var"]).sum()
    n = len(aligned)
    if n == 0:
        return {"breaches": 0, "n_obs": 0, "breach_rate": None, "expected_rate": 1 - confidence, "lr_stat": None, "p_value": None, "reject_null": None}
    p = 1 - confidence
    p_hat = breaches / n
    if p_hat in (0, 1):
        lr_stat = np.nan
        p_value = np.nan
    else:
        num = ((1 - p) ** (n - breaches)) * (p ** breaches)
        den = ((1 - p_hat) ** (n - breaches)) * (p_hat ** breaches)
        lr_stat = -2 * np.log(num / den)
        p_value = 1 - chi2.cdf(lr_stat, df=1)
    reject_null = bool(p_value is not None and not np.isnan(p_value) and p_value < 0.05)
    return {"breaches": int(breaches), "n_obs": int(n), "breach_rate": float(p_hat), "expected_rate": float(p), "lr_stat": float(lr_stat) if not np.isnan(lr_stat) else None, "p_value": float(p_value) if not np.isnan(p_value) else None, "reject_null": reject_null}

def compute_all_metrics(returns, confidence=0.95, periods_per_year=252, risk_free_rate=0.0):
    return {
        "volatility_annualized": annualized_volatility(returns, periods_per_year),
        "VaR_95": historical_var(returns, confidence),
        "CVaR_95": historical_cvar(returns, confidence),
        "max_drawdown": max_drawdown(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": calmar_ratio(returns, periods_per_year)
    }
