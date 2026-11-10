import numpy as np
import pandas as pd

def decompose_risk(weights, cov_matrix):
    w = weights.reindex(cov_matrix.columns).values
    cov = cov_matrix.values
    port_vol = float(np.sqrt(w @ cov @ w))
    marginal_contrib = (cov @ w) / port_vol
    component_contrib = w * marginal_contrib
    percent_contrib = component_contrib / port_vol
    return pd.DataFrame({
        "weight": w,
        "marginal_contribution": marginal_contrib,
        "component_contribution": component_contrib,
        "percent_of_risk": percent_contrib
    }, index=cov_matrix.columns).sort_values("percent_of_risk", ascending=False)
