import numpy as np
import pandas as pd
import shap

def get_feature_importance(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)
    else:
        return pd.Series(dtype=float)
    return pd.Series(values, index=feature_cols).sort_values(ascending=False)

def compute_shap_values(model, X_background, X_explain, feature_cols):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
    except Exception:
        explainer = shap.Explainer(model.predict, X_background)
        shap_values = explainer(X_explain).values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)
