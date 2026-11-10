import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def historical_baseline(df, roll_col="roll_vol_21", periods_per_year=252):
    return df[roll_col] * np.sqrt(periods_per_year)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {"MAE": mae, "RMSE": rmse}

def get_model_zoo():
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    }

def train_and_select(X_train, y_train, X_val, y_val, feature_cols):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train[feature_cols])
    X_val_s = scaler.transform(X_val[feature_cols])
    results = {}
    fitted = {}
    for name, model in get_model_zoo().items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_val_s)
        results[name] = evaluate(y_val, preds)
        fitted[name] = model
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    return fitted[best_name], best_name, results, scaler

class EnsembleAverage:
    def __init__(self, models):
        self.models = models
    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)
