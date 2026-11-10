import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import fetch_prices, compute_asset_returns, compute_portfolio_returns
from features import build_feature_frame, chronological_split
from models import historical_baseline, evaluate, train_and_select, EnsembleAverage
from risk_metrics import compute_all_metrics, rolling_var_series, kupiec_test
from garch_baseline import rolling_garch_forecast
from explainability import get_feature_importance, compute_shap_values
from covariance import compute_covariance_matrix, compute_correlation_matrix
from var_engine import var_comparison
from stress_engine import run_stress_tests
from risk_decomposition import decompose_risk
from risk_metrics import exception_analysis
from report import generate_report

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
WEIGHTS = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.2, "AMZN": 0.15, "SPY": 0.15}
START = "2015-01-01"
END = "2025-01-01"

def main():
    import os
    os.makedirs("artifacts/reports", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    prices = fetch_prices(TICKERS, START, END)
    asset_returns = compute_asset_returns(prices)
    portfolio_returns = compute_portfolio_returns(asset_returns, WEIGHTS)
    df, feature_cols = build_feature_frame(portfolio_returns)
    train, val, test = chronological_split(df)
    y_train, y_val, y_test = train["target_vol"], val["target_vol"], test["target_vol"]
    best_model, best_name, val_results, scaler = train_and_select(train, y_train, val, y_val, feature_cols)
    X_test_s = scaler.transform(test[feature_cols])
    ml_preds = best_model.predict(X_test_s)
    ml_metrics = evaluate(y_test, ml_preds)
    baseline_preds = historical_baseline(test)
    baseline_metrics = evaluate(y_test, baseline_preds)
    train_end_idx = len(train) + len(val)
    garch_preds_full = rolling_garch_forecast(portfolio_returns.loc[df.index], train_end_idx)
    garch_preds = garch_preds_full.reindex(test.index).dropna()
    garch_y = y_test.reindex(garch_preds.index)
    garch_metrics = evaluate(garch_y, garch_preds)
    port_risk = compute_all_metrics(portfolio_returns)
    var_series = rolling_var_series(portfolio_returns)
    backtest = kupiec_test(portfolio_returns, var_series)
    importance = get_feature_importance(best_model, feature_cols)
    X_train_bg_s = scaler.transform(train[feature_cols].sample(min(100, len(train)), random_state=42))
    shap_importance = compute_shap_values(best_model, X_train_bg_s, X_test_s, feature_cols)
    joblib.dump({"model": best_model, "scaler": scaler, "feature_cols": feature_cols, "name": best_name}, "artifacts/best_model.joblib")
    report = {
        "best_model": best_name,
        "validation_results": val_results,
        "test_ml_metrics": ml_metrics,
        "test_baseline_metrics": baseline_metrics,
        "test_garch_metrics": garch_metrics,
        "portfolio_risk_metrics": {k: float(v) for k, v in port_risk.items()},
        "var_backtest_kupiec": backtest,
        "feature_importance": importance.to_dict(),
        "shap_importance": shap_importance.to_dict()
    }
    with open("artifacts/report.json", "w") as f:
        json.dump(report, f, indent=2)
    plt.figure(figsize=(10, 5))
    plt.plot(test.index, y_test.values, label="Actual Volatility")
    plt.plot(test.index, ml_preds, label=f"ML Prediction ({best_name})")
    plt.plot(test.index, baseline_preds.values, label="Historical Baseline")
    plt.plot(garch_preds.index, garch_preds.values, label="GARCH(1,1) Baseline")
    plt.legend()
    plt.title("Predicted vs Actual Portfolio Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.tight_layout()
    plt.savefig("plots/volatility_prediction.png")
    plt.close()
    importance.to_csv("artifacts/feature_importance.csv", header=["importance"])
    shap_importance.to_csv("artifacts/shap_importance.csv", header=["mean_abs_shap"])
    plt.figure(figsize=(8, 5))
    shap_importance.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Features by Mean |SHAP Value|")
    plt.xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig("plots/shap_importance.png")
    plt.close()
    weights_series = pd.Series(WEIGHTS)
    weights_series = weights_series / weights_series.sum()
    cov_matrix = compute_covariance_matrix(asset_returns)
    corr_matrix = compute_correlation_matrix(asset_returns)
    var_results = var_comparison(portfolio_returns, asset_returns, weights_series)
    stress_results = run_stress_tests(asset_returns, weights_series)
    decomposition_df = decompose_risk(weights_series, cov_matrix)
    exceptions = exception_analysis(portfolio_returns, var_series)
    generate_report("artifacts/reports/risk_report.pdf", portfolio_returns, asset_returns, weights_series, port_risk, var_results, stress_results, decomposition_df, backtest, exceptions, corr_matrix)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
