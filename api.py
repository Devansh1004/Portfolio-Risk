import os
import uuid
import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from data import fetch_prices, compute_asset_returns, compute_portfolio_returns
from features import build_feature_frame
from risk_pipeline import run_full_risk_analysis
from report import generate_report

app = FastAPI(title="Portfolio Risk Predictor")
bundle = joblib.load("artifacts/best_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]

class PortfolioRequest(BaseModel):
    tickers: list
    weights: dict
    lookback_days: int = 400
    confidence: float = 0.95

def date_range(lookback_days):
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_volatility")
def predict_volatility(req: PortfolioRequest):
    start, end = date_range(req.lookback_days)
    prices = fetch_prices(req.tickers, start, end)
    asset_returns = compute_asset_returns(prices)
    portfolio_returns = compute_portfolio_returns(asset_returns, req.weights)
    df, _ = build_feature_frame(portfolio_returns)
    latest = df.iloc[[-1]][feature_cols]
    X = scaler.transform(latest)
    pred = float(model.predict(X)[0])
    return {"predicted_next_period_volatility": pred}

@app.post("/risk_metrics")
def risk_metrics(req: PortfolioRequest):
    start, end = date_range(req.lookback_days)
    result = run_full_risk_analysis(req.tickers, req.weights, start, end, req.confidence)
    return {k: float(v) for k, v in result["risk_metrics"].items()}

@app.post("/var")
def var_endpoint(req: PortfolioRequest):
    start, end = date_range(req.lookback_days)
    result = run_full_risk_analysis(req.tickers, req.weights, start, end, req.confidence)
    return result["var_results"]

@app.post("/covariance")
def covariance_endpoint(req: PortfolioRequest):
    start, end = date_range(req.lookback_days)
    result = run_full_risk_analysis(req.tickers, req.weights, start, end, req.confidence)
    return {"covariance": result["cov_matrix"].to_dict(), "correlation": result["corr_matrix"].to_dict()}

@app.post("/stress_test")
def stress_test_endpoint(req: PortfolioRequest):
    start, end = date_range(req.lookback_days)
    result = run_full_risk_analysis(req.tickers, req.weights, start, end, req.confidence)
    return {"scenarios": result["stress_results"]}

@app.post("/risk_decomposition")
def risk_decomposition_endpoint(req: PortfolioRequest):
    start, end = date_range(req.lookback_days)
    result = run_full_risk_analysis(req.tickers, req.weights, start, end, req.confidence)
    return result["decomposition_df"].to_dict(orient="index")

@app.post("/backtest")
def backtest_endpoint(req: PortfolioRequest):
    start, end = date_range(req.lookback_days)
    result = run_full_risk_analysis(req.tickers, req.weights, start, end, req.confidence)
    return {"kupiec_test": result["backtest"], "exceptions": result["exceptions"]}

@app.post("/report")
def report_endpoint(req: PortfolioRequest):
    start, end = date_range(req.lookback_days)
    result = run_full_risk_analysis(req.tickers, req.weights, start, end, req.confidence)
    os.makedirs("artifacts/reports", exist_ok=True)
    path = f"artifacts/reports/risk_report_{uuid.uuid4().hex[:8]}.pdf"
    generate_report(path, result["portfolio_returns"], result["asset_returns"], result["weights"], result["risk_metrics"], result["var_results"], result["stress_results"], result["decomposition_df"], result["backtest"], result["exceptions"], result["corr_matrix"])
    return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))
