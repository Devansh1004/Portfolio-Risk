import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
from data import fetch_prices, compute_asset_returns, compute_portfolio_returns
from features import build_feature_frame
from risk_metrics import compute_all_metrics

app = FastAPI(title="Portfolio Risk Predictor")
bundle = joblib.load("models/best_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]

class PortfolioRequest(BaseModel):
    tickers: list
    weights: dict
    lookback_days: int = 400

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_volatility")
def predict_volatility(req: PortfolioRequest):
    end = datetime.today()
    start = end - timedelta(days=req.lookback_days)
    prices = fetch_prices(req.tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    asset_returns = compute_asset_returns(prices)
    portfolio_returns = compute_portfolio_returns(asset_returns, req.weights)
    df, _ = build_feature_frame(portfolio_returns)
    latest = df.iloc[[-1]][feature_cols]
    X = scaler.transform(latest)
    pred = float(model.predict(X)[0])
    risk = compute_all_metrics(portfolio_returns)
    return {
        "predicted_next_period_volatility": pred,
        "current_risk_metrics": {k: float(v) for k, v in risk.items()}
    }

@app.post("/risk_metrics")
def risk_metrics(req: PortfolioRequest):
    end = datetime.today()
    start = end - timedelta(days=req.lookback_days)
    prices = fetch_prices(req.tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    asset_returns = compute_asset_returns(prices)
    portfolio_returns = compute_portfolio_returns(asset_returns, req.weights)
    risk = compute_all_metrics(portfolio_returns)
    return {k: float(v) for k, v in risk.items()}
