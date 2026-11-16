import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from data import fetch_prices, compute_asset_returns, compute_portfolio_returns
from features import build_feature_frame
from risk_metrics import compute_all_metrics, rolling_var_series, kupiec_test
from explainability import get_feature_importance, compute_shap_values

st.title("Portfolio Risk Predictor")
bundle = joblib.load("models/best_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]

tickers_input = st.text_input("Tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN,SPY")
weights_input = st.text_input("Weights (comma separated, same order)", "0.25,0.25,0.2,0.15,0.15")
lookback = st.slider("Lookback days", 100, 1000, 400)

if st.button("Run Analysis"):
    tickers = [t.strip() for t in tickers_input.split(",")]
    weights_list = [float(w.strip()) for w in weights_input.split(",")]
    weights = dict(zip(tickers, weights_list))
    end = datetime.today()
    start = end - timedelta(days=lookback)
    prices = fetch_prices(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    asset_returns = compute_asset_returns(prices)
    portfolio_returns = compute_portfolio_returns(asset_returns, weights)
    df, _ = build_feature_frame(portfolio_returns)
    latest = df.iloc[[-1]][feature_cols]
    X_latest = scaler.transform(latest)
    pred = float(model.predict(X_latest)[0])
    risk = compute_all_metrics(portfolio_returns)
    var_series = rolling_var_series(portfolio_returns)
    backtest = kupiec_test(portfolio_returns, var_series)
    st.subheader("Predicted Next-Period Annualized Volatility")
    st.write(pred)
    st.subheader("Risk Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Volatility (ann.)", f"{risk['volatility_annualized']:.4f}")
    col1.metric("VaR 95%", f"{risk['VaR_95']:.4f}")
    col2.metric("CVaR 95%", f"{risk['CVaR_95']:.4f}")
    col2.metric("Max Drawdown", f"{risk['max_drawdown']:.4f}")
    col3.metric("Sharpe Ratio", f"{risk['sharpe_ratio']:.3f}")
    col3.metric("Sortino Ratio", f"{risk['sortino_ratio']:.3f}")
    st.metric("Calmar Ratio", f"{risk['calmar_ratio']:.3f}")
    st.subheader("VaR Backtest (Kupiec Test, 95%)")
    st.write(backtest)
    if backtest["reject_null"] is True:
        st.warning("Model rejects the null: observed breach rate significantly differs from expected 5%.")
    elif backtest["reject_null"] is False:
        st.success("Model fails to reject the null: VaR breach rate is statistically consistent with 95% confidence.")
    st.subheader("Portfolio Cumulative Returns")
    st.line_chart((1 + portfolio_returns).cumprod())
    st.subheader("Rolling 21-Day Volatility")
    st.line_chart(portfolio_returns.rolling(21).std())
    st.subheader("Rolling VaR 95%")
    st.line_chart(var_series)
    st.subheader("Feature Importance")
    importance = get_feature_importance(model, feature_cols)
    if len(importance) > 0:
        st.bar_chart(importance.head(10))
    else:
        st.info("Selected model does not expose native feature importance.")
    st.subheader("SHAP Importance (Mean |SHAP Value|)")
    X_train_bg = scaler.transform(df[feature_cols].sample(min(100, len(df)), random_state=42))
    shap_importance = compute_shap_values(model, X_train_bg, X_latest, feature_cols)
    st.bar_chart(shap_importance.head(10))

if os.path.exists("plots/shap_importance.png"):
    st.subheader("Saved SHAP Importance (from last training run)")
    st.image("plots/shap_importance.png")
