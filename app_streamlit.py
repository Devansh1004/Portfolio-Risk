import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from features import build_feature_frame
from risk_pipeline import run_full_risk_analysis
from explainability import get_feature_importance, compute_shap_values
from report import generate_report

st.set_page_config(layout="wide")
st.title("Portfolio Risk Predictor")
bundle = joblib.load("artifacts/best_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]

tickers_input = st.text_input("Tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN,SPY")
weights_input = st.text_input("Weights (comma separated, same order)", "0.25,0.25,0.2,0.15,0.15")
lookback = st.slider("Lookback days", 200, 1500, 600)
confidence = st.slider("VaR confidence level", 0.90, 0.99, 0.95)

if st.button("Run Analysis"):
    tickers = [t.strip() for t in tickers_input.split(",")]
    weights_list = [float(w.strip()) for w in weights_input.split(",")]
    weights = dict(zip(tickers, weights_list))
    end = datetime.today()
    start = end - timedelta(days=lookback)
    result = run_full_risk_analysis(tickers, weights, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), confidence)
    st.session_state["result"] = result
    df, _ = build_feature_frame(result["portfolio_returns"])
    latest = df.iloc[[-1]][feature_cols]
    X_latest = scaler.transform(latest)
    st.session_state["pred"] = float(model.predict(X_latest)[0])
    st.session_state["X_latest"] = X_latest
    st.session_state["df"] = df

if "result" in st.session_state:
    result = st.session_state["result"]
    risk = result["risk_metrics"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "VaR Comparison", "Correlation", "Stress Testing", "Risk Decomposition", "Backtesting"])

    with tab1:
        st.subheader("Predicted Next-Period Annualized Volatility")
        st.write(st.session_state["pred"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Volatility (ann.)", f"{risk['volatility_annualized']:.4f}")
        col1.metric("VaR 95%", f"{risk['VaR_95']:.4f}")
        col2.metric("CVaR 95%", f"{risk['CVaR_95']:.4f}")
        col2.metric("Max Drawdown", f"{risk['max_drawdown']:.4f}")
        col3.metric("Sharpe Ratio", f"{risk['sharpe_ratio']:.3f}")
        col3.metric("Sortino Ratio", f"{risk['sortino_ratio']:.3f}")
        st.metric("Calmar Ratio", f"{risk['calmar_ratio']:.3f}")
        st.subheader("Portfolio Cumulative Returns")
        st.line_chart((1 + result["portfolio_returns"]).cumprod())
        st.subheader("Rolling 21-Day Volatility")
        st.line_chart(result["portfolio_returns"].rolling(21).std())
        st.subheader("Feature Importance")
        importance = get_feature_importance(model, feature_cols)
        if len(importance) > 0:
            st.bar_chart(importance.head(10))
        st.subheader("SHAP Importance")
        X_train_bg = scaler.transform(st.session_state["df"][feature_cols].sample(min(100, len(st.session_state["df"])), random_state=42))
        shap_importance = compute_shap_values(model, X_train_bg, st.session_state["X_latest"], feature_cols)
        st.bar_chart(shap_importance.head(10))

    with tab2:
        st.subheader("Historical vs Parametric vs Monte Carlo VaR")
        var_df = pd.DataFrame(result["var_results"]).T
        st.dataframe(var_df)
        st.bar_chart(var_df)

    with tab3:
        st.subheader("Asset Correlation Matrix")
        st.dataframe(result["corr_matrix"].style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1))
        st.subheader("Annualized Covariance Matrix")
        st.dataframe(result["cov_matrix"])

    with tab4:
        st.subheader("Stress Test Scenarios")
        for scenario in result["stress_results"]:
            st.write(scenario)

    with tab5:
        st.subheader("Risk Contribution by Asset")
        st.dataframe(result["decomposition_df"])
        st.bar_chart(result["decomposition_df"]["percent_of_risk"])

    with tab6:
        st.subheader("VaR Backtest (Kupiec Test)")
        st.write(result["backtest"])
        if result["backtest"]["reject_null"] is True:
            st.warning("Model rejects the null: breach rate significantly differs from expected.")
        elif result["backtest"]["reject_null"] is False:
            st.success("Model fails to reject the null: VaR breach rate is statistically consistent.")
        st.subheader("Rolling VaR")
        st.line_chart(result["var_series"])
        st.subheader("Exception Analysis")
        st.write(result["exceptions"])

    st.divider()
    if st.button("Generate PDF Report"):
        os.makedirs("artifacts/reports", exist_ok=True)
        path = "artifacts/reports/risk_report.pdf"
        generate_report(path, result["portfolio_returns"], result["asset_returns"], result["weights"], risk, result["var_results"], result["stress_results"], result["decomposition_df"], result["backtest"], result["exceptions"], result["corr_matrix"])
        with open(path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="portfolio_risk_report.pdf", mime="application/pdf")
