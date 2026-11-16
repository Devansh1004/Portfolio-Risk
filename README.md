# Portfolio Risk Predictor

Predicts next-period (5-day) annualized portfolio volatility with ML, benchmarked against a
21-day historical-volatility baseline and a rolling GARCH(1,1) baseline. Also computes standard
portfolio risk metrics (VaR, CVaR, drawdown, Sharpe/Sortino/Calmar), backtests VaR with a Kupiec
test, explains the model with SHAP, and serves everything through a FastAPI service and a
Streamlit dashboard.

## Results

Example run on AAPL / MSFT / GOOGL / AMZN / SPY (25/25/20/15/15), 2015-2025, chronological
60/20/20 split. Test period: 2022-12-30 to 2024-12-23.

| Method (test set) | MAE | RMSE |
|---|---|---|
| Historical baseline (21-day vol) | **0.0603** | **0.0763** |
| GARCH(1,1) baseline | 0.0620 | 0.0772 |
| ML model (Ridge, selected on validation) | 0.0621 | 0.0819 |

**Honest takeaway:** the ML model did *not* beat the simple historical-volatility baseline on this
test window. Volatility is very persistent and all features come from the same return series.
The portfolio's historical 95% VaR passed the Kupiec backtest (5.48% breach rate vs 5% expected,
p = 0.30). See [RESULTS.md](RESULTS.md) for the full tables, SHAP analysis, and limitations.

![Predicted vs actual volatility](plots/volatility_prediction.png)

## Structure

```
data.py              fetch prices (yfinance), asset and portfolio returns
features.py          lagged/rolling features, forward 5-day volatility target, chronological split
models.py            Ridge / RandomForest / GradientBoosting, selection by validation RMSE, ensemble helper
garch_baseline.py    rolling-refit GARCH(1,1) one-step-ahead volatility forecast
risk_metrics.py      volatility, VaR, CVaR, max drawdown, Sharpe/Sortino/Calmar, Kupiec test
explainability.py    native feature importance + SHAP
train.py             end-to-end pipeline (fetch -> features -> train -> evaluate -> save)
api.py               FastAPI service serving the saved model
app_streamlit.py     Streamlit dashboard
notebook.ipynb       exploratory walkthrough
tests/test_basic.py  unit tests for metrics, features, splits
models/              saved model, report.json, feature/SHAP importance CSVs
plots/               generated figures
RESULTS.md           documented results of the latest training run
```

## Setup

Requires Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Fetches data from Yahoo Finance (internet required), then writes:

- `models/best_model.joblib` (model, scaler, feature list)
- `models/report.json` (all metrics)
- `models/feature_importance.csv`, `models/shap_importance.csv`
- `plots/volatility_prediction.png`, `plots/shap_importance.png`

The GARCH rolling refit and SHAP computation make a full run take a few minutes.

## Run the API

Run from the project root (the model is loaded from `models/best_model.joblib`).

```bash
uvicorn api:app --reload
```

- `GET /health`
- `POST /predict_volatility`
- `POST /risk_metrics`

Example body:

```json
{"tickers": ["AAPL","MSFT","GOOGL","AMZN","SPY"], "weights": {"AAPL":0.25,"MSFT":0.25,"GOOGL":0.2,"AMZN":0.15,"SPY":0.15}, "lookback_days": 400}
```

Interactive docs at http://127.0.0.1:8000/docs.

## Run the dashboard

```bash
streamlit run app_streamlit.py
```

## Tests

```bash
pytest tests/
```

## Method notes

- **Target:** realized annualized volatility over the next 5 trading days.
- **Features:** current and lagged returns, rolling std / mean / min / max over 5, 10 and 21 days, absolute return (18 total). The target is built from strictly future returns, and the split is strictly chronological (60/20/20) to avoid look-ahead bias.
- **Model selection:** Ridge, RandomForest and GradientBoosting are compared on validation RMSE; the best is evaluated once on the test set.
- **Baselines:** 21-day rolling historical volatility (annualized), and a rolling-refit GARCH(1,1) one-step-ahead forecast (refit every 5 steps for speed).
- **VaR backtest:** rolling 252-day historical VaR at 95%, tested with the Kupiec proportion-of-failures test.

## Limitations

- Results come from one split and one run, without confidence intervals.
- The rolling VaR window includes the current day's return, so the backtest is slightly optimistic (see RESULTS.md).
- Not financial advice; this is an educational / research project.
