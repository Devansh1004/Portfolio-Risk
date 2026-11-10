import time
import pandas as pd
import yfinance as yf

def fetch_prices(tickers, start, end, retries=3, pause=2):
    series_by_ticker = {}
    for ticker in tickers:
        for attempt in range(retries):
            try:
                hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
                if not hist.empty:
                    series_by_ticker[ticker] = hist["Close"]
                    break
            except Exception:
                pass
            time.sleep(pause)
        else:
            raise RuntimeError(f"Could not fetch data for {ticker} after {retries} attempts")
    data = pd.DataFrame(series_by_ticker)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data.dropna()

def compute_asset_returns(prices):
    return prices.pct_change().dropna()

def compute_portfolio_returns(asset_returns, weights):
    w = pd.Series(weights)
    w = w / w.sum()
    return asset_returns[w.index].dot(w)
