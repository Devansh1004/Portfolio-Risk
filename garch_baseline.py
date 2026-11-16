import numpy as np
import pandas as pd
from arch import arch_model

def rolling_garch_forecast(returns, train_end_idx, periods_per_year=252, refit_every=5):
    scaled = returns * 100
    forecasts = []
    dates = []
    res = None
    for step, i in enumerate(range(train_end_idx, len(returns))):
        window = scaled.iloc[:i]
        if res is None or step % refit_every == 0:
            model = arch_model(window, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
            res = model.fit(disp="off")
        f = res.forecast(horizon=1, reindex=False)
        variance = f.variance.values[-1, 0]
        daily_vol = np.sqrt(variance) / 100
        forecasts.append(daily_vol * np.sqrt(periods_per_year))
        dates.append(returns.index[i])
    return pd.Series(forecasts, index=dates)
