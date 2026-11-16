import numpy as np
import pandas as pd

def build_feature_frame(returns, lags=(1, 2, 3, 5), roll_windows=(5, 10, 21), target_window=5, periods_per_year=252):
    df = pd.DataFrame(index=returns.index)
    df["ret"] = returns
    for lag in lags:
        df[f"ret_lag_{lag}"] = returns.shift(lag)
    for w in roll_windows:
        df[f"roll_vol_{w}"] = returns.rolling(w).std()
        df[f"roll_mean_{w}"] = returns.rolling(w).mean()
        df[f"roll_min_{w}"] = returns.rolling(w).min()
        df[f"roll_max_{w}"] = returns.rolling(w).max()
    df["abs_ret"] = returns.abs()
    df["target_vol"] = returns.shift(-1).rolling(target_window).std().shift(-(target_window - 1)) * np.sqrt(periods_per_year)
    df = df.dropna()
    feature_cols = [c for c in df.columns if c not in ("target_vol",)]
    return df, feature_cols

def chronological_split(df, train_frac=0.6, val_frac=0.2):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test
