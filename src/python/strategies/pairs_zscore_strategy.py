import pandas as pd
import numpy as np

def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def build_spread(prices_A: pd.Series, prices_B: pd.Series) -> pd.Series:
    log_A = np.log(prices_A)
    log_B = np.log(prices_B)
    spread = log_A - log_B
    spread.name = f"spread_{prices_A.name}_{prices_B.name}"
    return spread

def generate_pairs_signals(spread: pd.Series,
                           window: int = 20,
                           entry_z: float = 1.0):
    z = rolling_zscore(spread, window=window)
    sig_A = pd.Series(0, index=z.index)
    sig_B = pd.Series(0, index=z.index)

    sig_A[z > entry_z] = -1     # Short A
    sig_B[z > entry_z] = 1      # Long B

    sig_A[z < -entry_z] = 1     # Long A
    sig_B[z < -entry_z] = -1    # Short B

    return sig_A, sig_B, z

def backtest_pairs_zscore(prices_A: pd.Series,
                          prices_B: pd.Series,
                          window: int = 20,
                          entry_z: float = 1.0,
                          cost_per_turnover: float = 0.0005,
                          periods_per_year: int = 252):
    spread = build_spread(prices_A, prices_B)
    sig_A, sig_B, z = generate_pairs_signals(spread, window=window, entry_z=entry_z)

    sig_A = sig_A.shift(1).fillna(0)
    sig_B = sig_B.shift(1).fillna(0)

    returns_A = prices_A.pct_change().dropna()
    returns_B = prices_B.pct_change().dropna()

    # align indices
    sig_A = sig_A.loc[returns_A.index]
    sig_B = sig_B.loc[returns_B.index]

    pair_ret = sig_A * returns_A + sig_B * returns_B

    turnover_A = sig_A.diff().abs()
    turnover_B = sig_B.diff().abs()
    turnover_A = turnover_A.loc[pair_ret.index]
    turnover_B = turnover_B.loc[pair_ret.index]

    pair_ret_tc = pair_ret - cost_per_turnover * (turnover_A + turnover_B)
    pair_ret_tc = pair_ret_tc.dropna()

    equity = (1 + pair_ret_tc).cumprod()

    # Metrics
    mu = pair_ret_tc.mean()
    sigma = pair_ret_tc.std()
    sharpe = np.sqrt(periods_per_year) * mu / sigma if sigma != 0 else np.nan

    stats = {
        "mean_return": mu,
        "volatility": sigma,
        "sharpe": sharpe,
    }

    return equity, pair_ret_tc, stats, z
