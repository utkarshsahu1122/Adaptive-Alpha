import pandas as pd
import numpy as np

''' Z-score tells you how many standard deviations a value is away from the mean.
 Rolling z-score: How far today's value is from the last window days’ mean, measured in window-day standard deviations.'''
def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std
    return z

def generate_zscore_signals(series: pd.Series,
                            window: int = 20,
                            entry_z: float = 1.0) -> pd.Series:
    z = rolling_zscore(series, window=window)
    signals = pd.Series(0, index=z.index)
    signals[z < -entry_z] = 1    # long
    signals[z > entry_z] = -1    # short
    return signals

def backtest_zscore_strategy(returns: pd.Series,
                             window: int = 20,
                             entry_z: float = 1.0,
                             cost_per_turnover: float = 0.0005,
                             periods_per_year: int = 252):
    """Returns equity curve and performance dict."""

    ''' Practical notes:

    - Often you convert these signals into trades by shifting (signals.shift(1)) so you act on yesterday’s signal at today’s open.
    - You may want to dropna or forward-fill if you need continuous positions.
    - Thresholds (±1) are arbitrary — you can tune them (±2, use hysteresis, or add stop/loss).
    '''

    signals = generate_zscore_signals(returns, window=window, entry_z=entry_z)
    position = signals.shift(1).fillna(0)
    strat_ret = position * returns

    ''' Optionally include a basic transaction cost per trade (e.g., 5 bps = 0.0005):
    - In trading, BPS (Basis Points) is a unit of measurement for small percentage changes, where 1 basis point equals 0.01%.

    Turnover = number of trades you make.
    A trade happens when your position changes.'''

    turnover = position.diff().abs()
    strat_ret_tc = strat_ret - cost_per_turnover * turnover

    equity = (1 + strat_ret_tc.dropna()).cumprod()

    ''' Sharpe ratio is a measure of risk-adjusted return that quantifies how much excess return a trading strategy generates for each unit of risk taken.
    Higher Sharpe = better strategy.
    '''
    mu = strat_ret_tc.mean()
    sigma = strat_ret_tc.std()
    sharpe = np.sqrt(periods_per_year) * mu / sigma if sigma != 0 else np.nan

    stats = {
        "mean_return": mu,
        "volatility": sigma,
        "sharpe": sharpe,
    }
    return equity, strat_ret_tc, stats
