import pandas as pd
import numpy as np

def volatility(returns):
    return returns.std() * (252 ** 0.5)

def risk_parity_weights(returns: pd.DataFrame):
    vol = volatility(returns)
    inv = 1 / vol
    w = inv / inv.sum()
    return w

def portfolio_returns(returns: pd.DataFrame, weights):
    return (returns * weights).sum(axis=1)
