import pandas as pd
from .base_backtester import BaseBacktester

class SingleAssetBacktester(BaseBacktester):
    def __init__(self, returns: pd.Series, cost_per_turnover: float = 0.0005):
        super().__init__(returns, cost_per_turnover)

    def run_strategy(self, signals: pd.Series):
        equity, strat_ret = self.run(signals)
        stats = self.performance(strat_ret, equity)
        return equity, strat_ret, stats

    def run_strategy_with_stoploss(self, signals: pd.Series, stop_drawdown: float = 0.05):
        equity, strat_ret = self.run_with_stoploss(signals, stop_drawdown=stop_drawdown)
        stats = self.performance(strat_ret, equity)
        stats["stop_drawdown"] = stop_drawdown
        return equity, strat_ret, stats
