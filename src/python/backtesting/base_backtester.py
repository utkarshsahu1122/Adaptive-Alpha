import pandas as pd
import numpy as np

class BaseBacktester:
    def __init__(self, returns: pd.Series, cost_per_turnover: float = 0.0005):
        self.returns = returns.dropna()
        self.cost_per_turnover = cost_per_turnover

    def _align_signals(self, signals: pd.Series) -> pd.Series:
        signals = signals.reindex(self.returns.index).fillna(0.0)
        return signals

    def run(self, signals: pd.Series):
        signals = self._align_signals(signals)

        position = signals.shift(1).fillna(0.0)

        strat_ret = position * self.returns

        turnover = position.diff().abs().fillna(0.0)
        strat_ret = strat_ret - self.cost_per_turnover * turnover

        equity = (1.0 + strat_ret).cumprod()
        return equity, strat_ret

    # We’ll add a method that stops trading when drawdown exceeds a threshold (say 5%).
    # Drawdown is the decline in an investment's value from its peak to its lowest trough before a new peak is reached.
    def run_with_stoploss(self, signals: pd.Series, stop_drawdown: float = 0.05):
        """
        Global stop: once max drawdown exceeds stop_drawdown (e.g. 0.05 = 5%),
        all positions are closed and remain flat until the end.
        """
        signals = self._align_signals(signals)

        r = self.returns
        equity_vals = []
        strat_rets = []

        current_equity = 1.0
        peak_equity = 1.0
        stopped = False
        prev_position = 0.0

        for t, (date, ret_t) in enumerate(r.items()):
            if not stopped:
                # use yesterday's signal
                if t == 0:
                    position = 0.0
                else:
                    position = float(signals.iloc[t-1])
            else:
                position = 0.0

            # compute return & cost
            position_change = position - prev_position
            trade_cost = self.cost_per_turnover * abs(position_change)
            strat_ret_t = position * ret_t - trade_cost

            current_equity *= (1.0 + strat_ret_t)
            peak_equity = max(peak_equity, current_equity)
            drawdown = 1.0 - current_equity / peak_equity

            # check stop-loss
            if drawdown > stop_drawdown:
                stopped = True
                position = 0.0  # flatten

            equity_vals.append(current_equity)
            strat_rets.append(strat_ret_t)
            prev_position = position

        equity = pd.Series(equity_vals, index=r.index, name="equity")
        strat_ret = pd.Series(strat_rets, index=r.index, name="strategy_ret")

        return equity, strat_ret

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        roll_max = equity.cummax()
        drawdown = equity / roll_max - 1.0
        return drawdown.min()  # negative number

    def performance(self, strat_ret: pd.Series, equity: pd.Series | None = None):
        strat_ret = strat_ret.dropna()

        mean = strat_ret.mean()
        vol = strat_ret.std()
        sharpe = np.sqrt(252) * mean / vol if vol != 0 else np.nan

        if equity is None:
            equity = (1.0 + strat_ret).cumprod()

        mdd = self.max_drawdown(equity)

        # win-rate: % of positive-return days
        win_rate = (strat_ret > 0).mean()

        return {
            "mean": mean,
            "vol": vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "win_rate": win_rate,
        }
