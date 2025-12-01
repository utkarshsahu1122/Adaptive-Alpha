import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

class AR1Model:
    def __init__(self):
        self.model = None
        self.results = None

    def fit(self, series: pd.Series):
        series = series.dropna()
        self.model = AutoReg(series, lags=1, old_names=False)
        self.results = self.model.fit()
        return self.results

    def predict(self, start=None, end=None):
        if self.results is None:
            raise ValueError("Model not fitted yet.")
        return self.results.predict(start=start, end=end)

    def one_step_ahead(self, series: pd.Series):
        """Fit on all but last, predict last."""
        series = series.dropna()
        self.fit(series.iloc[:-1])
        pred_last = self.predict(start=series.index[-1], end=series.index[-1])
        return pred_last.iloc[0]
