## DAY 1 – Randomness Analysis
- Collected daily returns for 5 tech stocks.
- Observed non-Gaussian behavior (fat tails, volatility clustering).
- Data stored as data/market_returns.csv.
- Next: model this stochastic behavior (Brownian motion → Day 5).

## DAY 2 – Volatility and Correlation Insights
- Computed annualized volatility for 5 assets.
- Observed volatility clustering in rolling window.
- Generated correlation heatmap for cross-asset relationships.
- These metrics will feed the risk module for Adaptive Alpha.
- Next: build first mean-reversion signal using z-scores (Day 3).

## Day 3 – Z-Score Mean Reversion Strategy

- Built a rolling z-score indicator on daily returns for AAPL (and can generalize to other assets).
- Trading rule: go long when z < -1, short when z > +1, flat otherwise, with basic transaction costs.
- Implemented a reusable strategy module in `src/python/strategies/zscore_strategy.py`.
- Measured performance via Sharpe ratio and equity curve.
- Observed [e.g. strategy only works in some regimes / sensitive to window length / suffers after costs].
- This strategy becomes Strategy #1 in Adaptive Alpha and later will:
  - Serve as a baseline against more complex ML/RL-driven strategies.
  - Be extended to pair-spread z-scores and portfolio-level signals.

## Day 4 – AAPL/MSFT Pairs Z-score Mean Reversion Strategy

- Constructed log spread between AAPL and MSFT and standardized it via rolling z-score.
- Trading rule: 
  - If spread z > +1: short AAPL, long MSFT.
  - If spread z < -1: long AAPL, short MSFT.
- Implemented pair strategy module in `src/python/strategies/pairs_zscore_strategy.py`.
- Measured equity curve and Sharpe after transaction costs.
- Key observations:
  - [“Strategy performs well in range-bound periods{stationary behaviour}, degrades during trending regimes.”]
- This becomes Strategy #2 for Adaptive Alpha and will later:
  - Serve as a structured baseline for more advanced cointegration + ML-based spread models.
  - Be integrated into portfolio/risk layer and possibly used as one of multiple signal components.

## Day 5 – Time Series Basics (Stationarity & AR(1))

- Explored time series behavior of AAPL returns and AAPL/MSFT spread.
- Visual checks and ACF plots suggested returns are near white noise, while spread/differences show some temporal structure.
- Fitted an AR(1) model to AAPL returns using statsmodels:
  - Recorded phi (lag-1 coefficient) and analyzed its implication for predictability.
- Implemented a reusable AR1Model class in `src/python/models/time_series_models.py`.
- Next steps:
  - Use AR(1) and related models to build simple forecasting features for strategies.
  - Later replace AR(1) with more powerful models (e.g., LSTM, DeepLOB).

## Day 6 – Backtesting Engine

- Designed a reusable backtester for single-asset strategies.
- Supports:
  - execution timing
  - transaction costs
  - P&L and Sharpe calculation
- Unified Strategy #1 and Strategy #2 into one evaluation interface.
- This structure mimics professional quant architecture:
  strategy code ≠ execution engine.
- Next: extend to multi-asset & risk layer.

## Day 7 – Portfolio Risk Layer

- Implemented volatility-based portfolio.
- Calculated portfolio returns, volatility, and Sharpe ratio.
- Built risk module in `src/python/risk/portfolio.py`.
- Introduced risk-parity allocation.
- First step towards:
  - multi-asset allocation
  - portfolio-level evaluation
  - RL risk constraints
- Risk management layer now available for strategies.
