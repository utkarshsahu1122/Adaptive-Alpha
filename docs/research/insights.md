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
