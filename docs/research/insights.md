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
