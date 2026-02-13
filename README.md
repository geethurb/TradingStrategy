# Quant Strategy Backtester (Streamlit)

A Streamlit app for backtesting common quantitative equity strategies with user-defined parameters, universes, and portfolio construction controls.

## Features
- Universe selection:
  - Preset universes (Dow 30, NASDAQ leaders, S&P core, ETFs)
  - Index/cap universes (Large Cap S&P 500, Mid Cap S&P 400, Small Cap S&P 600, CRSP US Total Market, Russell 3000, Russell 2000, Russell Microcap, NASDAQ-100)
  - Major S&P 500 sector universes (Technology, Health Care, Financials, Industrials, etc.)
  - Custom ticker list input
- Data layer:
  - Incremental local SQLite price store at `data/price_history.sqlite3`
  - Stores both daily close and daily volume (schema auto-upgrades if volume column was missing)
  - Stores per-ticker fundamental snapshots (P/E, P/B, EPS) with periodic refresh
  - Uses a unified fetch path for price+volume (single query/download pass) for efficiency
  - Reuses database data when coverage exists
  - Downloads only missing date/ticker segments and upserts into the database
  - If a ticker is still all-missing in the requested range after DB read, the app retries a direct download for that ticker
  - Leading missing data is treated as pre-listing/IPO period (not forced to zero)
  - Delisting assumption for partial histories: if a ticker stops reporting prices and remains missing for 20+ trading days, subsequent dates are treated as price = 0
- Common quant strategies:
  - Buy & Hold Equal Weight
  - SMA/EMA Crossover (with optional hysteresis band)
  - MACD Trend
  - Moving Average Reversion (signal-based / fixed-days / hybrid exits)
  - RSI Mean Reversion (signal-based / fixed-days / hybrid exits)
  - Bollinger Mean Reversion (signal-based / fixed-days / hybrid exits)
  - Donchian Breakout
  - Time-Series Momentum (entry/exit thresholds + rebalance frequency)
  - Cross-Sectional Momentum
  - Dual Momentum (relative + absolute filter)
  - Volatility-Adjusted Momentum
  - 52-Week High Rotation
  - Cross-Sectional Mean Reversion
  - Inverse Volatility (Risk Parity Lite)
  - Low Volatility Rotation
- User-defined backtest inputs:
  - Date range
  - Initial capital
  - Transaction costs
  - Full selected universe is always considered for signals
  - Position side mode (`Long Only`, `Long/Short`, `Short Only`)
  - Holding weighting mode (`Equal Weight`, `Signal Strength`, `Signal Strength / Volatility`, `Hold Until Sell (No Rebalance)`)
  - Hold-until-sell mode can enforce a max single-position weight cap (% of portfolio)
  - Global maximum holdings cap (limits number of concurrent portfolio positions)
  - If holdings cap binds, candidates are ranked by signal strength and top-ranked names are selected
  - No-leverage execution rule: buy notional is capped by available cash (after same-day sells and fees)
  - Default-on illiquidity filter (minimum median dollar volume OR minimum median shares traded over configurable lookback)
  - Optional fundamental filters (Trailing P/E, Price/Book, Trailing EPS) with a minimum-coverage guard so filtering is skipped when data is too sparse
  - Manual ticker exclusion list (ignore selected stocks during backtest)
  - Strategy-specific parameters
  - Optional stop loss (disabled unless explicitly enabled)
- Results:
  - Strategy vs benchmark performance metrics (Equal-Weight Universe, Market-Cap Universe Buy & Hold initial weights, S&P 500 via SPY, NASDAQ-100 via QQQ)
  - Equity and drawdown charts
  - Final ending portfolio (positions + cash)
  - Executed trade log (date, action, ticker, shares, price, fees) with filters and CSV export
  - Extreme short-period move attribution (which stocks drove outsized gains/losses)
  - Turnover and trade activity statistics
- Live signals page:
  - Run any strategy on the selected universe using latest market data
  - Long and short recommendations (buy/short, increases/reductions, exits, flips)
  - Current target portfolio weights and allocation values
  - Filtered full-signal table with CSV export
- Strategy explainer page:
  - Plain-English, parameter-aware explanation of each strategy
  - Clarifies entry/exit rules, rebalance logic, and allocation behavior
  - Explains how side mode, weighting scheme, and holdings cap change target construction
  - Includes action definitions for long and short flows (`BUY`, `SHORT`, `INCREASE LONG/SHORT`, `SELL`, `COVER`, `FLIP`)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Data source: Yahoo Finance via `yfinance`.
- Fundamental metrics are point-in-time snapshots from Yahoo Finance and can be missing for some symbols (especially non-equities and loss-making names); the app can skip fundamental filtering if selected metrics fail the configured coverage threshold.
- Backtest execution uses a 1-trading-day signal lag to reduce lookahead bias.
- Trades are executed in whole shares only (no fractional shares).
- Strategies never allocate to NaN/zero-price assets; execution also blocks trades when price is non-positive.
- This is a research/education tool and not investment advice.
