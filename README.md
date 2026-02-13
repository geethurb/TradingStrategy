# Quant Strategy Backtester (Streamlit)

A Streamlit app for backtesting common long-only quantitative equity strategies with user-defined parameters and stock universes.

## Features
- Universe selection:
  - Preset universes (Dow 30, NASDAQ leaders, S&P core, ETFs)
  - Custom ticker list input
- Common quant strategies:
  - Buy & Hold Equal Weight
  - SMA Crossover
  - EMA Crossover
  - RSI Mean Reversion
  - Bollinger Mean Reversion
  - Donchian Breakout
  - Time-Series Momentum
  - Cross-Sectional Momentum
  - Cross-Sectional Mean Reversion
  - Inverse Volatility (Risk Parity Lite)
- User-defined backtest inputs:
  - Date range
  - Initial capital
  - Transaction costs
  - Strategy-specific parameters
- Results:
  - Strategy vs benchmark performance metrics
  - Equity and drawdown charts
  - Latest holdings and turnover statistics

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Data source: Yahoo Finance via `yfinance`.
- This is a research/education tool and not investment advice.
