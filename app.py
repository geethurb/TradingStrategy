from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252
REBALANCE_FREQ_MAP = {
    "Daily": "D",
    "Weekly": "W-FRI",
    "Monthly": "ME",
    "Quarterly": "QE",
}

PRESET_UNIVERSES: Dict[str, List[str]] = {
    "Dow 30": [
        "AAPL",
        "AMGN",
        "AXP",
        "BA",
        "CAT",
        "CRM",
        "CSCO",
        "CVX",
        "DIS",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "PG",
        "TRV",
        "UNH",
        "V",
        "VZ",
        "WBA",
        "WMT",
        "XOM",
    ],
    "NASDAQ 100 Leaders": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "GOOG",
        "TSLA",
        "AVGO",
        "COST",
        "NFLX",
        "AMD",
        "ADBE",
        "PEP",
        "CSCO",
        "INTC",
        "QCOM",
        "TXN",
        "AMGN",
        "INTU",
        "AMAT",
        "BKNG",
        "CMCSA",
        "SBUX",
        "ADP",
        "GILD",
        "MDLZ",
        "ISRG",
        "LRCX",
        "MU",
    ],
    "S&P 100 Core": [
        "AAPL",
        "ABBV",
        "ABT",
        "ACN",
        "ADBE",
        "AIG",
        "AMD",
        "AMGN",
        "AMT",
        "AMZN",
        "AXP",
        "BA",
        "BAC",
        "BK",
        "BKNG",
        "BLK",
        "C",
        "CAT",
        "CMCSA",
        "COP",
        "COST",
        "CRM",
        "CSCO",
        "CVS",
        "CVX",
        "DHR",
        "DIS",
        "EMR",
        "F",
        "GE",
        "GILD",
        "GM",
        "GOOG",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KO",
        "LIN",
        "LLY",
        "MA",
        "MCD",
        "MDT",
        "META",
        "MMM",
        "MO",
        "MRK",
        "MS",
        "MSFT",
        "NEE",
        "NFLX",
        "NKE",
        "NVDA",
        "ORCL",
        "PEP",
        "PFE",
        "PG",
        "PM",
        "QCOM",
        "RTX",
        "SBUX",
        "T",
        "TMO",
        "TSLA",
        "TXN",
        "UNH",
        "UNP",
        "UPS",
        "USB",
        "V",
        "VZ",
        "WFC",
        "WMT",
        "XOM",
    ],
    "US Sector ETFs": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XLC"],
    "Global Index ETFs": ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VNQ", "TLT", "LQD", "GLD", "DBC"],
}

STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "Buy & Hold Equal Weight": "Static equal-weight portfolio across the selected universe.",
    "SMA Crossover": "Long assets where fast SMA is above slow SMA.",
    "EMA Crossover": "Long assets where fast EMA is above slow EMA.",
    "RSI Mean Reversion": "Buy oversold assets (low RSI) for a configurable holding period.",
    "Bollinger Mean Reversion": "Buy when price is sufficiently below rolling mean in z-score terms.",
    "Donchian Breakout": "Trend-following breakout with entry/exit channels.",
    "Time-Series Momentum": "Long assets with positive trailing returns.",
    "Cross-Sectional Momentum": "Rebalance into top-N strongest assets by trailing return.",
    "Cross-Sectional Mean Reversion": "Rebalance into bottom-N weakest assets by trailing return.",
    "Inverse Volatility (Risk Parity Lite)": "Rebalance by inverse volatility weights.",
}


def parse_tickers(text: str) -> List[str]:
    if not text:
        return []
    raw = re.split(r"[\s,;]+", text.upper())
    cleaned = [ticker.strip() for ticker in raw if ticker.strip()]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(cleaned))


def normalize_long_only(weights: pd.DataFrame) -> pd.DataFrame:
    weights = weights.clip(lower=0.0)
    row_sum = weights.sum(axis=1).replace(0.0, np.nan)
    normalized = weights.div(row_sum, axis=0)
    return normalized.fillna(0.0)


def get_rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> List[pd.Timestamp]:
    if frequency == "D":
        return list(index)

    marker = pd.Series(1, index=index)
    rebalance_dates: List[pd.Timestamp] = []
    for _, group in marker.groupby(pd.Grouper(freq=frequency)):
        if not group.empty:
            rebalance_dates.append(group.index[-1])
    return rebalance_dates


@st.cache_data(show_spinner=False)
def download_prices(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    end_plus_one = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end_plus_one,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw["Close"].to_frame(name=tickers[0])

    close.columns = [str(col).upper() for col in close.columns]
    close = close.sort_index().ffill().dropna(how="all")
    return close


def equal_weight_positions(prices: pd.DataFrame) -> pd.DataFrame:
    available = prices.notna().astype(float)
    return normalize_long_only(available)


def sma_crossover(prices: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
    if fast_window >= slow_window:
        raise ValueError("Fast SMA window must be smaller than slow SMA window.")

    fast_ma = prices.rolling(fast_window).mean()
    slow_ma = prices.rolling(slow_window).mean()
    signal = (fast_ma > slow_ma).astype(float)
    signal = signal.where(prices.notna(), 0.0)
    return normalize_long_only(signal)


def ema_crossover(prices: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
    if fast_window >= slow_window:
        raise ValueError("Fast EMA window must be smaller than slow EMA window.")

    fast_ema = prices.ewm(span=fast_window, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_window, adjust=False).mean()
    signal = (fast_ema > slow_ema).astype(float)
    signal = signal.where(prices.notna(), 0.0)
    return normalize_long_only(signal)


def compute_rsi(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain.div(avg_loss.replace(0.0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def rsi_mean_reversion(prices: pd.DataFrame, window: int, oversold: float, hold_days: int) -> pd.DataFrame:
    rsi = compute_rsi(prices, window)
    signal = (rsi < oversold).astype(float)
    if hold_days > 1:
        signal = signal.rolling(hold_days, min_periods=1).max()
    signal = signal.where(prices.notna(), 0.0)
    return normalize_long_only(signal)


def bollinger_mean_reversion(prices: pd.DataFrame, window: int, z_entry: float, hold_days: int) -> pd.DataFrame:
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std().replace(0.0, np.nan)
    z_score = (prices - mean).div(std)
    signal = (z_score < -abs(z_entry)).astype(float)
    if hold_days > 1:
        signal = signal.rolling(hold_days, min_periods=1).max()
    signal = signal.where(prices.notna(), 0.0)
    return normalize_long_only(signal)


def donchian_breakout(prices: pd.DataFrame, entry_window: int, exit_window: int) -> pd.DataFrame:
    if exit_window >= entry_window:
        raise ValueError("Exit window should be smaller than entry window for Donchian breakout.")

    entry_level = prices.rolling(entry_window).max().shift(1)
    exit_level = prices.rolling(exit_window).min().shift(1)

    position = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for ticker in prices.columns:
        in_position = 0.0
        ticker_prices = prices[ticker]

        for idx in prices.index:
            px = ticker_prices.loc[idx]
            ent = entry_level.at[idx, ticker]
            ex = exit_level.at[idx, ticker]

            if pd.isna(px):
                position.at[idx, ticker] = in_position
                continue

            if in_position == 0.0 and pd.notna(ent) and px > ent:
                in_position = 1.0
            elif in_position == 1.0 and pd.notna(ex) and px < ex:
                in_position = 0.0

            position.at[idx, ticker] = in_position

    return normalize_long_only(position)


def time_series_momentum(prices: pd.DataFrame, lookback: int, threshold: float) -> pd.DataFrame:
    trailing_return = prices.pct_change(lookback)
    signal = (trailing_return > threshold).astype(float)
    signal = signal.where(prices.notna(), 0.0)
    return normalize_long_only(signal)


def cross_sectional_rank(
    prices: pd.DataFrame,
    lookback: int,
    top_n: int,
    rebalance_frequency: str,
    reverse: bool,
) -> pd.DataFrame:
    trailing_return = prices.pct_change(lookback)
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    rebalance_dates = get_rebalance_dates(prices.index, rebalance_frequency)
    for dt in rebalance_dates:
        if dt not in trailing_return.index:
            continue

        scores = trailing_return.loc[dt].dropna()
        if scores.empty:
            weights.loc[dt] = 0.0
            continue

        n = int(min(top_n, len(scores)))
        if n <= 0:
            weights.loc[dt] = 0.0
            continue

        selected = scores.nsmallest(n).index if reverse else scores.nlargest(n).index
        row = pd.Series(0.0, index=prices.columns)
        row.loc[selected] = 1.0 / n
        weights.loc[dt] = row

    weights = weights.ffill().fillna(0.0)
    weights = weights.where(prices.notna(), 0.0)
    return normalize_long_only(weights)


def inverse_volatility(prices: pd.DataFrame, vol_window: int, rebalance_frequency: str) -> pd.DataFrame:
    returns = prices.pct_change()
    trailing_vol = returns.rolling(vol_window).std()
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    rebalance_dates = get_rebalance_dates(prices.index, rebalance_frequency)
    for dt in rebalance_dates:
        if dt not in trailing_vol.index:
            continue

        vol = trailing_vol.loc[dt].replace(0.0, np.nan).dropna()
        if vol.empty:
            weights.loc[dt] = 0.0
            continue

        inverse = 1.0 / vol
        scaled = inverse / inverse.sum()

        row = pd.Series(0.0, index=prices.columns)
        row.loc[scaled.index] = scaled.values
        weights.loc[dt] = row

    weights = weights.ffill().fillna(0.0)
    weights = weights.where(prices.notna(), 0.0)
    return normalize_long_only(weights)


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float,
    fee_bps: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    returns = prices.pct_change().fillna(0.0)
    weights = target_weights.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)
    weights = weights.where(prices.notna(), 0.0)
    weights = normalize_long_only(weights)

    live_weights = weights.shift(1).fillna(0.0)
    gross_returns = (live_weights * returns).sum(axis=1)

    turnover = (weights - live_weights).abs().sum(axis=1)
    transaction_cost = turnover * (fee_bps / 10_000.0)

    net_returns = gross_returns - transaction_cost
    equity_curve = initial_capital * (1.0 + net_returns).cumprod()

    return net_returns, equity_curve, turnover


def performance_metrics(returns: pd.Series, equity_curve: pd.Series) -> Dict[str, float]:
    if returns.empty or equity_curve.empty:
        return {key: np.nan for key in ["Total Return", "CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar", "Win Rate"]}

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    years = len(returns) / TRADING_DAYS_PER_YEAR
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    annual_return = returns.mean() * TRADING_DAYS_PER_YEAR
    volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = annual_return / volatility if volatility > 0 else np.nan

    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    win_rate = (returns > 0).mean()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Calmar": calmar,
        "Win Rate": win_rate,
    }


def build_strategy_weights(strategy_name: str, prices: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    if strategy_name == "Buy & Hold Equal Weight":
        return equal_weight_positions(prices)
    if strategy_name == "SMA Crossover":
        return sma_crossover(prices, int(params["fast_window"]), int(params["slow_window"]))
    if strategy_name == "EMA Crossover":
        return ema_crossover(prices, int(params["fast_window"]), int(params["slow_window"]))
    if strategy_name == "RSI Mean Reversion":
        return rsi_mean_reversion(
            prices,
            window=int(params["rsi_window"]),
            oversold=float(params["oversold"]),
            hold_days=int(params["hold_days"]),
        )
    if strategy_name == "Bollinger Mean Reversion":
        return bollinger_mean_reversion(
            prices,
            window=int(params["bb_window"]),
            z_entry=float(params["z_entry"]),
            hold_days=int(params["hold_days"]),
        )
    if strategy_name == "Donchian Breakout":
        return donchian_breakout(
            prices,
            entry_window=int(params["entry_window"]),
            exit_window=int(params["exit_window"]),
        )
    if strategy_name == "Time-Series Momentum":
        return time_series_momentum(
            prices,
            lookback=int(params["lookback"]),
            threshold=float(params["threshold"]),
        )
    if strategy_name == "Cross-Sectional Momentum":
        return cross_sectional_rank(
            prices,
            lookback=int(params["lookback"]),
            top_n=int(params["top_n"]),
            rebalance_frequency=str(params["rebalance_freq"]),
            reverse=False,
        )
    if strategy_name == "Cross-Sectional Mean Reversion":
        return cross_sectional_rank(
            prices,
            lookback=int(params["lookback"]),
            top_n=int(params["top_n"]),
            rebalance_frequency=str(params["rebalance_freq"]),
            reverse=True,
        )
    if strategy_name == "Inverse Volatility (Risk Parity Lite)":
        return inverse_volatility(
            prices,
            vol_window=int(params["vol_window"]),
            rebalance_frequency=str(params["rebalance_freq"]),
        )

    raise ValueError(f"Unknown strategy: {strategy_name}")


def render_metrics_table(strategy_returns: pd.Series, strategy_equity: pd.Series, benchmark_returns: pd.Series, benchmark_equity: pd.Series) -> None:
    strategy_stats = performance_metrics(strategy_returns, strategy_equity)
    benchmark_stats = performance_metrics(benchmark_returns, benchmark_equity)

    table = pd.DataFrame([strategy_stats, benchmark_stats], index=["Strategy", "Equal-Weight Benchmark"])

    percent_cols = ["Total Return", "CAGR", "Volatility", "Max Drawdown", "Win Rate"]
    for col in percent_cols:
        table[col] = table[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    number_cols = ["Sharpe", "Calmar"]
    for col in number_cols:
        table[col] = table[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    st.dataframe(table, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Quant Strategy Backtester", layout="wide")

    st.title("Quant Strategy Backtester")
    st.caption("Backtest common long-only quantitative strategies with user-defined universes and parameters.")

    with st.sidebar:
        st.header("1) Universe")
        source = st.radio("Universe source", ["Preset", "Custom"], horizontal=True)

        selected_tickers: List[str]
        if source == "Preset":
            preset_name = st.selectbox("Preset universe", list(PRESET_UNIVERSES.keys()))
            selected_tickers = PRESET_UNIVERSES[preset_name].copy()
            extras = parse_tickers(st.text_input("Optional extra tickers", ""))
            selected_tickers.extend(extras)
        else:
            custom_text = st.text_area(
                "Tickers (comma / space separated)",
                value="AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, XOM, UNH, JNJ",
                height=110,
            )
            selected_tickers = parse_tickers(custom_text)

        selected_tickers = list(dict.fromkeys([ticker.upper() for ticker in selected_tickers]))
        max_tickers = st.slider("Max tickers", min_value=1, max_value=200, value=min(30, max(1, len(selected_tickers))), step=1)
        selected_tickers = selected_tickers[:max_tickers]
        st.caption(f"Selected: {len(selected_tickers)} ticker(s)")

        st.header("2) Data & Costs")
        default_start = date.today() - timedelta(days=365 * 5)
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=date.today())
        initial_capital = st.number_input("Initial capital ($)", min_value=1_000.0, value=100_000.0, step=1_000.0)
        fee_bps = st.number_input("Transaction cost (bps per 1.0 turnover)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)

        st.header("3) Strategy")
        strategy_name = st.selectbox("Strategy", list(STRATEGY_DESCRIPTIONS.keys()))
        st.caption(STRATEGY_DESCRIPTIONS[strategy_name])

        params: Dict[str, float | str] = {}
        if strategy_name in {"SMA Crossover", "EMA Crossover"}:
            params["fast_window"] = st.number_input("Fast window", min_value=2, max_value=300, value=20, step=1)
            params["slow_window"] = st.number_input("Slow window", min_value=5, max_value=500, value=100, step=1)

        elif strategy_name == "RSI Mean Reversion":
            params["rsi_window"] = st.number_input("RSI window", min_value=2, max_value=100, value=14, step=1)
            params["oversold"] = st.slider("Oversold threshold", min_value=5, max_value=50, value=30, step=1)
            params["hold_days"] = st.number_input("Hold days after signal", min_value=1, max_value=30, value=5, step=1)

        elif strategy_name == "Bollinger Mean Reversion":
            params["bb_window"] = st.number_input("Rolling window", min_value=5, max_value=150, value=20, step=1)
            params["z_entry"] = st.slider("Entry z-score", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
            params["hold_days"] = st.number_input("Hold days after signal", min_value=1, max_value=30, value=5, step=1)

        elif strategy_name == "Donchian Breakout":
            params["entry_window"] = st.number_input("Entry window", min_value=10, max_value=300, value=55, step=1)
            params["exit_window"] = st.number_input("Exit window", min_value=5, max_value=200, value=20, step=1)

        elif strategy_name == "Time-Series Momentum":
            params["lookback"] = st.number_input("Lookback days", min_value=5, max_value=300, value=126, step=1)
            params["threshold"] = st.slider("Minimum trailing return", min_value=-0.20, max_value=0.20, value=0.0, step=0.01)

        elif strategy_name in {"Cross-Sectional Momentum", "Cross-Sectional Mean Reversion"}:
            params["lookback"] = st.number_input("Lookback days", min_value=5, max_value=300, value=63, step=1)
            params["top_n"] = st.number_input("Number of holdings", min_value=1, max_value=max(1, len(selected_tickers)), value=min(5, max(1, len(selected_tickers))), step=1)
            selected_freq = st.selectbox("Rebalance frequency", list(REBALANCE_FREQ_MAP.keys()), index=2)
            params["rebalance_freq"] = REBALANCE_FREQ_MAP[selected_freq]

        elif strategy_name == "Inverse Volatility (Risk Parity Lite)":
            params["vol_window"] = st.number_input("Volatility window", min_value=10, max_value=252, value=60, step=1)
            selected_freq = st.selectbox("Rebalance frequency", list(REBALANCE_FREQ_MAP.keys()), index=2)
            params["rebalance_freq"] = REBALANCE_FREQ_MAP[selected_freq]

        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Configure inputs in the sidebar and click Run Backtest.")
        return

    if not selected_tickers:
        st.error("Please select at least one ticker.")
        return

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return

    with st.spinner("Downloading historical prices..."):
        prices = download_prices(tuple(selected_tickers), start_date.isoformat(), end_date.isoformat())

    if prices.empty:
        st.error("No price data returned. Check your ticker list and date range.")
        return

    min_obs = max(40, int(0.40 * len(prices)))
    prices = prices.dropna(axis=1, thresh=min_obs)
    if prices.empty:
        st.error("Not enough valid history for the selected universe in this date range.")
        return

    downloaded_tickers = list(prices.columns)
    missing_tickers = sorted(set(selected_tickers) - set(downloaded_tickers))

    st.subheader("Universe")
    st.write(f"Using {len(downloaded_tickers)} ticker(s) with sufficient data.")
    st.code(", ".join(downloaded_tickers), language="text")
    if missing_tickers:
        st.warning(f"Skipped {len(missing_tickers)} ticker(s) with no/insufficient data: {', '.join(missing_tickers)}")

    try:
        strategy_weights = build_strategy_weights(strategy_name, prices, params)
    except ValueError as exc:
        st.error(str(exc))
        return

    strategy_returns, strategy_equity, turnover = run_backtest(prices, strategy_weights, initial_capital, fee_bps)

    benchmark_weights = equal_weight_positions(prices)
    benchmark_returns, benchmark_equity, _ = run_backtest(prices, benchmark_weights, initial_capital, fee_bps=0.0)

    st.subheader("Performance")
    render_metrics_table(strategy_returns, strategy_equity, benchmark_returns, benchmark_equity)

    equity_frame = pd.DataFrame(
        {
            "Strategy": strategy_equity,
            "Equal-Weight Benchmark": benchmark_equity,
        }
    )

    fig_equity = go.Figure()
    for col in equity_frame.columns:
        fig_equity.add_trace(go.Scatter(x=equity_frame.index, y=equity_frame[col], mode="lines", name=col))
    fig_equity.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value")
    st.plotly_chart(fig_equity, use_container_width=True)

    strategy_drawdown = strategy_equity / strategy_equity.cummax() - 1.0
    benchmark_drawdown = benchmark_equity / benchmark_equity.cummax() - 1.0

    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(x=strategy_drawdown.index, y=strategy_drawdown, mode="lines", name="Strategy"))
    fig_drawdown.add_trace(go.Scatter(x=benchmark_drawdown.index, y=benchmark_drawdown, mode="lines", name="Benchmark"))
    fig_drawdown.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown")
    st.plotly_chart(fig_drawdown, use_container_width=True)

    latest_weights = strategy_weights.iloc[-1]
    latest_weights = latest_weights[latest_weights > 0].sort_values(ascending=False)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Current Holdings")
        if latest_weights.empty:
            st.write("No active positions on the latest date.")
        else:
            holdings_df = latest_weights.rename("Weight").to_frame()
            holdings_df["Weight"] = holdings_df["Weight"].map(lambda x: f"{x:.2%}")
            st.dataframe(holdings_df, use_container_width=True)

    with col_right:
        st.subheader("Turnover")
        turnover_summary = pd.DataFrame(
            {
                "Average Daily Turnover": [f"{turnover.mean():.2f}"],
                "Median Daily Turnover": [f"{turnover.median():.2f}"],
                "Max Daily Turnover": [f"{turnover.max():.2f}"],
            }
        )
        st.dataframe(turnover_summary, use_container_width=True)


if __name__ == "__main__":
    main()
