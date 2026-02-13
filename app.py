from __future__ import annotations

import re
import io
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

TRADING_DAYS_PER_YEAR = 252
REBALANCE_FREQ_MAP = {
    "Daily": "D",
    "Weekly": "W-FRI",
    "Monthly": "ME",
    "Quarterly": "QE",
}
REBALANCE_FREQ_LABEL_MAP = {value: key for key, value in REBALANCE_FREQ_MAP.items()}
VALID_REBALANCE_FREQUENCIES = set(REBALANCE_FREQ_MAP.values())
EXIT_MODE_OPTIONS = ["Signal-Based", "Fixed Days", "Hybrid"]
POSITION_SIDE_OPTIONS = ["Long Only", "Long/Short", "Short Only"]
WEIGHTING_SCHEME_OPTIONS = [
    "Equal Weight",
    "Signal Strength",
    "Signal Strength / Volatility",
    "Hold Until Sell (No Rebalance)",
]
EXECUTION_STYLE_TARGET = "Target Rebalance"
EXECUTION_STYLE_HOLD = "Hold Until Sell"
SP500_SECTOR_UNIVERSE_MAP = {
    "Sector: Information Technology": "Information Technology",
    "Sector: Health Care": "Health Care",
    "Sector: Financials": "Financials",
    "Sector: Consumer Discretionary": "Consumer Discretionary",
    "Sector: Communication Services": "Communication Services",
    "Sector: Industrials": "Industrials",
    "Sector: Consumer Staples": "Consumer Staples",
    "Sector: Energy": "Energy",
    "Sector: Utilities": "Utilities",
    "Sector: Real Estate": "Real Estate",
    "Sector: Materials": "Materials",
}
INDEX_UNIVERSE_OPTIONS = [
    "Large Cap (S&P 500)",
    "Mid Cap (S&P 400)",
    "Small Cap (S&P 600)",
    "CRSP US Total Market",
    "Russell 3000",
    "Russell 2000",
    "Russell Microcap",
    "NASDAQ-100",
] + list(SP500_SECTOR_UNIVERSE_MAP.keys())
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
INDEX_SOURCE_URLS = {
    "Russell 2000": "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
    "Russell 3000": "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund",
    "Russell Microcap": "https://www.ishares.com/us/products/239774/ishares-micro-cap-etf/1467271812596.ajax?fileType=csv&fileName=IWC_holdings&dataType=fund",
    "NASDAQ-100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "S&P 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
VANGUARD_HOLDINGS_API_TEMPLATE = "https://investor.vanguard.com/vmf/api/{fund_identifier}/portfolio-holding/stock.json"
PRICE_DB_PATH = Path(__file__).resolve().parent / "data" / "price_history.sqlite3"
DB_COVERAGE_TOLERANCE_DAYS = 3
DB_REFRESH_LOOKBACK_DAYS = 7
DELISTING_CONFIRMATION_DAYS = 20
EARLIEST_DATE_INPUT = date(1900, 1, 1)
FUNDAMENTAL_REFRESH_DAYS = 7
DEFAULT_FUNDAMENTAL_MIN_COVERAGE = 0.70
FUNDAMENTAL_METRIC_LABELS: Dict[str, str] = {
    "trailing_pe": "Trailing P/E",
    "forward_pe": "Forward P/E",
    "price_to_book": "Price/Book",
    "trailing_eps": "Trailing EPS",
    "forward_eps": "Forward EPS",
}
FUNDAMENTAL_METRIC_ORDER: Tuple[str, ...] = tuple(FUNDAMENTAL_METRIC_LABELS.keys())

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
    "SMA Crossover": "Trend-following crossover with optional hysteresis band to reduce whipsaws.",
    "EMA Crossover": "Trend-following crossover with optional hysteresis band to reduce whipsaws.",
    "MACD Trend": "Long when MACD histogram is above entry threshold and exit below exit threshold.",
    "Moving Average Reversion": "Buy deep pullbacks below moving average, then exit back toward/through the mean.",
    "RSI Mean Reversion": "Buy oversold assets and exit by signal, fixed days, or a hybrid of both.",
    "Bollinger Mean Reversion": "Buy deeply negative z-score moves and exit by signal, fixed days, or hybrid.",
    "Donchian Breakout": "Trend-following breakout with entry/exit channels.",
    "Time-Series Momentum": "Entry/exit by trailing-return thresholds with configurable rebalance timing.",
    "Cross-Sectional Momentum": "Rebalance into top-N strongest assets by trailing return.",
    "Dual Momentum": "Rebalance into top-N assets by relative momentum, filtered by absolute momentum.",
    "Volatility-Adjusted Momentum": "Rebalance into top-N assets by momentum divided by volatility.",
    "52-Week High Rotation": "Rebalance into assets closest to their trailing high (strength near highs).",
    "Cross-Sectional Mean Reversion": "Rebalance into bottom-N weakest assets by trailing return.",
    "Inverse Volatility (Risk Parity Lite)": "Rebalance by inverse volatility weights.",
    "Low Volatility Rotation": "Rebalance into top-N lowest-volatility assets.",
}


def normalize_ticker_list(tickers: List[str]) -> List[str]:
    normalized: List[str] = []
    for item in tickers:
        ticker = str(item).strip().upper()
        if not ticker or ticker in {"NAN", "N/A", "-"}:
            continue
        ticker = ticker.replace(".", "-").replace("/", "-")
        ticker = re.sub(r"\s+", "", ticker)
        if re.fullmatch(r"[A-Z0-9\-\^]+", ticker):
            normalized.append(ticker)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(normalized))


def parse_tickers(text: str) -> List[str]:
    if not text:
        return []
    raw = re.split(r"[\s,;]+", text.upper())
    return normalize_ticker_list([ticker.strip() for ticker in raw if ticker.strip()])


def to_finite_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def clean_positive_ratio(value: Any) -> float:
    numeric = to_finite_float(value)
    if pd.notna(numeric) and numeric > 0:
        return float(numeric)
    return np.nan


def sanitize_fundamental_metric_series(values: pd.Series, metric: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric))
    if metric in {"trailing_pe", "forward_pe", "price_to_book"}:
        numeric = numeric.where(numeric > 0)
    return numeric


def format_fundamental_filter_description(metric_filters: Dict[str, Dict[str, float]]) -> str:
    parts: List[str] = []
    for metric in FUNDAMENTAL_METRIC_ORDER:
        bounds = metric_filters.get(metric)
        if not bounds:
            continue
        label = FUNDAMENTAL_METRIC_LABELS.get(metric, metric)
        min_value = bounds.get("min")
        max_value = bounds.get("max")
        if min_value is not None and max_value is not None:
            parts.append(f"{label} in [{float(min_value):.2f}, {float(max_value):.2f}]")
        elif min_value is not None:
            parts.append(f"{label} >= {float(min_value):.2f}")
        elif max_value is not None:
            parts.append(f"{label} <= {float(max_value):.2f}")
    return "; ".join(parts)


def format_fundamental_coverage_summary(metric_coverage: Dict[str, float]) -> str:
    parts: List[str] = []
    for metric in FUNDAMENTAL_METRIC_ORDER:
        if metric in metric_coverage:
            label = FUNDAMENTAL_METRIC_LABELS.get(metric, metric)
            parts.append(f"{label}: {100.0 * float(metric_coverage[metric]):.0f}%")
    return ", ".join(parts)


def parse_ishares_holdings_tickers(url: str) -> List[str]:
    response = requests.get(url, headers=HTTP_HEADERS, timeout=30)
    response.raise_for_status()

    lines = response.text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Ticker,")), None)
    if header_idx is None:
        raise ValueError("Could not parse holdings file (Ticker header missing).")

    holdings = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    if "Ticker" not in holdings.columns:
        raise ValueError("Holdings file did not contain a Ticker column.")

    if "Asset Class" in holdings.columns:
        holdings = holdings[holdings["Asset Class"].astype(str).str.contains("Equity", case=False, na=False)]

    tickers = holdings["Ticker"].astype(str).tolist()
    return normalize_ticker_list(tickers)


def parse_nasdaq_100_tickers() -> List[str]:
    response = requests.get(INDEX_SOURCE_URLS["NASDAQ-100"], headers=HTTP_HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = None
    for candidate in soup.find_all("table", {"class": "wikitable"}):
        header = [th.get_text(" ", strip=True).lower() for th in candidate.find_all("tr")[0].find_all(["th", "td"])]
        if any("ticker" in h for h in header) and any("company" in h for h in header):
            table = candidate
            break

    if table is None:
        raise ValueError("Could not locate the NASDAQ-100 constituents table.")

    header_cells = [th.get_text(" ", strip=True).lower() for th in table.find_all("tr")[0].find_all(["th", "td"])]
    ticker_idx = next((i for i, name in enumerate(header_cells) if "ticker" in name or "symbol" in name), None)
    if ticker_idx is None:
        raise ValueError("NASDAQ-100 table missing ticker column.")

    tickers: List[str] = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if cells and len(cells) > ticker_idx:
            tickers.append(cells[ticker_idx].get_text(strip=True))

    return normalize_ticker_list(tickers)


def parse_wikipedia_ticker_table(url: str, required_header_keywords: Tuple[str, ...]) -> List[str]:
    response = requests.get(url, headers=HTTP_HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for candidate in soup.find_all("table", {"class": "wikitable"}):
        rows = candidate.find_all("tr")
        if not rows:
            continue

        header = [th.get_text(" ", strip=True).lower() for th in rows[0].find_all(["th", "td"])]
        if not all(any(keyword in h for h in header) for keyword in required_header_keywords):
            continue

        ticker_idx = next((i for i, name in enumerate(header) if "ticker" in name or "symbol" in name), None)
        if ticker_idx is None:
            continue

        tickers: List[str] = []
        for row in rows[1:]:
            cells = row.find_all("td")
            if cells and len(cells) > ticker_idx:
                tickers.append(cells[ticker_idx].get_text(strip=True))

        normalized = normalize_ticker_list(tickers)
        if normalized:
            return normalized

    raise ValueError("Could not locate a valid constituents table on the source page.")


def parse_sp500_sector_tickers(sector_name: str) -> List[str]:
    response = requests.get(INDEX_SOURCE_URLS["S&P 500"], headers=HTTP_HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    target_sector = re.sub(r"\s+", " ", sector_name).strip().casefold()

    for candidate in soup.find_all("table", {"class": "wikitable"}):
        rows = candidate.find_all("tr")
        if not rows:
            continue

        header = [th.get_text(" ", strip=True).lower() for th in rows[0].find_all(["th", "td"])]
        ticker_idx = next((i for i, name in enumerate(header) if "symbol" in name or "ticker" in name), None)
        sector_idx = next((i for i, name in enumerate(header) if "gics sector" in name), None)
        if ticker_idx is None or sector_idx is None:
            continue

        tickers: List[str] = []
        for row in rows[1:]:
            cells = row.find_all("td")
            if not cells or len(cells) <= max(ticker_idx, sector_idx):
                continue
            sector_value = re.sub(r"\s+", " ", cells[sector_idx].get_text(" ", strip=True)).strip().casefold()
            if sector_value == target_sector:
                tickers.append(cells[ticker_idx].get_text(strip=True))

        normalized = normalize_ticker_list(tickers)
        if normalized:
            return normalized

    raise ValueError(f"Could not find S&P 500 sector constituents for: {sector_name}")


def parse_vanguard_stock_holdings_tickers(fund_identifier: str) -> List[str]:
    url = VANGUARD_HOLDINGS_API_TEMPLATE.format(fund_identifier=fund_identifier)
    response = requests.get(url, headers=HTTP_HEADERS, params={"start": 1, "count": 20000}, timeout=30)
    response.raise_for_status()

    payload = response.json()
    entities = payload.get("fund", {}).get("entity", [])
    tickers = [str(item.get("ticker", "")).strip() for item in entities if isinstance(item, dict)]
    normalized = normalize_ticker_list(tickers)
    if not normalized:
        raise ValueError(f"No equity tickers returned from Vanguard holdings API for {fund_identifier}.")
    return normalized


@st.cache_data(show_spinner=False)
def load_index_universe(index_name: str) -> List[str]:
    if index_name in {"S&P 500", "Large Cap (S&P 500)"}:
        return parse_wikipedia_ticker_table(INDEX_SOURCE_URLS["S&P 500"], required_header_keywords=("symbol", "security", "gics"))
    if index_name == "Mid Cap (S&P 400)":
        return parse_wikipedia_ticker_table(INDEX_SOURCE_URLS["S&P 400"], required_header_keywords=("symbol", "security"))
    if index_name == "Small Cap (S&P 600)":
        return parse_wikipedia_ticker_table(INDEX_SOURCE_URLS["S&P 600"], required_header_keywords=("symbol", "security"))
    if index_name == "CRSP US Total Market":
        return parse_vanguard_stock_holdings_tickers("VTI")
    if index_name == "Russell 3000":
        return parse_ishares_holdings_tickers(INDEX_SOURCE_URLS["Russell 3000"])
    if index_name == "Russell 2000":
        return parse_ishares_holdings_tickers(INDEX_SOURCE_URLS["Russell 2000"])
    if index_name == "Russell Microcap":
        return parse_ishares_holdings_tickers(INDEX_SOURCE_URLS["Russell Microcap"])
    if index_name == "NASDAQ-100":
        return parse_nasdaq_100_tickers()
    if index_name in SP500_SECTOR_UNIVERSE_MAP:
        return parse_sp500_sector_tickers(SP500_SECTOR_UNIVERSE_MAP[index_name])
    raise ValueError(f"Unknown index universe: {index_name}")


def normalize_long_only(weights: pd.DataFrame) -> pd.DataFrame:
    weights = weights.clip(lower=0.0)
    row_sum = weights.sum(axis=1).replace(0.0, np.nan)
    normalized = weights.div(row_sum, axis=0)
    return normalized.fillna(0.0)


def normalize_gross_exposure(weights: pd.DataFrame) -> pd.DataFrame:
    clean = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = clean.abs().sum(axis=1).replace(0.0, np.nan)
    normalized = clean.div(gross, axis=0)
    return normalized.fillna(0.0)


def enforce_tradeable_weights(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    clean = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    clean = clean.where(prices > 0, 0.0)
    return normalize_long_only(clean)


def validate_position_side(position_side: str) -> None:
    if position_side not in POSITION_SIDE_OPTIONS:
        raise ValueError(
            f"Invalid position side: {position_side}. "
            f"Valid values: {', '.join(POSITION_SIDE_OPTIONS)}"
        )


def validate_weighting_scheme(weighting_scheme: str) -> None:
    if weighting_scheme not in WEIGHTING_SCHEME_OPTIONS:
        raise ValueError(
            f"Invalid weighting scheme: {weighting_scheme}. "
            f"Valid values: {', '.join(WEIGHTING_SCHEME_OPTIONS)}"
        )


def get_execution_style_from_weighting(weighting_scheme: str) -> str:
    return EXECUTION_STYLE_HOLD if weighting_scheme == "Hold Until Sell (No Rebalance)" else EXECUTION_STYLE_TARGET


def compute_weight_strength(
    raw_strength: pd.Series,
    weighting_scheme: str,
    vol_row: pd.Series | None,
) -> pd.Series:
    positive = raw_strength.clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if weighting_scheme == "Equal Weight":
        return (positive > 0).astype(float)
    if weighting_scheme == "Signal Strength":
        return positive
    if weighting_scheme == "Signal Strength / Volatility":
        if vol_row is None:
            return positive
        safe_vol = vol_row.replace(0.0, np.nan)
        return positive.div(safe_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if weighting_scheme == "Hold Until Sell (No Rebalance)":
        return (positive > 0).astype(float)
    raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")


def construct_portfolio_weights(
    base_long_weights: pd.DataFrame,
    signal_scores: pd.DataFrame,
    prices: pd.DataFrame,
    position_side: str,
    weighting_scheme: str,
    max_holdings: int,
    weighting_vol_window: int = 20,
) -> pd.DataFrame:
    if max_holdings < 1:
        raise ValueError("max_holdings must be at least 1.")
    if weighting_vol_window < 2:
        raise ValueError("weighting_vol_window must be at least 2.")
    validate_position_side(position_side)
    validate_weighting_scheme(weighting_scheme)
    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    index = prices.index
    columns = prices.columns
    tradeable_mask = prices.notna() & (prices > 0)

    base = base_long_weights.reindex(index=index, columns=columns).fillna(0.0)
    base = base.where(tradeable_mask, 0.0)

    scores = signal_scores.reindex(index=index, columns=columns)
    scores = scores.where(tradeable_mask).replace([np.inf, -np.inf], np.nan)

    trailing_vol: pd.DataFrame | None = None
    if weighting_scheme == "Signal Strength / Volatility":
        trailing_vol = prices.pct_change(fill_method=None).rolling(int(weighting_vol_window)).std()
        trailing_vol = trailing_vol.reindex(index=index, columns=columns)

    constructed = pd.DataFrame(0.0, index=index, columns=columns)
    eps = 1e-12

    for dt in index:
        valid = tradeable_mask.loc[dt]
        if not bool(valid.any()):
            continue

        base_row = base.loc[dt]
        score_row = scores.loc[dt].fillna(0.0)
        signed_score_row = score_row.copy()
        has_pos = bool((signed_score_row > eps).any())
        has_neg = bool((signed_score_row < -eps).any())
        if not (has_pos and has_neg):
            centered = signed_score_row - float(signed_score_row.median())
            if bool((centered.abs() > eps).any()):
                signed_score_row = centered
        vol_row = trailing_vol.loc[dt] if trailing_vol is not None else None
        row = pd.Series(0.0, index=columns, dtype=float)

        if position_side == "Long Only":
            long_candidates = base_row > 0
            if not bool(long_candidates.any()):
                constructed.loc[dt] = row
                continue

            long_raw = score_row.where(long_candidates, 0.0).clip(lower=0.0)
            candidate_rank = long_raw.copy()
            if float(candidate_rank.sum()) <= eps:
                candidate_rank = base_row.where(long_candidates, 0.0).fillna(0.0)

            candidate_idx = long_candidates[long_candidates].index
            if len(candidate_idx) > int(max_holdings):
                selected_idx = candidate_rank.nlargest(int(max_holdings)).index
            else:
                selected_idx = candidate_idx

            selected_mask = pd.Series(False, index=columns)
            selected_mask.loc[selected_idx] = True
            selected_raw = long_raw.where(selected_mask, 0.0)
            long_strength = compute_weight_strength(selected_raw, weighting_scheme, vol_row)
            if float(long_strength.sum()) <= eps:
                long_strength = base_row.where(selected_mask, 0.0).fillna(0.0)
            long_strength = long_strength[long_strength > eps]
            if long_strength.empty:
                constructed.loc[dt] = row
                continue

            row.loc[long_strength.index] = (long_strength / long_strength.sum()).values
            constructed.loc[dt] = row
            continue

        if position_side == "Short Only":
            short_raw = (-signed_score_row).clip(lower=0.0)
            short_candidates = short_raw > eps
            if not bool(short_candidates.any()):
                constructed.loc[dt] = row
                continue

            candidate_idx = short_candidates[short_candidates].index
            if len(candidate_idx) > int(max_holdings):
                selected_idx = short_raw.nlargest(int(max_holdings)).index
            else:
                selected_idx = candidate_idx

            selected_mask = pd.Series(False, index=columns)
            selected_mask.loc[selected_idx] = True
            selected_raw = short_raw.where(selected_mask, 0.0)
            short_strength = compute_weight_strength(selected_raw, weighting_scheme, vol_row)
            if float(short_strength.sum()) <= eps:
                short_strength = selected_raw
            short_strength = short_strength[short_strength > eps]
            if short_strength.empty:
                constructed.loc[dt] = row
                continue

            row.loc[short_strength.index] = -(short_strength / short_strength.sum()).values
            constructed.loc[dt] = row
            continue

        # Long/Short: rank by absolute signal strength, then allocate gross 50/50 by side.
        long_strength = compute_weight_strength(signed_score_row.clip(lower=0.0), weighting_scheme, vol_row)
        short_strength = compute_weight_strength((-signed_score_row).clip(lower=0.0), weighting_scheme, vol_row)
        signed_strength = long_strength - short_strength
        active = signed_strength[abs(signed_strength) > eps]
        if active.empty:
            constructed.loc[dt] = row
            continue

        if len(active) > int(max_holdings):
            keep_idx = active.abs().nlargest(int(max_holdings)).index
            signed_strength = signed_strength.where(signed_strength.index.isin(keep_idx), 0.0)

        selected_longs = signed_strength[signed_strength > eps]
        selected_shorts = -signed_strength[signed_strength < -eps]
        long_sum = float(selected_longs.sum())
        short_sum = float(selected_shorts.sum())
        if long_sum <= eps and short_sum <= eps:
            constructed.loc[dt] = row
            continue

        if long_sum > eps and short_sum > eps:
            long_budget = 0.5
            short_budget = 0.5
        elif long_sum > eps:
            long_budget = 1.0
            short_budget = 0.0
        else:
            long_budget = 0.0
            short_budget = 1.0

        if long_sum > eps:
            row.loc[selected_longs.index] = (selected_longs / long_sum * long_budget).values
        if short_sum > eps:
            row.loc[selected_shorts.index] = -(selected_shorts / short_sum * short_budget).values
        constructed.loc[dt] = row

    constructed = constructed.where(tradeable_mask, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = constructed.abs().sum(axis=1)
    over = gross > 1.0 + 1e-9
    if bool(over.any()):
        constructed.loc[over] = constructed.loc[over].div(gross.loc[over], axis=0)
    return constructed.fillna(0.0)


def validate_rebalance_frequency(frequency: str) -> None:
    if frequency not in VALID_REBALANCE_FREQUENCIES:
        raise ValueError(
            f"Invalid rebalance frequency: {frequency}. "
            f"Valid values: {', '.join(sorted(VALID_REBALANCE_FREQUENCIES))}"
        )


def get_rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> List[pd.Timestamp]:
    validate_rebalance_frequency(frequency)
    if frequency == "D":
        return list(index)

    marker = pd.Series(1, index=index)
    rebalance_dates: List[pd.Timestamp] = []
    for _, group in marker.groupby(pd.Grouper(freq=frequency)):
        if not group.empty:
            rebalance_dates.append(group.index[-1])
    return rebalance_dates


def init_price_database() -> None:
    PRICE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PRICE_DB_PATH, timeout=30) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            )
            """
        )
        columns = {str(row[1]).lower() for row in conn.execute("PRAGMA table_info(daily_prices)").fetchall()}
        if "volume" not in columns:
            conn.execute("ALTER TABLE daily_prices ADD COLUMN volume REAL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices (date)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fundamentals_snapshot (
                ticker TEXT PRIMARY KEY,
                asof_date TEXT NOT NULL,
                trailing_pe REAL,
                forward_pe REAL,
                price_to_book REAL,
                trailing_eps REAL,
                forward_eps REAL,
                quote_type TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamentals_asof_date ON fundamentals_snapshot (asof_date)")
        conn.commit()


def get_price_database_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(PRICE_DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def download_market_data_from_yfinance(tickers: Tuple[str, ...], start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

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
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy() if "Close" in raw.columns.get_level_values(0) else pd.DataFrame(index=raw.index)
        volume = raw["Volume"].copy() if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame(index=raw.index)
    else:
        close = raw["Close"].to_frame(name=tickers[0]) if "Close" in raw.columns else pd.DataFrame(index=raw.index, columns=[tickers[0]])
        volume = raw["Volume"].to_frame(name=tickers[0]) if "Volume" in raw.columns else pd.DataFrame(index=raw.index, columns=[tickers[0]])

    close.columns = [str(col).upper() for col in close.columns]
    volume.columns = [str(col).upper() for col in volume.columns]
    close = close.sort_index().dropna(how="all")
    volume = volume.sort_index().dropna(how="all")
    return close, volume


def download_prices_from_yfinance(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    close, _ = download_market_data_from_yfinance(tickers, start, end)
    return close


def download_volumes_from_yfinance(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    _, volume = download_market_data_from_yfinance(tickers, start, end)
    return volume


def fetch_fundamentals_from_yfinance(tickers: Tuple[str, ...]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(
            columns=["ticker", "asof_date", "trailing_pe", "forward_pe", "price_to_book", "trailing_eps", "forward_eps", "quote_type"]
        )

    asof_date = pd.Timestamp.utcnow().tz_localize(None).normalize().strftime("%Y-%m-%d")
    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        info: Dict[str, Any] = {}
        try:
            info = yf.Ticker(ticker).get_info() or {}
        except Exception:
            info = {}

        rows.append(
            {
                "ticker": str(ticker).upper(),
                "asof_date": asof_date,
                "trailing_pe": clean_positive_ratio(info.get("trailingPE")),
                "forward_pe": clean_positive_ratio(info.get("forwardPE")),
                "price_to_book": clean_positive_ratio(info.get("priceToBook")),
                "trailing_eps": to_finite_float(info.get("trailingEps", info.get("epsTrailingTwelveMonths"))),
                "forward_eps": to_finite_float(info.get("forwardEps", info.get("epsForward"))),
                "quote_type": str(info.get("quoteType", "")).strip().upper(),
            }
        )

    return pd.DataFrame(rows)


def read_fundamentals_from_database(conn: sqlite3.Connection, tickers: Tuple[str, ...]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["asof_date", *FUNDAMENTAL_METRIC_ORDER, "quote_type"])

    placeholders = ",".join(["?"] * len(tickers))
    query = (
        f"SELECT ticker, asof_date, trailing_pe, forward_pe, price_to_book, trailing_eps, forward_eps, quote_type "
        f"FROM fundamentals_snapshot WHERE ticker IN ({placeholders})"
    )
    rows = pd.read_sql_query(query, conn, params=list(tickers))
    if rows.empty:
        return pd.DataFrame(columns=["asof_date", *FUNDAMENTAL_METRIC_ORDER, "quote_type"], index=pd.Index([], name="ticker"))

    rows["ticker"] = rows["ticker"].astype(str).str.upper()
    rows["asof_date"] = pd.to_datetime(rows["asof_date"], errors="coerce")
    for metric in FUNDAMENTAL_METRIC_ORDER:
        if metric in rows.columns:
            rows[metric] = sanitize_fundamental_metric_series(rows[metric], metric)
    rows = rows.set_index("ticker").sort_index()
    return rows.reindex(list(tickers))


def upsert_fundamentals_into_database(conn: sqlite3.Connection, fundamentals: pd.DataFrame) -> int:
    if fundamentals.empty:
        return 0

    records: List[Tuple[str, str, float | None, float | None, float | None, float | None, float | None, str | None]] = []
    for _, row in fundamentals.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        asof_raw = row.get("asof_date", pd.Timestamp.utcnow().tz_localize(None).normalize())
        asof_date = pd.Timestamp(asof_raw).strftime("%Y-%m-%d")
        trailing_pe = to_finite_float(row.get("trailing_pe"))
        forward_pe = to_finite_float(row.get("forward_pe"))
        price_to_book = to_finite_float(row.get("price_to_book"))
        trailing_eps = to_finite_float(row.get("trailing_eps"))
        forward_eps = to_finite_float(row.get("forward_eps"))
        quote_type = str(row.get("quote_type", "")).strip().upper() or None
        records.append(
            (
                ticker,
                asof_date,
                float(trailing_pe) if pd.notna(trailing_pe) else None,
                float(forward_pe) if pd.notna(forward_pe) else None,
                float(price_to_book) if pd.notna(price_to_book) else None,
                float(trailing_eps) if pd.notna(trailing_eps) else None,
                float(forward_eps) if pd.notna(forward_eps) else None,
                quote_type,
            )
        )

    if not records:
        return 0

    conn.executemany(
        """
        INSERT INTO fundamentals_snapshot (
            ticker, asof_date, trailing_pe, forward_pe, price_to_book, trailing_eps, forward_eps, quote_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            asof_date = excluded.asof_date,
            trailing_pe = excluded.trailing_pe,
            forward_pe = excluded.forward_pe,
            price_to_book = excluded.price_to_book,
            trailing_eps = excluded.trailing_eps,
            forward_eps = excluded.forward_eps,
            quote_type = excluded.quote_type
        """,
        records,
    )
    conn.commit()
    return len(records)


def download_fundamental_data(tickers: Tuple[str, ...], refresh_days: int = FUNDAMENTAL_REFRESH_DAYS) -> pd.DataFrame:
    normalized_tickers = tuple(normalize_ticker_list(list(tickers)))
    if not normalized_tickers:
        return pd.DataFrame(columns=["asof_date", *FUNDAMENTAL_METRIC_ORDER, "quote_type"])

    refresh_days = max(1, int(refresh_days))
    stale_cutoff = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=refresh_days)
    init_price_database()

    with get_price_database_connection() as conn:
        db_fundamentals = read_fundamentals_from_database(conn, normalized_tickers)
        stale_tickers: List[str] = []
        for ticker in normalized_tickers:
            if ticker not in db_fundamentals.index:
                stale_tickers.append(ticker)
                continue
            asof_value = db_fundamentals.at[ticker, "asof_date"] if "asof_date" in db_fundamentals.columns else pd.NaT
            if pd.isna(asof_value) or pd.Timestamp(asof_value) < stale_cutoff:
                stale_tickers.append(ticker)

        if stale_tickers:
            fetched = fetch_fundamentals_from_yfinance(tuple(stale_tickers))
            if not fetched.empty:
                upsert_fundamentals_into_database(conn, fetched)
            db_fundamentals = read_fundamentals_from_database(conn, normalized_tickers)

    if db_fundamentals.empty:
        empty = pd.DataFrame(index=list(normalized_tickers), columns=["asof_date", *FUNDAMENTAL_METRIC_ORDER, "quote_type"])
        empty.index.name = "ticker"
        return empty
    return db_fundamentals.reindex(list(normalized_tickers))


def filter_stocks_by_fundamentals(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    metric_filters: Dict[str, Dict[str, float]],
    min_coverage: float,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, float], List[str], bool]:
    if prices.empty or not metric_filters:
        return prices, [], [], {}, [], False

    min_coverage = float(np.clip(float(min_coverage), 0.0, 1.0))
    tickers = list(prices.columns)
    aligned_fundamentals = fundamentals.reindex(tickers)

    metric_coverage: Dict[str, float] = {}
    low_coverage_metrics: List[str] = []
    denominator = max(1, len(tickers))

    for metric in metric_filters:
        if metric not in aligned_fundamentals.columns:
            metric_coverage[metric] = 0.0
            low_coverage_metrics.append(metric)
            continue
        sanitized = sanitize_fundamental_metric_series(aligned_fundamentals[metric], metric)
        coverage = float(sanitized.notna().sum()) / float(denominator)
        metric_coverage[metric] = coverage
        if coverage < min_coverage:
            low_coverage_metrics.append(metric)

    if low_coverage_metrics:
        return prices, [], [], metric_coverage, sorted(low_coverage_metrics), False

    pass_mask = pd.Series(True, index=tickers, dtype=bool)
    missing_mask = pd.Series(False, index=tickers, dtype=bool)
    for metric, bounds in metric_filters.items():
        values = sanitize_fundamental_metric_series(aligned_fundamentals[metric], metric).reindex(tickers)
        metric_pass = values.notna()
        missing_mask = missing_mask | values.isna()
        min_value = bounds.get("min")
        max_value = bounds.get("max")
        if min_value is not None:
            metric_pass = metric_pass & (values >= float(min_value))
        if max_value is not None:
            metric_pass = metric_pass & (values <= float(max_value))
        pass_mask = pass_mask & metric_pass.fillna(False)

    filtered = prices.loc[:, pass_mask]
    excluded_tickers = sorted(pass_mask.index[~pass_mask].tolist())
    missing_tickers = sorted(missing_mask.index[missing_mask].tolist())
    return filtered, excluded_tickers, missing_tickers, metric_coverage, [], True


def filter_illiquid_stocks(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    min_median_dollar_volume: float,
    min_median_share_volume: float,
    lookback_days: int,
) -> Tuple[pd.DataFrame, List[str], List[str], bool]:
    if prices.empty:
        return prices, [], [], False

    lookback_days = max(5, int(lookback_days))
    min_median_dollar_volume = max(0.0, float(min_median_dollar_volume))
    min_median_share_volume = max(0.0, float(min_median_share_volume))

    volumes = volumes.reindex(index=prices.index, columns=prices.columns)
    if volumes.empty or int(volumes.notna().sum().sum()) == 0:
        return prices, [], sorted(prices.columns.tolist()), False

    window = min(len(prices.index), lookback_days)
    recent_index = prices.index[-window:]
    recent_prices = prices.loc[recent_index].where(prices.loc[recent_index] > 0)
    recent_volumes = volumes.loc[recent_index]
    recent_dollar_volume = recent_prices * recent_volumes

    median_dollar_volume = recent_dollar_volume.median(axis=0, skipna=True)
    median_share_volume = recent_volumes.median(axis=0, skipna=True)
    no_volume_tickers = sorted(median_share_volume[median_share_volume.isna()].index.tolist())
    dollar_pass = (median_dollar_volume >= min_median_dollar_volume).fillna(False)
    share_pass = (median_share_volume >= min_median_share_volume).fillna(False)
    liquid_mask = (dollar_pass | share_pass).fillna(False)
    illiquid_mask = (~liquid_mask) & median_share_volume.notna()
    illiquid_tickers = sorted(median_share_volume[illiquid_mask].index.tolist())

    filtered_prices = prices.loc[:, liquid_mask]
    return filtered_prices, illiquid_tickers, no_volume_tickers, True


def get_data_bounds_from_database(conn: sqlite3.Connection, tickers: Tuple[str, ...], field: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "min_date", "max_date"])
    if field not in {"close", "volume"}:
        raise ValueError(f"Unsupported data field: {field}")

    placeholders = ",".join(["?"] * len(tickers))
    query = (
        f"SELECT ticker, MIN(date) AS min_date, MAX(date) AS max_date "
        f"FROM daily_prices WHERE ticker IN ({placeholders}) AND {field} IS NOT NULL GROUP BY ticker"
    )
    bounds = pd.read_sql_query(query, conn, params=list(tickers))
    if bounds.empty:
        return bounds

    bounds["ticker"] = bounds["ticker"].astype(str).str.upper()
    bounds["min_date"] = pd.to_datetime(bounds["min_date"])
    bounds["max_date"] = pd.to_datetime(bounds["max_date"])
    return bounds


def get_price_bounds_from_database(conn: sqlite3.Connection, tickers: Tuple[str, ...]) -> pd.DataFrame:
    return get_data_bounds_from_database(conn, tickers, field="close")


def read_prices_from_database(
    conn: sqlite3.Connection,
    tickers: Tuple[str, ...],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(tickers))
    query = (
        f"SELECT date, ticker, close FROM daily_prices "
        f"WHERE ticker IN ({placeholders}) AND date BETWEEN ? AND ? "
        f"ORDER BY date"
    )
    params: List[str] = list(tickers) + [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
    rows = pd.read_sql_query(query, conn, params=params)
    if rows.empty:
        return pd.DataFrame(columns=list(tickers))

    rows["date"] = pd.to_datetime(rows["date"])
    prices = rows.pivot(index="date", columns="ticker", values="close").sort_index()
    return prices.reindex(columns=list(tickers))


def read_volumes_from_database(
    conn: sqlite3.Connection,
    tickers: Tuple[str, ...],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(tickers))
    query = (
        f"SELECT date, ticker, volume FROM daily_prices "
        f"WHERE ticker IN ({placeholders}) AND date BETWEEN ? AND ? "
        f"ORDER BY date"
    )
    params: List[str] = list(tickers) + [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
    rows = pd.read_sql_query(query, conn, params=params)
    if rows.empty:
        return pd.DataFrame(columns=list(tickers))

    rows["date"] = pd.to_datetime(rows["date"])
    volumes = rows.pivot(index="date", columns="ticker", values="volume").sort_index()
    return volumes.reindex(columns=list(tickers))


def apply_delisting_zero_assumption(prices: pd.DataFrame, confirmation_days: int = DELISTING_CONFIRMATION_DAYS) -> pd.DataFrame:
    if prices.empty:
        return prices
    if confirmation_days < 1:
        raise ValueError("confirmation_days must be at least 1.")

    adjusted = prices.copy().sort_index()
    for ticker in adjusted.columns:
        series = adjusted[ticker]
        valid_idx = series[series.notna()].index
        if len(valid_idx) == 0:
            continue

        first_valid = valid_idx[0]
        last_valid = valid_idx[-1]

        between_mask = (adjusted.index >= first_valid) & (adjusted.index <= last_valid)
        adjusted.loc[between_mask, ticker] = adjusted.loc[between_mask, ticker].ffill()

        after_idx = adjusted.index[adjusted.index > last_valid]
        # Treat long missing tails as potential delistings; keep short tails as missing data.
        if len(after_idx) >= confirmation_days:
            adjusted.loc[after_idx, ticker] = 0.0

    return adjusted


def upsert_prices_into_database(conn: sqlite3.Connection, prices: pd.DataFrame, volumes: pd.DataFrame | None = None) -> int:
    if prices.empty:
        return 0

    if volumes is None:
        volumes = pd.DataFrame(index=prices.index, columns=prices.columns)
    else:
        volumes = volumes.reindex(index=prices.index, columns=prices.columns)

    stacked_close = prices.stack(future_stack=True).reset_index()
    stacked_close.columns = ["date", "ticker", "close"]
    stacked_close = stacked_close.dropna(subset=["close"])

    records: List[Tuple[str, str, float, float | None]] = []
    for dt, ticker, close in stacked_close.itertuples(index=False, name=None):
        volume_value = volumes.at[dt, ticker] if ticker in volumes.columns and dt in volumes.index else np.nan
        volume_db_value = float(volume_value) if pd.notna(volume_value) else None
        records.append((str(ticker).upper(), pd.Timestamp(dt).strftime("%Y-%m-%d"), float(close), volume_db_value))

    conn.executemany(
        """
        INSERT INTO daily_prices (ticker, date, close, volume)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            close = excluded.close,
            volume = COALESCE(excluded.volume, daily_prices.volume)
        """,
        records,
    )
    conn.commit()
    return len(records)


def compute_missing_data_window(
    ticker: str,
    bound_map: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    coverage_tolerance: pd.Timedelta,
    refresh_lookback: pd.Timedelta,
) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if ticker not in bound_map:
        return start_date, end_date

    min_date, max_date = bound_map[ticker]
    download_start: pd.Timestamp | None = None
    download_end: pd.Timestamp | None = None

    if start_date < (min_date - coverage_tolerance):
        missing_prefix_end = min(end_date, min_date - pd.Timedelta(days=1))
        if start_date <= missing_prefix_end:
            download_start = start_date
            download_end = missing_prefix_end

    if end_date > (max_date + coverage_tolerance):
        missing_suffix_start = max(start_date, max_date - refresh_lookback)
        if missing_suffix_start <= end_date:
            download_start = missing_suffix_start if download_start is None else min(download_start, missing_suffix_start)
            download_end = end_date if download_end is None else max(download_end, end_date)

    return download_start, download_end


def download_price_volume_data(tickers: Tuple[str, ...], start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normalized_tickers = tuple(normalize_ticker_list(list(tickers)))
    if not normalized_tickers:
        return pd.DataFrame(), pd.DataFrame()

    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if start_date > end_date:
        return pd.DataFrame(), pd.DataFrame()

    init_price_database()
    coverage_tolerance = pd.Timedelta(days=DB_COVERAGE_TOLERANCE_DAYS)
    refresh_lookback = pd.Timedelta(days=DB_REFRESH_LOOKBACK_DAYS)

    with get_price_database_connection() as conn:
        close_bounds = get_data_bounds_from_database(conn, normalized_tickers, field="close")
        close_bound_map: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {
            str(row["ticker"]).upper(): (pd.Timestamp(row["min_date"]), pd.Timestamp(row["max_date"]))
            for _, row in close_bounds.iterrows()
        }
        volume_bounds = get_data_bounds_from_database(conn, normalized_tickers, field="volume")
        volume_bound_map: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {
            str(row["ticker"]).upper(): (pd.Timestamp(row["min_date"]), pd.Timestamp(row["max_date"]))
            for _, row in volume_bounds.iterrows()
        }

        tickers_to_update: List[str] = []
        update_starts: List[pd.Timestamp] = []
        update_ends: List[pd.Timestamp] = []

        for ticker in normalized_tickers:
            close_start, close_end = compute_missing_data_window(
                ticker=ticker,
                bound_map=close_bound_map,
                start_date=start_date,
                end_date=end_date,
                coverage_tolerance=coverage_tolerance,
                refresh_lookback=refresh_lookback,
            )
            volume_start, volume_end = compute_missing_data_window(
                ticker=ticker,
                bound_map=volume_bound_map,
                start_date=start_date,
                end_date=end_date,
                coverage_tolerance=coverage_tolerance,
                refresh_lookback=refresh_lookback,
            )

            windows = [(s, e) for s, e in [(close_start, close_end), (volume_start, volume_end)] if s is not None and e is not None]
            if windows:
                tickers_to_update.append(ticker)
                update_starts.append(min(s for s, _ in windows))
                update_ends.append(max(e for _, e in windows))

        if tickers_to_update:
            fetch_start = min(update_starts).strftime("%Y-%m-%d")
            fetch_end = max(update_ends).strftime("%Y-%m-%d")
            fetched_prices, fetched_volumes = download_market_data_from_yfinance(tuple(sorted(set(tickers_to_update))), fetch_start, fetch_end)
            if not fetched_prices.empty:
                upsert_prices_into_database(conn, fetched_prices, volumes=fetched_volumes)

        db_prices = read_prices_from_database(conn, normalized_tickers, start_date, end_date)
        db_volumes = read_volumes_from_database(conn, normalized_tickers, start_date, end_date)

        # Safety pass: if any ticker is still fully missing in-range for either field, retry once.
        missing_in_range = [
            ticker
            for ticker in normalized_tickers
            if (ticker not in db_prices.columns or db_prices[ticker].dropna().empty)
            or (ticker not in db_volumes.columns or db_volumes[ticker].dropna().empty)
        ]
        if missing_in_range:
            retry_prices, retry_volumes = download_market_data_from_yfinance(tuple(sorted(set(missing_in_range))), start, end)
            if not retry_prices.empty:
                upsert_prices_into_database(conn, retry_prices, volumes=retry_volumes)
                db_prices = read_prices_from_database(conn, normalized_tickers, start_date, end_date)
                db_volumes = read_volumes_from_database(conn, normalized_tickers, start_date, end_date)

    if db_prices.empty:
        fallback, fallback_volumes = download_market_data_from_yfinance(normalized_tickers, start, end)
        if not fallback.empty:
            with get_price_database_connection() as conn:
                upsert_prices_into_database(conn, fallback, volumes=fallback_volumes)
        fallback = fallback.sort_index().dropna(how="all")
        fallback_volumes = fallback_volumes.sort_index().dropna(how="all")
        return apply_delisting_zero_assumption(fallback), fallback_volumes

    db_prices = db_prices.sort_index().dropna(how="all")
    db_volumes = db_volumes.sort_index().dropna(how="all")
    return apply_delisting_zero_assumption(db_prices), db_volumes


def download_prices(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    prices, _ = download_price_volume_data(tickers, start, end)
    return prices


def download_volumes(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    _, volumes = download_price_volume_data(tickers, start, end)
    return volumes


def equal_weight_positions(prices: pd.DataFrame) -> pd.DataFrame:
    available = prices.notna().astype(float)
    return normalize_long_only(available)


def apply_entry_exit_logic(
    prices: pd.DataFrame,
    entry_signal: pd.DataFrame,
    exit_signal: pd.DataFrame,
    exit_mode: str,
    max_hold_days: int | None = None,
    stop_loss_pct: float = 0.0,
) -> pd.DataFrame:
    if exit_mode not in EXIT_MODE_OPTIONS:
        raise ValueError(f"Invalid exit mode: {exit_mode}")
    if exit_mode in {"Fixed Days", "Hybrid"} and (max_hold_days is None or max_hold_days < 1):
        raise ValueError("max_hold_days must be at least 1 for Fixed Days or Hybrid exits.")
    if stop_loss_pct < 0:
        raise ValueError("stop_loss_pct cannot be negative.")

    entry_signal = entry_signal.fillna(False).astype(bool)
    exit_signal = exit_signal.fillna(False).astype(bool)
    position = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for ticker in prices.columns:
        in_position = False
        days_in_trade = 0
        entry_price = np.nan

        for idx in prices.index:
            px = prices.at[idx, ticker]
            if pd.isna(px):
                in_position = False
                days_in_trade = 0
                entry_price = np.nan
                position.at[idx, ticker] = 0.0
                continue

            if not in_position:
                if entry_signal.at[idx, ticker]:
                    in_position = True
                    days_in_trade = 0
                    entry_price = float(px)
            else:
                days_in_trade += 1
                hit_signal_exit = exit_mode in {"Signal-Based", "Hybrid"} and exit_signal.at[idx, ticker]
                hit_time_exit = exit_mode in {"Fixed Days", "Hybrid"} and days_in_trade >= int(max_hold_days)
                hit_stop_loss = False

                if stop_loss_pct > 0 and pd.notna(entry_price):
                    hit_stop_loss = (float(px) / float(entry_price) - 1.0) <= -float(stop_loss_pct)

                if hit_signal_exit or hit_time_exit or hit_stop_loss:
                    in_position = False
                    days_in_trade = 0
                    entry_price = np.nan

            position.at[idx, ticker] = 1.0 if in_position else 0.0

    return normalize_long_only(position.where(prices.notna(), 0.0))


def moving_average_crossover(
    prices: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    band_pct: float,
    use_exponential: bool,
) -> pd.DataFrame:
    if fast_window >= slow_window:
        raise ValueError("Fast window must be smaller than slow window.")
    if band_pct < 0:
        raise ValueError("Band percentage cannot be negative.")

    if use_exponential:
        fast_ma = prices.ewm(span=fast_window, adjust=False).mean()
        slow_ma = prices.ewm(span=slow_window, adjust=False).mean()
    else:
        fast_ma = prices.rolling(fast_window).mean()
        slow_ma = prices.rolling(slow_window).mean()

    upper_band = slow_ma * (1.0 + band_pct / 100.0)
    lower_band = slow_ma * (1.0 - band_pct / 100.0)

    entry_signal = fast_ma > upper_band
    exit_signal = fast_ma < lower_band
    return apply_entry_exit_logic(
        prices=prices,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        exit_mode="Signal-Based",
    )


def sma_crossover(prices: pd.DataFrame, fast_window: int, slow_window: int, band_pct: float) -> pd.DataFrame:
    return moving_average_crossover(
        prices=prices,
        fast_window=fast_window,
        slow_window=slow_window,
        band_pct=band_pct,
        use_exponential=False,
    )


def ema_crossover(prices: pd.DataFrame, fast_window: int, slow_window: int, band_pct: float) -> pd.DataFrame:
    return moving_average_crossover(
        prices=prices,
        fast_window=fast_window,
        slow_window=slow_window,
        band_pct=band_pct,
        use_exponential=True,
    )


def macd_trend(
    prices: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    signal_window: int,
    hist_entry: float,
    hist_exit: float,
) -> pd.DataFrame:
    if fast_window >= slow_window:
        raise ValueError("MACD fast window must be smaller than slow window.")
    if signal_window < 2:
        raise ValueError("MACD signal window must be at least 2.")
    if hist_exit > hist_entry:
        raise ValueError("MACD exit histogram threshold should be <= entry threshold.")

    fast_ema = prices.ewm(span=fast_window, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_window, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    hist = macd_line - signal_line

    entry_signal = hist >= hist_entry
    exit_signal = hist <= hist_exit
    return apply_entry_exit_logic(
        prices=prices,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        exit_mode="Signal-Based",
    )


def moving_average_reversion(
    prices: pd.DataFrame,
    window: int,
    entry_deviation_pct: float,
    exit_mode: str,
    exit_deviation_pct: float,
    max_hold_days: int | None,
    stop_loss_pct: float,
) -> pd.DataFrame:
    if entry_deviation_pct <= 0:
        raise ValueError("Entry deviation must be positive.")
    if exit_mode in {"Signal-Based", "Hybrid"} and exit_deviation_pct > entry_deviation_pct:
        raise ValueError("Exit deviation should be less than or equal to entry deviation.")

    mean = prices.rolling(window).mean()
    deviation = prices.div(mean) - 1.0
    entry_signal = deviation <= -(entry_deviation_pct / 100.0)
    exit_signal = deviation >= -(exit_deviation_pct / 100.0)
    return apply_entry_exit_logic(
        prices=prices,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        exit_mode=exit_mode,
        max_hold_days=max_hold_days,
        stop_loss_pct=stop_loss_pct,
    )


def compute_rsi(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain.div(avg_loss.replace(0.0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss > 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def rsi_mean_reversion(
    prices: pd.DataFrame,
    window: int,
    oversold: float,
    exit_mode: str,
    exit_rsi: float,
    max_hold_days: int | None,
    stop_loss_pct: float,
) -> pd.DataFrame:
    if exit_mode in {"Signal-Based", "Hybrid"} and exit_rsi <= oversold:
        raise ValueError("RSI exit threshold should be higher than oversold entry threshold.")

    rsi = compute_rsi(prices, window)
    entry_signal = rsi < oversold
    exit_signal = rsi >= exit_rsi
    return apply_entry_exit_logic(
        prices=prices,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        exit_mode=exit_mode,
        max_hold_days=max_hold_days,
        stop_loss_pct=stop_loss_pct,
    )


def bollinger_mean_reversion(
    prices: pd.DataFrame,
    window: int,
    z_entry: float,
    exit_mode: str,
    z_exit: float,
    max_hold_days: int | None,
    stop_loss_pct: float,
) -> pd.DataFrame:
    if exit_mode in {"Signal-Based", "Hybrid"} and z_exit <= -abs(z_entry):
        raise ValueError("Bollinger exit z-score should be above the entry z-score.")

    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std().replace(0.0, np.nan)
    z_score = (prices - mean).div(std)
    entry_signal = z_score <= -abs(z_entry)
    exit_signal = z_score >= z_exit
    return apply_entry_exit_logic(
        prices=prices,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        exit_mode=exit_mode,
        max_hold_days=max_hold_days,
        stop_loss_pct=stop_loss_pct,
    )


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


def time_series_momentum(
    prices: pd.DataFrame,
    lookback: int,
    entry_threshold: float,
    exit_threshold: float,
    rebalance_frequency: str,
) -> pd.DataFrame:
    if exit_threshold > entry_threshold:
        raise ValueError("Exit threshold should be less than or equal to entry threshold.")

    trailing_return = prices.pct_change(lookback, fill_method=None)
    rebalance_dates = set(get_rebalance_dates(prices.index, rebalance_frequency))

    position = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for ticker in prices.columns:
        in_position = False

        for idx in prices.index:
            px = prices.at[idx, ticker]
            score = trailing_return.at[idx, ticker]

            if pd.isna(px):
                in_position = False
                position.at[idx, ticker] = 0.0
                continue

            if in_position and pd.notna(score) and score <= exit_threshold:
                in_position = False
            elif (not in_position) and idx in rebalance_dates and pd.notna(score) and score >= entry_threshold:
                in_position = True

            position.at[idx, ticker] = 1.0 if in_position else 0.0

    return normalize_long_only(position.where(prices.notna(), 0.0))


def cross_sectional_rank(
    prices: pd.DataFrame,
    lookback: int,
    top_n: int,
    rebalance_frequency: str,
    reverse: bool,
) -> pd.DataFrame:
    trailing_return = prices.pct_change(lookback, fill_method=None)
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


def dual_momentum(
    prices: pd.DataFrame,
    lookback: int,
    top_n: int,
    absolute_threshold: float,
    rebalance_frequency: str,
) -> pd.DataFrame:
    trailing_return = prices.pct_change(lookback, fill_method=None)
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    rebalance_dates = get_rebalance_dates(prices.index, rebalance_frequency)
    for dt in rebalance_dates:
        if dt not in trailing_return.index:
            continue

        scores = trailing_return.loc[dt].dropna()
        scores = scores[scores >= absolute_threshold]
        if scores.empty:
            weights.loc[dt] = 0.0
            continue

        n = int(min(top_n, len(scores)))
        if n <= 0:
            weights.loc[dt] = 0.0
            continue

        selected = scores.nlargest(n).index
        row = pd.Series(0.0, index=prices.columns)
        row.loc[selected] = 1.0 / n
        weights.loc[dt] = row

    weights = weights.ffill().fillna(0.0)
    weights = weights.where(prices.notna(), 0.0)
    return normalize_long_only(weights)


def volatility_adjusted_momentum(
    prices: pd.DataFrame,
    lookback: int,
    vol_window: int,
    top_n: int,
    min_return: float,
    rebalance_frequency: str,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("Number of holdings must be at least 1.")

    trailing_return = prices.pct_change(lookback, fill_method=None)
    trailing_vol = prices.pct_change(fill_method=None).rolling(vol_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    score = trailing_return.div(trailing_vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    rebalance_dates = get_rebalance_dates(prices.index, rebalance_frequency)
    for dt in rebalance_dates:
        if dt not in score.index:
            continue

        score_row = score.loc[dt].dropna()
        if score_row.empty:
            weights.loc[dt] = 0.0
            continue

        return_row = trailing_return.loc[dt]
        eligible = score_row[return_row.loc[score_row.index] >= min_return]
        if eligible.empty:
            weights.loc[dt] = 0.0
            continue

        n = int(min(top_n, len(eligible)))
        selected = eligible.nlargest(n).index
        row = pd.Series(0.0, index=prices.columns)
        row.loc[selected] = 1.0 / n
        weights.loc[dt] = row

    weights = weights.ffill().fillna(0.0)
    weights = weights.where(prices.notna(), 0.0)
    return normalize_long_only(weights)


def fifty_two_week_high_rotation(
    prices: pd.DataFrame,
    lookback: int,
    top_n: int,
    min_high_ratio: float,
    rebalance_frequency: str,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("Number of holdings must be at least 1.")
    if not (0.0 < min_high_ratio <= 1.0):
        raise ValueError("min_high_ratio must be in (0, 1].")

    rolling_high = prices.rolling(lookback).max()
    high_ratio = prices.div(rolling_high).replace([np.inf, -np.inf], np.nan)
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    rebalance_dates = get_rebalance_dates(prices.index, rebalance_frequency)
    for dt in rebalance_dates:
        if dt not in high_ratio.index:
            continue

        ratio = high_ratio.loc[dt].dropna()
        ratio = ratio[ratio >= min_high_ratio]
        if ratio.empty:
            weights.loc[dt] = 0.0
            continue

        n = int(min(top_n, len(ratio)))
        selected = ratio.nlargest(n).index
        row = pd.Series(0.0, index=prices.columns)
        row.loc[selected] = 1.0 / n
        weights.loc[dt] = row

    weights = weights.ffill().fillna(0.0)
    weights = weights.where(prices.notna(), 0.0)
    return normalize_long_only(weights)


def inverse_volatility(prices: pd.DataFrame, vol_window: int, rebalance_frequency: str) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
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


def low_volatility_rotation(
    prices: pd.DataFrame,
    vol_window: int,
    top_n: int,
    rebalance_frequency: str,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("Number of holdings must be at least 1.")

    returns = prices.pct_change(fill_method=None)
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

        n = int(min(top_n, len(vol)))
        selected = vol.nsmallest(n).index
        row = pd.Series(0.0, index=prices.columns)
        row.loc[selected] = 1.0 / n
        weights.loc[dt] = row

    weights = weights.ffill().fillna(0.0)
    weights = weights.where(prices.notna(), 0.0)
    return normalize_long_only(weights)


def classify_executed_trade_action(previous_shares: float, new_shares: float) -> str:
    eps = 1e-12
    if new_shares > previous_shares + eps:
        if previous_shares < -eps and new_shares <= eps:
            return "Cover"
        if previous_shares < -eps and new_shares > eps:
            return "Flip to Long"
        return "Buy"

    if new_shares < previous_shares - eps:
        if previous_shares > eps and new_shares >= -eps:
            return "Sell"
        if previous_shares > eps and new_shares < -eps:
            return "Flip to Short"
        return "Short"

    return "Hold"


def allocate_proportional_with_cap(strength: pd.Series, total_weight: float, per_asset_cap: float) -> pd.Series:
    if strength.empty or total_weight <= 0:
        return pd.Series(0.0, index=strength.index, dtype=float)

    cap = max(0.0, min(1.0, float(per_asset_cap)))
    if cap <= 0:
        return pd.Series(0.0, index=strength.index, dtype=float)

    clean_strength = strength.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if float(clean_strength.sum()) <= 1e-12:
        clean_strength = pd.Series(1.0, index=clean_strength.index)

    target_total = min(float(total_weight), cap * len(clean_strength))
    allocation = pd.Series(0.0, index=clean_strength.index, dtype=float)
    remaining = target_total
    active = clean_strength[clean_strength > 0].index.tolist()

    for _ in range(len(clean_strength) + 2):
        if not active or remaining <= 1e-12:
            break

        active_strength = clean_strength.loc[active]
        strength_sum = float(active_strength.sum())
        if strength_sum <= 1e-12:
            break

        proposal = active_strength / strength_sum * remaining
        cap_left = cap - allocation.loc[active]
        clipped = np.minimum(proposal.values, cap_left.values)
        allocation.loc[active] = allocation.loc[active].values + clipped

        used = float(np.sum(clipped))
        remaining = max(0.0, remaining - used)
        active = [ticker for ticker in active if allocation.at[ticker] < cap - 1e-12]

    return allocation.clip(lower=0.0)


def compute_hold_until_sell_desired_shares(
    current_shares: pd.Series,
    target_row: pd.Series,
    px: pd.Series,
    valid_prices: pd.Series,
    portfolio_value_before: float,
    max_single_position_weight: float,
) -> pd.Series:
    desired_shares = current_shares.copy()
    if portfolio_value_before <= 0:
        return desired_shares

    eps = 1e-12
    cap = max(0.0, min(1.0, float(max_single_position_weight)))
    current_values = current_shares * px
    desired_values = current_values.copy()

    current_sign = pd.Series(
        np.where(current_shares > eps, 1.0, np.where(current_shares < -eps, -1.0, 0.0)),
        index=current_shares.index,
    )
    signal_sign = pd.Series(
        np.where(target_row > eps, 1.0, np.where(target_row < -eps, -1.0, 0.0)),
        index=target_row.index,
    )
    signal_sign = signal_sign.where(valid_prices, 0.0)

    exit_mask = valid_prices & (current_sign != 0.0) & (signal_sign != current_sign)
    desired_values.loc[exit_mask] = 0.0

    keep_mask = valid_prices & (current_sign != 0.0) & (signal_sign == current_sign)
    if cap < 1.0:
        keep_weights = desired_values.abs() / portfolio_value_before
        overweight_keep = keep_mask & (keep_weights > cap + eps)
        if bool(overweight_keep.any()):
            desired_values.loc[overweight_keep] = np.sign(desired_values.loc[overweight_keep]) * cap * portfolio_value_before

    current_gross_after_keep = float(desired_values.loc[keep_mask].abs().sum() / portfolio_value_before)
    available_gross = max(0.0, 1.0 - current_gross_after_keep)

    entry_mask = valid_prices & (current_sign == 0.0) & (signal_sign != 0.0)
    entry_strength = target_row.abs().where(entry_mask, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    entry_strength = entry_strength[entry_strength > eps]
    if not entry_strength.empty and available_gross > eps and cap > 0:
        entry_weights = allocate_proportional_with_cap(entry_strength, available_gross, cap)
        if float(entry_weights.sum()) > eps:
            entry_sign = signal_sign.loc[entry_weights.index]
            desired_values.loc[entry_weights.index] = entry_sign * entry_weights * portfolio_value_before

    desired_shares.loc[valid_prices] = desired_values.loc[valid_prices].div(px.loc[valid_prices]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return desired_shares


def quantize_to_whole_shares(shares: pd.Series, valid_prices: pd.Series) -> pd.Series:
    quantized = shares.copy()
    quantized.loc[valid_prices] = np.trunc(quantized.loc[valid_prices].astype(float))
    quantized.loc[~valid_prices] = 0.0
    return quantized.fillna(0.0)


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float,
    fee_bps: float,
    record_trade_log: bool = True,
    signal_lag_days: int = 1,
    record_position_weights: bool = False,
    execution_style: str = EXECUTION_STYLE_TARGET,
    max_single_position_weight: float = 1.0,
) -> Dict[str, Any]:
    if signal_lag_days < 0:
        raise ValueError("signal_lag_days must be >= 0.")
    if execution_style not in {EXECUTION_STYLE_TARGET, EXECUTION_STYLE_HOLD}:
        raise ValueError(f"Unknown execution_style: {execution_style}")
    if not (0.0 < float(max_single_position_weight) <= 1.0):
        raise ValueError("max_single_position_weight must be in (0, 1].")

    weights = target_weights.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)
    if signal_lag_days > 0:
        # Use prior-day signals for execution to avoid same-bar lookahead bias.
        weights = weights.shift(int(signal_lag_days)).fillna(0.0)
    tradeable_mask = prices.notna() & (prices > 0)
    weights = weights.where(tradeable_mask, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = weights.abs().sum(axis=1)
    over = gross > 1.0 + 1e-9
    if bool(over.any()):
        weights.loc[over] = weights.loc[over].div(gross.loc[over], axis=0)

    fee_rate = fee_bps / 10_000.0
    shares = pd.Series(0.0, index=prices.columns, dtype=float)
    cash = float(initial_capital)
    prev_equity = float(initial_capital)

    equity_values: List[float] = []
    net_returns: List[float] = []
    turnover_values: List[float] = []
    fee_values: List[float] = []
    buy_notional_values: List[float] = []
    sell_notional_values: List[float] = []
    buy_trade_counts: List[int] = []
    sell_trade_counts: List[int] = []
    cash_values: List[float] = []

    trade_rows: List[Dict[str, Any]] = []
    max_trade_log_rows = 250_000
    trade_log_truncated = False
    total_trade_events = 0
    position_weight_rows: List[pd.Series] = []

    for dt in prices.index:
        price_row = prices.loc[dt]
        valid_prices = price_row.notna() & (price_row > 0)
        px = price_row.fillna(0.0).astype(float)

        current_values = shares * px
        portfolio_value_before = cash + float(current_values.sum())

        target_row = weights.loc[dt].copy()
        target_row = target_row.where(valid_prices, 0.0).fillna(0.0)
        current_shares = shares.copy()
        if execution_style == EXECUTION_STYLE_HOLD:
            desired_shares = compute_hold_until_sell_desired_shares(
                current_shares=current_shares,
                target_row=target_row,
                px=px,
                valid_prices=valid_prices,
                portfolio_value_before=portfolio_value_before,
                max_single_position_weight=max_single_position_weight,
            )
        else:
            target_gross = float(target_row.abs().sum())
            if target_gross > 1.0 + 1e-9:
                target_row = target_row / target_gross
                target_gross = float(target_row.abs().sum())

            scale = 1.0
            if fee_rate > 0 and portfolio_value_before > 0 and target_gross >= 0.999:
                # Keep full-investment targets feasible by reserving expected transaction costs.
                for _ in range(4):
                    desired_values_est = target_row * portfolio_value_before * scale
                    trade_notional_est = float((desired_values_est - current_values).abs().sum())
                    fee_est = trade_notional_est * fee_rate
                    new_scale = max((portfolio_value_before - fee_est) / portfolio_value_before, 0.0)
                    if abs(new_scale - scale) < 1e-10:
                        break
                    scale = new_scale

            desired_values = target_row * portfolio_value_before * scale
            desired_shares = current_shares.copy()
            desired_shares.loc[valid_prices] = desired_values.loc[valid_prices].div(px.loc[valid_prices]).fillna(0.0)
        desired_shares = quantize_to_whole_shares(desired_shares, valid_prices)

        delta_shares = pd.Series(0.0, index=prices.columns, dtype=float)
        delta_shares.loc[valid_prices] = desired_shares.loc[valid_prices] - current_shares.loc[valid_prices]

        # No-leverage cash rule: buy-side notional (including fees) cannot exceed available cash.
        prelim_buys_mask = delta_shares > 1e-12
        prelim_sells_mask = delta_shares < -1e-12
        prelim_buy_notional = float((delta_shares.loc[prelim_buys_mask] * px.loc[prelim_buys_mask]).sum())
        prelim_sell_notional = float(((-delta_shares.loc[prelim_sells_mask]) * px.loc[prelim_sells_mask]).sum())
        if prelim_buy_notional > 0:
            max_affordable_buy_notional = (cash + prelim_sell_notional * (1.0 - fee_rate)) / (1.0 + fee_rate)
            max_affordable_buy_notional = max(0.0, float(max_affordable_buy_notional))
            if prelim_buy_notional > max_affordable_buy_notional + 1e-10:
                buy_scale = max_affordable_buy_notional / prelim_buy_notional if prelim_buy_notional > 0 else 0.0
                delta_shares.loc[prelim_buys_mask] = delta_shares.loc[prelim_buys_mask] * buy_scale
                desired_shares = current_shares + delta_shares
                desired_shares = quantize_to_whole_shares(desired_shares, valid_prices)
                delta_shares.loc[valid_prices] = desired_shares.loc[valid_prices] - current_shares.loc[valid_prices]

        signed_trade_values = delta_shares * px
        abs_trade_values = signed_trade_values.abs()
        trade_notional = float(abs_trade_values.sum())
        transaction_fee = trade_notional * fee_rate

        buys_mask = delta_shares > 1e-12
        sells_mask = delta_shares < -1e-12
        buy_notional = float((delta_shares.loc[buys_mask] * px.loc[buys_mask]).sum())
        sell_notional = float(((-delta_shares.loc[sells_mask]) * px.loc[sells_mask]).sum())
        buy_count = int(buys_mask.sum())
        sell_count = int(sells_mask.sum())
        total_trade_events += buy_count + sell_count

        cash = cash - float(signed_trade_values.sum()) - transaction_fee
        if cash < 0 and abs(cash) < 1e-8:
            cash = 0.0
        shares.loc[valid_prices] = desired_shares.loc[valid_prices]

        end_values = shares * px
        portfolio_value_after = cash + float(end_values.sum())

        day_return = (portfolio_value_after / prev_equity - 1.0) if prev_equity > 0 else 0.0
        day_turnover = (trade_notional / portfolio_value_before) if portfolio_value_before > 0 else 0.0

        equity_values.append(portfolio_value_after)
        net_returns.append(day_return)
        turnover_values.append(day_turnover)
        fee_values.append(transaction_fee)
        buy_notional_values.append(buy_notional)
        sell_notional_values.append(sell_notional)
        buy_trade_counts.append(buy_count)
        sell_trade_counts.append(sell_count)
        cash_values.append(cash)

        if record_position_weights:
            if portfolio_value_after > 0:
                position_weight_rows.append((end_values / portfolio_value_after).fillna(0.0))
            else:
                position_weight_rows.append(pd.Series(0.0, index=prices.columns))

        if record_trade_log and buy_count + sell_count > 0:
            from_weight = current_values / portfolio_value_before if portfolio_value_before > 0 else pd.Series(0.0, index=prices.columns)
            to_weight = end_values / portfolio_value_after if portfolio_value_after > 0 else pd.Series(0.0, index=prices.columns)
            fee_allocation = (
                abs_trade_values * (transaction_fee / trade_notional)
                if trade_notional > 0
                else pd.Series(0.0, index=prices.columns)
            )

            traded_mask = (buys_mask | sells_mask)
            for ticker in prices.columns[traded_mask]:
                if len(trade_rows) >= max_trade_log_rows:
                    trade_log_truncated = True
                    break

                delta = float(delta_shares.at[ticker])
                prev_sh = float(current_shares.at[ticker])
                new_sh = float(desired_shares.at[ticker])
                trade_rows.append(
                    {
                        "Date": pd.Timestamp(dt),
                        "Ticker": ticker,
                        "Action": classify_executed_trade_action(prev_sh, new_sh),
                        "Shares": abs(delta),
                        "Price": float(px.at[ticker]),
                        "Trade Value": float(abs(delta * float(px.at[ticker]))),
                        "Fee": float(fee_allocation.at[ticker]),
                        "From Weight": float(from_weight.at[ticker]) if np.isfinite(from_weight.at[ticker]) else 0.0,
                        "To Weight": float(to_weight.at[ticker]) if np.isfinite(to_weight.at[ticker]) else 0.0,
                    }
                )

        prev_equity = portfolio_value_after

    index = prices.index
    returns_series = pd.Series(net_returns, index=index, name="Net Return")
    equity_series = pd.Series(equity_values, index=index, name="Equity")
    turnover_series = pd.Series(turnover_values, index=index, name="Turnover")
    fee_series = pd.Series(fee_values, index=index, name="Transaction Fee")
    cash_series = pd.Series(cash_values, index=index, name="Cash")

    trade_summary = pd.DataFrame(
        {
            "Buy Notional": buy_notional_values,
            "Sell Notional": sell_notional_values,
            "Buy Trades": buy_trade_counts,
            "Sell Trades": sell_trade_counts,
            "Turnover": turnover_values,
            "Transaction Fee": fee_values,
            "Cash": cash_values,
        },
        index=index,
    )

    trade_log = pd.DataFrame(trade_rows)
    if not trade_log.empty:
        trade_log = trade_log.sort_values("Date").reset_index(drop=True)

    if record_position_weights and position_weight_rows:
        position_weights = pd.DataFrame(position_weight_rows, index=index, columns=prices.columns).fillna(0.0)
    else:
        position_weights = pd.DataFrame(index=index, columns=prices.columns, dtype=float)

    latest_prices = prices.iloc[-1].fillna(0.0)
    final_equity = float(equity_series.iloc[-1]) if not equity_series.empty else float(initial_capital)
    portfolio_rows: List[Dict[str, float | str | None]] = []
    for ticker in prices.columns:
        sh = float(shares.at[ticker])
        px = float(latest_prices.at[ticker])
        if abs(sh) <= 1e-10 or px <= 0:
            continue
        market_value = sh * px
        if abs(market_value) <= 1e-6:
            continue
        portfolio_rows.append(
            {
                "Ticker": ticker,
                "Type": "Long" if sh > 0 else "Short",
                "Shares": sh,
                "Last Price": px,
                "Market Value": market_value,
                "Weight": (market_value / final_equity) if final_equity > 0 else 0.0,
                "Gross Weight": (abs(market_value) / final_equity) if final_equity > 0 else 0.0,
            }
        )

    final_portfolio = pd.DataFrame(portfolio_rows)
    if not final_portfolio.empty:
        final_portfolio = final_portfolio.sort_values("Market Value", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    else:
        final_portfolio = pd.DataFrame(columns=["Ticker", "Type", "Shares", "Last Price", "Market Value", "Weight", "Gross Weight"])

    cash_weight = (cash / final_equity) if final_equity > 0 else 0.0
    cash_row = pd.DataFrame(
        [
            {
                "Ticker": "CASH",
                "Type": "Cash",
                "Shares": np.nan,
                "Last Price": 1.0,
                "Market Value": cash,
                "Weight": cash_weight,
                "Gross Weight": 0.0,
            }
        ]
    )
    if final_portfolio.empty:
        final_portfolio = cash_row.copy()
    else:
        final_portfolio = pd.concat([final_portfolio, cash_row], ignore_index=True)

    return {
        "returns": returns_series,
        "equity_curve": equity_series,
        "turnover": turnover_series,
        "fees": fee_series,
        "cash": cash_series,
        "trade_summary": trade_summary,
        "trade_log": trade_log,
        "total_trade_events": total_trade_events,
        "trade_log_truncated": trade_log_truncated,
        "final_portfolio": final_portfolio,
        "position_weights": position_weights,
    }


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


def build_strategy_weights(strategy_name: str, prices: pd.DataFrame, params: Dict[str, float | int | str]) -> pd.DataFrame:
    signal_prices = prices.where(prices > 0)
    raw_weights: pd.DataFrame

    if strategy_name == "Buy & Hold Equal Weight":
        raw_weights = equal_weight_positions(signal_prices)
    elif strategy_name == "SMA Crossover":
        raw_weights = sma_crossover(
            prices=signal_prices,
            fast_window=int(params["fast_window"]),
            slow_window=int(params["slow_window"]),
            band_pct=float(params["band_pct"]),
        )
    elif strategy_name == "EMA Crossover":
        raw_weights = ema_crossover(
            prices=signal_prices,
            fast_window=int(params["fast_window"]),
            slow_window=int(params["slow_window"]),
            band_pct=float(params["band_pct"]),
        )
    elif strategy_name == "MACD Trend":
        raw_weights = macd_trend(
            prices=signal_prices,
            fast_window=int(params["fast_window"]),
            slow_window=int(params["slow_window"]),
            signal_window=int(params["signal_window"]),
            hist_entry=float(params["hist_entry"]),
            hist_exit=float(params["hist_exit"]),
        )
    elif strategy_name == "Moving Average Reversion":
        raw_weights = moving_average_reversion(
            prices=signal_prices,
            window=int(params["ma_window"]),
            entry_deviation_pct=float(params["entry_deviation_pct"]),
            exit_mode=str(params["exit_mode"]),
            exit_deviation_pct=float(params["exit_deviation_pct"]),
            max_hold_days=int(params["max_hold_days"]) if "max_hold_days" in params else None,
            stop_loss_pct=float(params["stop_loss_pct"]),
        )
    elif strategy_name == "RSI Mean Reversion":
        raw_weights = rsi_mean_reversion(
            prices=signal_prices,
            window=int(params["rsi_window"]),
            oversold=float(params["oversold"]),
            exit_mode=str(params["exit_mode"]),
            exit_rsi=float(params["exit_rsi"]),
            max_hold_days=int(params["max_hold_days"]) if "max_hold_days" in params else None,
            stop_loss_pct=float(params["stop_loss_pct"]),
        )
    elif strategy_name == "Bollinger Mean Reversion":
        raw_weights = bollinger_mean_reversion(
            prices=signal_prices,
            window=int(params["bb_window"]),
            z_entry=float(params["z_entry"]),
            exit_mode=str(params["exit_mode"]),
            z_exit=float(params["z_exit"]),
            max_hold_days=int(params["max_hold_days"]) if "max_hold_days" in params else None,
            stop_loss_pct=float(params["stop_loss_pct"]),
        )
    elif strategy_name == "Donchian Breakout":
        raw_weights = donchian_breakout(
            signal_prices,
            entry_window=int(params["entry_window"]),
            exit_window=int(params["exit_window"]),
        )
    elif strategy_name == "Time-Series Momentum":
        raw_weights = time_series_momentum(
            prices=signal_prices,
            lookback=int(params["lookback"]),
            entry_threshold=float(params["entry_threshold"]),
            exit_threshold=float(params["exit_threshold"]),
            rebalance_frequency=str(params["rebalance_freq"]),
        )
    elif strategy_name == "Cross-Sectional Momentum":
        raw_weights = cross_sectional_rank(
            signal_prices,
            lookback=int(params["lookback"]),
            top_n=int(params["top_n"]),
            rebalance_frequency=str(params["rebalance_freq"]),
            reverse=False,
        )
    elif strategy_name == "Dual Momentum":
        raw_weights = dual_momentum(
            prices=signal_prices,
            lookback=int(params["lookback"]),
            top_n=int(params["top_n"]),
            absolute_threshold=float(params["absolute_threshold"]),
            rebalance_frequency=str(params["rebalance_freq"]),
        )
    elif strategy_name == "Volatility-Adjusted Momentum":
        raw_weights = volatility_adjusted_momentum(
            prices=signal_prices,
            lookback=int(params["lookback"]),
            vol_window=int(params["vol_window"]),
            top_n=int(params["top_n"]),
            min_return=float(params["min_return"]),
            rebalance_frequency=str(params["rebalance_freq"]),
        )
    elif strategy_name == "52-Week High Rotation":
        raw_weights = fifty_two_week_high_rotation(
            prices=signal_prices,
            lookback=int(params["lookback"]),
            top_n=int(params["top_n"]),
            min_high_ratio=float(params["min_high_ratio"]),
            rebalance_frequency=str(params["rebalance_freq"]),
        )
    elif strategy_name == "Cross-Sectional Mean Reversion":
        raw_weights = cross_sectional_rank(
            signal_prices,
            lookback=int(params["lookback"]),
            top_n=int(params["top_n"]),
            rebalance_frequency=str(params["rebalance_freq"]),
            reverse=True,
        )
    elif strategy_name == "Inverse Volatility (Risk Parity Lite)":
        raw_weights = inverse_volatility(
            signal_prices,
            vol_window=int(params["vol_window"]),
            rebalance_frequency=str(params["rebalance_freq"]),
        )
    elif strategy_name == "Low Volatility Rotation":
        raw_weights = low_volatility_rotation(
            prices=signal_prices,
            vol_window=int(params["vol_window"]),
            top_n=int(params["top_n"]),
            rebalance_frequency=str(params["rebalance_freq"]),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return enforce_tradeable_weights(raw_weights, signal_prices)


def build_strategy_signal_strength(strategy_name: str, prices: pd.DataFrame, params: Dict[str, float | int | str]) -> pd.DataFrame:
    signal_prices = prices.where(prices > 0)
    score: pd.DataFrame

    if strategy_name == "Buy & Hold Equal Weight":
        score = pd.DataFrame(1.0, index=signal_prices.index, columns=signal_prices.columns)
    elif strategy_name == "SMA Crossover":
        fast_ma = signal_prices.rolling(int(params["fast_window"])).mean()
        slow_ma = signal_prices.rolling(int(params["slow_window"])).mean()
        score = fast_ma.div(slow_ma.replace(0.0, np.nan)) - 1.0
    elif strategy_name == "EMA Crossover":
        fast_ma = signal_prices.ewm(span=int(params["fast_window"]), adjust=False).mean()
        slow_ma = signal_prices.ewm(span=int(params["slow_window"]), adjust=False).mean()
        score = fast_ma.div(slow_ma.replace(0.0, np.nan)) - 1.0
    elif strategy_name == "MACD Trend":
        fast_ema = signal_prices.ewm(span=int(params["fast_window"]), adjust=False).mean()
        slow_ema = signal_prices.ewm(span=int(params["slow_window"]), adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=int(params["signal_window"]), adjust=False).mean()
        score = macd_line - signal_line
    elif strategy_name == "Moving Average Reversion":
        mean = signal_prices.rolling(int(params["ma_window"])).mean()
        deviation = signal_prices.div(mean) - 1.0
        score = -deviation
    elif strategy_name == "RSI Mean Reversion":
        rsi = compute_rsi(signal_prices, int(params["rsi_window"]))
        score = (50.0 - rsi) / 50.0
    elif strategy_name == "Bollinger Mean Reversion":
        mean = signal_prices.rolling(int(params["bb_window"])).mean()
        std = signal_prices.rolling(int(params["bb_window"])).std().replace(0.0, np.nan)
        z_score = (signal_prices - mean).div(std)
        score = -z_score
    elif strategy_name == "Donchian Breakout":
        channel_high = signal_prices.rolling(int(params["entry_window"])).max().shift(1)
        score = signal_prices.div(channel_high.replace(0.0, np.nan)) - 1.0
    elif strategy_name == "Time-Series Momentum":
        score = signal_prices.pct_change(int(params["lookback"]), fill_method=None)
    elif strategy_name == "Cross-Sectional Momentum":
        score = signal_prices.pct_change(int(params["lookback"]), fill_method=None)
    elif strategy_name == "Dual Momentum":
        score = signal_prices.pct_change(int(params["lookback"]), fill_method=None) - float(params["absolute_threshold"])
    elif strategy_name == "Volatility-Adjusted Momentum":
        trailing_return = signal_prices.pct_change(int(params["lookback"]), fill_method=None)
        trailing_vol = signal_prices.pct_change(fill_method=None).rolling(int(params["vol_window"])).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        score = trailing_return.div(trailing_vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    elif strategy_name == "52-Week High Rotation":
        rolling_high = signal_prices.rolling(int(params["lookback"])).max()
        score = signal_prices.div(rolling_high.replace(0.0, np.nan)) - 1.0
    elif strategy_name == "Cross-Sectional Mean Reversion":
        trailing_return = signal_prices.pct_change(int(params["lookback"]), fill_method=None)
        score = -trailing_return
    elif strategy_name == "Inverse Volatility (Risk Parity Lite)":
        trailing_vol = signal_prices.pct_change(fill_method=None).rolling(int(params["vol_window"])).std()
        score = 1.0 / trailing_vol.replace(0.0, np.nan)
    elif strategy_name == "Low Volatility Rotation":
        trailing_vol = signal_prices.pct_change(fill_method=None).rolling(int(params["vol_window"])).std()
        score = 1.0 / trailing_vol.replace(0.0, np.nan)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    score = score.reindex(index=signal_prices.index, columns=signal_prices.columns)
    score = score.where(signal_prices.notna())
    return score.replace([np.inf, -np.inf], np.nan)


def build_standard_benchmark_results(
    reference_index: pd.DatetimeIndex,
    start_date: date,
    end_date: date,
    initial_capital: float,
) -> Dict[str, Dict[str, pd.Series]]:
    benchmark_results: Dict[str, Dict[str, pd.Series]] = {}
    benchmark_prices = download_prices(("SPY", "QQQ"), start_date.isoformat(), end_date.isoformat())
    if benchmark_prices.empty:
        return benchmark_results

    benchmark_prices = benchmark_prices.reindex(reference_index).sort_index().ffill()
    benchmark_map = [("SPY", "S&P 500 (SPY)"), ("QQQ", "NASDAQ-100 (QQQ)")]
    for ticker, label in benchmark_map:
        if ticker not in benchmark_prices.columns:
            continue

        px = benchmark_prices[[ticker]].copy()
        px[ticker] = px[ticker].where(px[ticker] > 0)
        if int(px[ticker].dropna().shape[0]) < 2:
            continue

        weights = equal_weight_positions(px.where(px > 0))
        result = run_backtest(
            px,
            weights,
            initial_capital=initial_capital,
            fee_bps=0.0,
            record_trade_log=False,
            record_position_weights=False,
            execution_style=EXECUTION_STYLE_TARGET,
        )
        benchmark_results[label] = {
            "returns": result["returns"],
            "equity_curve": result["equity_curve"],
        }

    return benchmark_results


@st.cache_data(show_spinner=False)
def fetch_current_market_caps(tickers: Tuple[str, ...]) -> Dict[str, float]:
    normalized = tuple(normalize_ticker_list(list(tickers)))
    if not normalized:
        return {}

    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    chunk_size = 200
    market_caps: Dict[str, float] = {}

    for i in range(0, len(normalized), chunk_size):
        chunk = normalized[i : i + chunk_size]
        try:
            response = requests.get(
                url,
                headers=HTTP_HEADERS,
                params={"symbols": ",".join(chunk)},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("quoteResponse", {}).get("result", [])
        except Exception:
            continue

        for item in results:
            symbol_raw = str(item.get("symbol", "")).strip()
            symbol_norm = normalize_ticker_list([symbol_raw])
            if not symbol_norm:
                continue
            symbol = symbol_norm[0]
            market_cap = item.get("marketCap")
            if isinstance(market_cap, (int, float)) and np.isfinite(market_cap) and market_cap > 0:
                market_caps[symbol] = float(market_cap)

    return market_caps


def build_universe_market_cap_hold_benchmark(
    prices: pd.DataFrame,
    initial_capital: float,
) -> Tuple[Dict[str, pd.Series] | None, str | None]:
    if prices.empty:
        return None, "Market-cap benchmark skipped: no prices available."

    start_row = prices.iloc[0]
    latest_row = prices.ffill().iloc[-1]
    start_tradeable = (start_row > 0) & start_row.notna()
    tickers = [ticker for ticker in prices.columns if bool(start_tradeable.get(ticker, False))]
    if not tickers:
        return None, "Market-cap benchmark skipped: no tradeable tickers on start date."

    current_market_caps = fetch_current_market_caps(tuple(tickers))
    estimated_start_caps = pd.Series(index=tickers, dtype=float)

    for ticker in tickers:
        start_px = float(start_row.get(ticker, np.nan))
        last_px = float(latest_row.get(ticker, np.nan))
        market_cap_now = float(current_market_caps.get(ticker, np.nan))
        if not (np.isfinite(start_px) and start_px > 0 and np.isfinite(last_px) and last_px > 0 and np.isfinite(market_cap_now) and market_cap_now > 0):
            continue
        implied_shares = market_cap_now / last_px
        if np.isfinite(implied_shares) and implied_shares > 0:
            estimated_start_caps.at[ticker] = implied_shares * start_px

    with_cap = int(estimated_start_caps.notna().sum())
    missing_cap_tickers = [ticker for ticker in tickers if pd.isna(estimated_start_caps.get(ticker))]
    note: str | None = None

    if with_cap == 0:
        initial_weights = pd.Series(1.0 / len(tickers), index=tickers)
        note = (
            "Market-cap benchmark fallback: no market-cap data retrieved, "
            "so equal-weight initial universe buy-and-hold was used instead."
        )
    else:
        if missing_cap_tickers:
            fill_cap = float(estimated_start_caps.dropna().median())
            if not np.isfinite(fill_cap) or fill_cap <= 0:
                fill_cap = float(estimated_start_caps.dropna().mean())
            if not np.isfinite(fill_cap) or fill_cap <= 0:
                fill_cap = 1.0
            estimated_start_caps.loc[missing_cap_tickers] = fill_cap
            note = (
                f"Market-cap benchmark uses estimated start-date market caps "
                f"(current market cap scaled by price ratio). "
                f"Filled missing caps for {len(missing_cap_tickers)} ticker(s) using median estimated cap."
            )
        else:
            note = (
                "Market-cap benchmark uses estimated start-date market caps "
                "(current market cap scaled by price ratio)."
            )

        initial_weights = estimated_start_caps / float(estimated_start_caps.sum())

    benchmark_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    benchmark_weights.loc[:, tickers] = initial_weights.reindex(tickers).values
    benchmark_result = run_backtest(
        prices=prices,
        target_weights=benchmark_weights,
        initial_capital=float(initial_capital),
        fee_bps=0.0,
        record_trade_log=False,
        record_position_weights=False,
        execution_style=EXECUTION_STYLE_HOLD,
        max_single_position_weight=1.0,
    )

    return {
        "returns": benchmark_result["returns"],
        "equity_curve": benchmark_result["equity_curve"],
    }, note


def render_metrics_table(
    strategy_returns: pd.Series,
    strategy_equity: pd.Series,
    benchmark_results: Dict[str, Dict[str, pd.Series]],
) -> None:
    rows: Dict[str, Dict[str, float]] = {"Strategy": performance_metrics(strategy_returns, strategy_equity)}
    for name, data in benchmark_results.items():
        rows[name] = performance_metrics(data["returns"], data["equity_curve"])

    table = pd.DataFrame.from_dict(rows, orient="index")

    percent_cols = ["Total Return", "CAGR", "Volatility", "Max Drawdown", "Win Rate"]
    for col in percent_cols:
        table[col] = table[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    number_cols = ["Sharpe", "Calmar"]
    for col in number_cols:
        table[col] = table[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    st.dataframe(table, use_container_width=True)


def render_final_portfolio(final_portfolio: pd.DataFrame, ending_value: float) -> None:
    if "Gross Weight" not in final_portfolio.columns:
        final_portfolio = final_portfolio.copy()
        final_portfolio["Gross Weight"] = final_portfolio["Weight"].abs() if "Weight" in final_portfolio.columns else 0.0
    if "Type" in final_portfolio.columns and "Equity" in final_portfolio["Type"].astype(str).unique().tolist():
        final_portfolio = final_portfolio.copy()
        final_portfolio.loc[final_portfolio["Type"] == "Equity", "Type"] = np.where(
            final_portfolio.loc[final_portfolio["Type"] == "Equity", "Shares"] >= 0,
            "Long",
            "Short",
        )

    cash_row = final_portfolio[final_portfolio["Ticker"] == "CASH"]
    cash_value = float(cash_row["Market Value"].iloc[0]) if not cash_row.empty else 0.0
    cash_weight = float(cash_row["Weight"].iloc[0]) if not cash_row.empty else 0.0
    holdings = final_portfolio[final_portfolio["Type"] != "Cash"].copy()
    gross_exposure = float(holdings["Gross Weight"].sum()) if ("Gross Weight" in holdings.columns and not holdings.empty) else 0.0
    long_count = int((holdings["Type"] == "Long").sum()) if not holdings.empty else 0
    short_count = int((holdings["Type"] == "Short").sum()) if not holdings.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ending Portfolio Value", f"${ending_value:,.2f}")
    with col2:
        st.metric("Cash", f"${cash_value:,.2f}", f"{cash_weight:.2%} weight")
    with col3:
        st.metric("Open Positions", f"{len(holdings)}", f"Long {long_count} / Short {short_count}")
    with col4:
        st.metric("Gross Exposure", f"{gross_exposure:.2%}")

    st.subheader("Final Holdings")
    if holdings.empty:
        st.write("No open positions at the end of the backtest.")
    else:
        top_holdings = holdings.assign(**{"Abs Market Value": holdings["Market Value"].abs()}).nlargest(15, "Abs Market Value")
        top_holdings = top_holdings.sort_values("Market Value", ascending=True)
        color_map = np.where(top_holdings["Type"] == "Long", "#2ca02c", "#d62728")
        fig_holdings = go.Figure()
        fig_holdings.add_trace(
            go.Bar(
                x=top_holdings["Market Value"],
                y=top_holdings["Ticker"],
                orientation="h",
                marker_color=color_map,
                name="Market Value",
            )
        )
        fig_holdings.update_layout(title="Top Final Holdings (Signed)", xaxis_title="Market Value ($)", yaxis_title="Ticker")
        st.plotly_chart(fig_holdings, use_container_width=True)

    portfolio_display = final_portfolio.copy()
    st.dataframe(
        portfolio_display.style.format(
            {
                "Shares": "{:,.0f}",
                "Last Price": "${:,.2f}",
                "Market Value": "${:,.2f}",
                "Weight": "{:.2%}",
                "Gross Weight": "{:.2%}",
            }
        ),
        use_container_width=True,
    )


def render_trade_log(
    trade_log: pd.DataFrame,
    trade_summary: pd.DataFrame,
    total_trade_events: int,
    trade_log_truncated: bool,
) -> None:
    trade_days = int(((trade_summary["Buy Trades"] + trade_summary["Sell Trades"]) > 0).sum())
    total_buy_notional = float(trade_summary["Buy Notional"].sum())
    total_sell_notional = float(trade_summary["Sell Notional"].sum())
    total_fees = float(trade_summary["Transaction Fee"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Trade Events", f"{total_trade_events:,}")
    with c2:
        st.metric("Trade Days", f"{trade_days:,}")
    with c3:
        st.metric("Total Buy Notional", f"${total_buy_notional:,.0f}")
    with c4:
        st.metric("Total Sell Notional", f"${total_sell_notional:,.0f}")
    st.caption(f"Total transaction fees: ${total_fees:,.2f}")

    activity = trade_summary[(trade_summary["Buy Trades"] + trade_summary["Sell Trades"]) > 0]
    if not activity.empty:
        fig_activity = go.Figure()
        fig_activity.add_trace(go.Bar(x=activity.index, y=activity["Buy Notional"], name="Buy Notional"))
        fig_activity.add_trace(go.Bar(x=activity.index, y=-activity["Sell Notional"], name="Sell Notional"))
        fig_activity.update_layout(
            title="Daily Trade Activity",
            xaxis_title="Date",
            yaxis_title="Notional ($)",
            barmode="relative",
        )
        st.plotly_chart(fig_activity, use_container_width=True)

    if trade_log.empty:
        st.info("No trades were executed for this backtest.")
        return

    if trade_log_truncated:
        st.warning(
            "Trade log display was truncated at 250,000 rows to keep the app responsive. "
            "Narrow the universe/date range for full detail."
        )

    min_trade_date = trade_log["Date"].min().date()
    max_trade_date = trade_log["Date"].max().date()
    all_actions = sorted(trade_log["Action"].dropna().astype(str).unique().tolist())
    all_tickers = sorted(trade_log["Ticker"].unique().tolist())

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        selected_dates = st.date_input(
            "Trade dates",
            value=(min_trade_date, max_trade_date),
            min_value=min_trade_date,
            max_value=max_trade_date,
            key="trade_log_dates",
        )
    with filter_col2:
        selected_actions = st.multiselect("Action", all_actions, default=all_actions, key="trade_log_actions")
    with filter_col3:
        selected_tickers = st.multiselect("Tickers", all_tickers, default=[], key="trade_log_tickers")
    with filter_col4:
        display_rows = st.select_slider(
            "Rows to display",
            options=[100, 250, 500, 1000, 2500, 5000, 10000, "All"],
            value=500,
            key="trade_log_rows",
        )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_filter = pd.Timestamp(selected_dates[0])
        end_filter = pd.Timestamp(selected_dates[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        start_filter = pd.Timestamp(min_trade_date)
        end_filter = pd.Timestamp(max_trade_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    filtered_log = trade_log[
        (trade_log["Date"] >= start_filter)
        & (trade_log["Date"] <= end_filter)
        & (trade_log["Action"].isin(selected_actions))
    ].copy()
    if selected_tickers:
        filtered_log = filtered_log[filtered_log["Ticker"].isin(selected_tickers)]

    filtered_log = filtered_log.sort_values("Date", ascending=False)
    if display_rows != "All":
        filtered_display = filtered_log.head(int(display_rows))
    else:
        filtered_display = filtered_log

    st.caption(
        f"Showing {len(filtered_display):,} of {len(filtered_log):,} filtered trades "
        f"({len(trade_log):,} rows currently loaded)."
    )

    st.dataframe(
        filtered_display.style.format(
            {
                "Shares": "{:,.0f}",
                "Price": "${:,.2f}",
                "Trade Value": "${:,.2f}",
                "Fee": "${:,.4f}",
                "From Weight": "{:.2%}",
                "To Weight": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    csv_bytes = filtered_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Trade Log (CSV)",
        data=csv_bytes,
        file_name="trade_log.csv",
        mime="text/csv",
        use_container_width=False,
    )


def render_extreme_move_attribution(
    prices: pd.DataFrame,
    equity_curve: pd.Series,
    fee_series: pd.Series,
    position_weights: pd.DataFrame,
) -> None:
    if equity_curve.empty or prices.empty or position_weights.empty or len(equity_curve) < 3:
        st.info("Not enough history to compute extreme-period attribution.")
        return

    max_window = min(20, max(1, len(equity_curve) - 1))
    window_days = st.slider(
        "Attribution window (trading days)",
        min_value=1,
        max_value=max_window,
        value=min(5, max_window),
        step=1,
        key=widget_key("bt", "attr_window"),
    )
    threshold_pct = st.slider(
        "Massive move threshold (absolute return, %)",
        min_value=0.5,
        max_value=30.0,
        value=5.0,
        step=0.5,
        key=widget_key("bt", "attr_threshold"),
    )
    max_events = st.slider(
        "Max extreme periods shown",
        min_value=3,
        max_value=30,
        value=10,
        step=1,
        key=widget_key("bt", "attr_events"),
    )
    top_contributors = st.slider(
        "Top contributors to display",
        min_value=3,
        max_value=30,
        value=10,
        step=1,
        key=widget_key("bt", "attr_top_n"),
    )

    period_returns = (equity_curve / equity_curve.shift(window_days) - 1.0).dropna()
    if period_returns.empty:
        st.info("No valid periods for attribution.")
        return

    massive = period_returns[period_returns.abs() >= (threshold_pct / 100.0)]
    if massive.empty:
        st.info(
            f"No {window_days}-day periods exceeded +/-{threshold_pct:.1f}% in this backtest. "
            "Lower the threshold to inspect smaller moves."
        )
        return

    massive = massive.reindex(massive.abs().sort_values(ascending=False).index).head(max_events)
    events: List[Dict[str, Any]] = []
    for end_dt, ret in massive.items():
        end_loc = equity_curve.index.get_loc(end_dt)
        start_dt = equity_curve.index[end_loc - window_days]
        events.append(
            {
                "Start Date": pd.Timestamp(start_dt),
                "End Date": pd.Timestamp(end_dt),
                "Window Days": window_days,
                "Period Return": float(ret),
            }
        )

    events_df = pd.DataFrame(events)
    display_events = events_df.copy()
    display_events["Period Return"] = display_events["Period Return"].map(lambda x: f"{x:.2%}")
    st.dataframe(display_events, use_container_width=True)

    option_map: Dict[str, Dict[str, Any]] = {}
    for row in events:
        label = (
            f"{row['Start Date'].date()} -> {row['End Date'].date()} "
            f"({row['Window Days']}d, {row['Period Return']:+.2%})"
        )
        option_map[label] = row

    selected_label = st.selectbox(
        "Select an extreme period to decompose",
        list(option_map.keys()),
        key=widget_key("bt", "attr_select_period"),
    )
    selected = option_map[selected_label]
    start_dt = selected["Start Date"]
    end_dt = selected["End Date"]
    period_return = float(selected["Period Return"])

    asset_returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    effective_weights = position_weights.shift(1).fillna(0.0)
    daily_contrib_pct = effective_weights.mul(asset_returns, axis=0).fillna(0.0)
    prev_equity = equity_curve.shift(1).replace(0.0, np.nan)
    daily_contrib_dollar = daily_contrib_pct.mul(prev_equity, axis=0).fillna(0.0)

    period_mask = (daily_contrib_pct.index > start_dt) & (daily_contrib_pct.index <= end_dt)
    if period_mask.sum() == 0:
        st.info("No overlapping contribution rows for the selected period.")
        return

    start_equity = float(equity_curve.loc[start_dt]) if start_dt in equity_curve.index else float(equity_curve.iloc[0])
    period_contrib_dollar = daily_contrib_dollar.loc[period_mask].sum(axis=0)
    period_contrib_pct = period_contrib_dollar / start_equity if start_equity > 0 else period_contrib_dollar * 0.0
    avg_weight = effective_weights.loc[period_mask].mean(axis=0)
    end_weight = position_weights.loc[end_dt] if end_dt in position_weights.index else pd.Series(0.0, index=prices.columns)

    contrib_df = pd.DataFrame(
        {
            "Ticker": prices.columns,
            "Contribution %pt": [float(period_contrib_pct.get(t, 0.0)) * 100.0 for t in prices.columns],
            "Contribution $": [float(period_contrib_dollar.get(t, 0.0)) for t in prices.columns],
            "Average Weight": [float(avg_weight.get(t, 0.0)) for t in prices.columns],
            "End Weight": [float(end_weight.get(t, 0.0)) for t in prices.columns],
        }
    )
    contrib_df = contrib_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    contrib_df["Abs Contribution %pt"] = contrib_df["Contribution %pt"].abs()
    contrib_df = contrib_df.sort_values("Abs Contribution %pt", ascending=False)

    positive = contrib_df[contrib_df["Contribution %pt"] > 0].head(top_contributors)
    negative = contrib_df[contrib_df["Contribution %pt"] < 0].sort_values("Contribution %pt", ascending=True).head(top_contributors)

    p_col, n_col = st.columns(2)
    with p_col:
        st.markdown("**Top Positive Contributors**")
        if positive.empty:
            st.info("No positive contributors in this period.")
        else:
            st.dataframe(
                positive[["Ticker", "Contribution %pt", "Contribution $", "Average Weight", "End Weight"]].style.format(
                    {
                        "Contribution %pt": "{:+.2f}",
                        "Contribution $": "${:+,.0f}",
                        "Average Weight": "{:.2%}",
                        "End Weight": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )
    with n_col:
        st.markdown("**Top Negative Contributors**")
        if negative.empty:
            st.info("No negative contributors in this period.")
        else:
            st.dataframe(
                negative[["Ticker", "Contribution %pt", "Contribution $", "Average Weight", "End Weight"]].style.format(
                    {
                        "Contribution %pt": "{:+.2f}",
                        "Contribution $": "${:+,.0f}",
                        "Average Weight": "{:.2%}",
                        "End Weight": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

    waterfall = contrib_df.head(top_contributors).sort_values("Contribution %pt")
    if not waterfall.empty:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=waterfall["Contribution %pt"],
                    y=waterfall["Ticker"],
                    orientation="h",
                    marker_color=np.where(waterfall["Contribution %pt"] >= 0, "#2ca02c", "#d62728"),
                    name="Contribution (% points)",
                )
            ]
        )
        fig.update_layout(
            title=f"Top Absolute Stock Contributors ({start_dt.date()} to {end_dt.date()})",
            xaxis_title="Contribution (percentage points)",
            yaxis_title="Ticker",
        )
        st.plotly_chart(fig, use_container_width=True)

    gross_contrib_pct = float(period_contrib_pct.sum())
    fee_pct = float(fee_series.loc[(fee_series.index > start_dt) & (fee_series.index <= end_dt)].sum() / start_equity) if start_equity > 0 else 0.0
    approx_net_pct = gross_contrib_pct - fee_pct
    residual_pct = period_return - approx_net_pct
    st.caption(
        f"Attribution check (short-period approximation): gross stock contribution {gross_contrib_pct:+.2%}, "
        f"fees {fee_pct:.2%}, estimated net {approx_net_pct:+.2%}, actual net {period_return:+.2%}, residual {residual_pct:+.2%}."
    )


def widget_key(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def render_strategy_params(strategy_name: str, selected_tickers: List[str], prefix: str) -> Dict[str, float | int | str]:
    params: Dict[str, float | int | str] = {}

    if strategy_name in {"SMA Crossover", "EMA Crossover"}:
        params["fast_window"] = st.number_input("Fast window", min_value=2, max_value=300, value=20, step=1, key=widget_key(prefix, "ma_fast"))
        params["slow_window"] = st.number_input("Slow window", min_value=5, max_value=500, value=100, step=1, key=widget_key(prefix, "ma_slow"))
        params["band_pct"] = st.slider("Hysteresis band (%)", min_value=0.0, max_value=5.0, value=0.2, step=0.1, key=widget_key(prefix, "ma_band"))

    elif strategy_name == "MACD Trend":
        params["fast_window"] = st.number_input("MACD fast EMA", min_value=2, max_value=100, value=12, step=1, key=widget_key(prefix, "macd_fast"))
        params["slow_window"] = st.number_input("MACD slow EMA", min_value=5, max_value=200, value=26, step=1, key=widget_key(prefix, "macd_slow"))
        params["signal_window"] = st.number_input("MACD signal EMA", min_value=2, max_value=100, value=9, step=1, key=widget_key(prefix, "macd_signal"))
        params["hist_entry"] = st.slider("Histogram entry threshold", min_value=-2.0, max_value=2.0, value=0.0, step=0.01, key=widget_key(prefix, "macd_hist_entry"))
        exit_default = min(0.0, float(params["hist_entry"]))
        params["hist_exit"] = st.slider(
            "Histogram exit threshold",
            min_value=-2.0,
            max_value=float(params["hist_entry"]),
            value=float(exit_default),
            step=0.01,
            key=widget_key(prefix, "macd_hist_exit"),
        )

    elif strategy_name == "Moving Average Reversion":
        params["ma_window"] = st.number_input("Moving average window", min_value=5, max_value=300, value=20, step=1, key=widget_key(prefix, "marev_window"))
        params["entry_deviation_pct"] = st.slider(
            "Entry below MA (%)",
            min_value=0.5,
            max_value=20.0,
            value=3.0,
            step=0.1,
            key=widget_key(prefix, "marev_entry_dev"),
        )
        params["exit_mode"] = st.selectbox("Exit mode", EXIT_MODE_OPTIONS, index=0, key=widget_key(prefix, "marev_exit_mode"))

        if params["exit_mode"] in {"Signal-Based", "Hybrid"}:
            exit_default = 0.0
            params["exit_deviation_pct"] = st.slider(
                "Exit below MA (%)",
                min_value=0.0,
                max_value=float(params["entry_deviation_pct"]),
                value=float(exit_default),
                step=0.1,
                key=widget_key(prefix, "marev_exit_dev"),
            )
        else:
            params["exit_deviation_pct"] = 0.0

        if params["exit_mode"] in {"Fixed Days", "Hybrid"}:
            params["max_hold_days"] = st.number_input(
                "Max hold days",
                min_value=1,
                max_value=90,
                value=7,
                step=1,
                key=widget_key(prefix, "marev_max_hold_days"),
            )

        use_stop_loss = st.toggle("Enable stop loss", value=False, key=widget_key(prefix, "marev_use_stop_loss"))
        if use_stop_loss:
            stop_loss_pct = st.slider("Stop loss (%)", min_value=1.0, max_value=90.0, value=5.0, step=0.5, key=widget_key(prefix, "marev_stop_loss"))
            params["stop_loss_pct"] = stop_loss_pct / 100.0
        else:
            params["stop_loss_pct"] = 0.0
            st.caption("Stop loss disabled.")

    elif strategy_name == "RSI Mean Reversion":
        params["rsi_window"] = st.number_input("RSI window", min_value=2, max_value=100, value=14, step=1, key=widget_key(prefix, "rsi_window"))
        params["oversold"] = st.slider("Oversold threshold", min_value=5, max_value=50, value=30, step=1, key=widget_key(prefix, "rsi_oversold"))
        params["exit_mode"] = st.selectbox("Exit mode", EXIT_MODE_OPTIONS, index=0, key=widget_key(prefix, "rsi_exit_mode"))

        if params["exit_mode"] in {"Signal-Based", "Hybrid"}:
            exit_rsi_min = min(95, int(params["oversold"]) + 1)
            exit_rsi_default = min(95, max(50, exit_rsi_min))
            params["exit_rsi"] = st.slider(
                "Exit RSI threshold",
                min_value=exit_rsi_min,
                max_value=95,
                value=exit_rsi_default,
                step=1,
                key=widget_key(prefix, "rsi_exit_rsi"),
            )
        else:
            params["exit_rsi"] = max(50, int(params["oversold"]) + 1)

        if params["exit_mode"] in {"Fixed Days", "Hybrid"}:
            params["max_hold_days"] = st.number_input(
                "Max hold days",
                min_value=1,
                max_value=60,
                value=5,
                step=1,
                key=widget_key(prefix, "rsi_max_hold_days"),
            )

        use_stop_loss = st.toggle("Enable stop loss", value=False, key=widget_key(prefix, "rsi_use_stop_loss"))
        if use_stop_loss:
            stop_loss_pct = st.slider("Stop loss (%)", min_value=1.0, max_value=90.0, value=5.0, step=0.5, key=widget_key(prefix, "rsi_stop_loss"))
            params["stop_loss_pct"] = stop_loss_pct / 100.0
        else:
            params["stop_loss_pct"] = 0.0
            st.caption("Stop loss disabled.")

    elif strategy_name == "Bollinger Mean Reversion":
        params["bb_window"] = st.number_input("Rolling window", min_value=5, max_value=150, value=20, step=1, key=widget_key(prefix, "bb_window"))
        params["z_entry"] = st.slider("Entry z-score", min_value=0.5, max_value=4.0, value=2.0, step=0.1, key=widget_key(prefix, "bb_entry"))
        params["exit_mode"] = st.selectbox("Exit mode", EXIT_MODE_OPTIONS, index=0, key=widget_key(prefix, "bb_exit_mode"))

        if params["exit_mode"] in {"Signal-Based", "Hybrid"}:
            z_exit_min = max(-2.0, round(-abs(float(params["z_entry"])) + 0.1, 1))
            z_exit_default = max(0.0, z_exit_min)
            params["z_exit"] = st.slider(
                "Exit z-score",
                min_value=float(z_exit_min),
                max_value=2.0,
                value=float(z_exit_default),
                step=0.1,
                key=widget_key(prefix, "bb_exit"),
            )
        else:
            params["z_exit"] = 0.0

        if params["exit_mode"] in {"Fixed Days", "Hybrid"}:
            params["max_hold_days"] = st.number_input(
                "Max hold days",
                min_value=1,
                max_value=60,
                value=5,
                step=1,
                key=widget_key(prefix, "bb_max_hold_days"),
            )

        use_stop_loss = st.toggle("Enable stop loss", value=False, key=widget_key(prefix, "bb_use_stop_loss"))
        if use_stop_loss:
            stop_loss_pct = st.slider("Stop loss (%)", min_value=1.0, max_value=90.0, value=5.0, step=0.5, key=widget_key(prefix, "bb_stop_loss"))
            params["stop_loss_pct"] = stop_loss_pct / 100.0
        else:
            params["stop_loss_pct"] = 0.0
            st.caption("Stop loss disabled.")

    elif strategy_name == "Donchian Breakout":
        params["entry_window"] = st.number_input("Entry window", min_value=10, max_value=300, value=55, step=1, key=widget_key(prefix, "donchian_entry"))
        params["exit_window"] = st.number_input("Exit window", min_value=5, max_value=200, value=20, step=1, key=widget_key(prefix, "donchian_exit"))

    elif strategy_name == "Time-Series Momentum":
        params["lookback"] = st.number_input("Lookback days", min_value=5, max_value=300, value=126, step=1, key=widget_key(prefix, "tsmom_lookback"))
        params["entry_threshold"] = st.slider("Entry trailing return", min_value=-0.10, max_value=0.30, value=0.03, step=0.01, key=widget_key(prefix, "tsmom_entry"))
        exit_default = min(0.0, float(params["entry_threshold"]))
        params["exit_threshold"] = st.slider(
            "Exit trailing return",
            min_value=-0.30,
            max_value=float(params["entry_threshold"]),
            value=float(exit_default),
            step=0.01,
            key=widget_key(prefix, "tsmom_exit"),
        )
        selected_freq = st.selectbox("Rebalance frequency", list(REBALANCE_FREQ_MAP.keys()), index=2, key=widget_key(prefix, "tsmom_rebalance"))
        params["rebalance_freq"] = REBALANCE_FREQ_MAP[selected_freq]
        st.caption("Entry is evaluated on rebalance dates; exit threshold is checked daily.")

    elif strategy_name in {"Cross-Sectional Momentum", "Cross-Sectional Mean Reversion"}:
        params["lookback"] = st.number_input("Lookback days", min_value=5, max_value=300, value=63, step=1, key=widget_key(prefix, "cross_lookback"))
        params["top_n"] = st.number_input(
            "Number of holdings",
            min_value=1,
            max_value=max(1, len(selected_tickers)),
            value=min(5, max(1, len(selected_tickers))),
            step=1,
            key=widget_key(prefix, "cross_top_n"),
        )
        selected_freq = st.selectbox("Rebalance frequency", list(REBALANCE_FREQ_MAP.keys()), index=2, key=widget_key(prefix, "cross_rebalance"))
        params["rebalance_freq"] = REBALANCE_FREQ_MAP[selected_freq]

    elif strategy_name == "Dual Momentum":
        params["lookback"] = st.number_input("Lookback days", min_value=5, max_value=300, value=126, step=1, key=widget_key(prefix, "dual_lookback"))
        params["top_n"] = st.number_input(
            "Number of holdings",
            min_value=1,
            max_value=max(1, len(selected_tickers)),
            value=min(5, max(1, len(selected_tickers))),
            step=1,
            key=widget_key(prefix, "dual_top_n"),
        )
        params["absolute_threshold"] = st.slider(
            "Absolute momentum threshold",
            min_value=-0.20,
            max_value=0.50,
            value=0.00,
            step=0.01,
            key=widget_key(prefix, "dual_abs_threshold"),
        )
        selected_freq = st.selectbox("Rebalance frequency", list(REBALANCE_FREQ_MAP.keys()), index=2, key=widget_key(prefix, "dual_rebalance"))
        params["rebalance_freq"] = REBALANCE_FREQ_MAP[selected_freq]

    elif strategy_name == "Volatility-Adjusted Momentum":
        params["lookback"] = st.number_input("Momentum lookback days", min_value=10, max_value=300, value=126, step=1, key=widget_key(prefix, "vamom_lookback"))
        params["vol_window"] = st.number_input("Volatility window", min_value=10, max_value=252, value=60, step=1, key=widget_key(prefix, "vamom_vol_window"))
        params["top_n"] = st.number_input(
            "Number of holdings",
            min_value=1,
            max_value=max(1, len(selected_tickers)),
            value=min(5, max(1, len(selected_tickers))),
            step=1,
            key=widget_key(prefix, "vamom_top_n"),
        )
        params["min_return"] = st.slider(
            "Minimum raw momentum return",
            min_value=-0.20,
            max_value=0.50,
            value=0.00,
            step=0.01,
            key=widget_key(prefix, "vamom_min_return"),
        )
        selected_freq = st.selectbox("Rebalance frequency", list(REBALANCE_FREQ_MAP.keys()), index=2, key=widget_key(prefix, "vamom_rebalance"))
        params["rebalance_freq"] = REBALANCE_FREQ_MAP[selected_freq]

    elif strategy_name == "52-Week High Rotation":
        params["lookback"] = st.number_input("High lookback days", min_value=63, max_value=504, value=252, step=1, key=widget_key(prefix, "highrot_lookback"))
        params["top_n"] = st.number_input(
            "Number of holdings",
            min_value=1,
            max_value=max(1, len(selected_tickers)),
            value=min(5, max(1, len(selected_tickers))),
            step=1,
            key=widget_key(prefix, "highrot_top_n"),
        )
        params["min_high_ratio"] = st.slider(
            "Minimum price / trailing high",
            min_value=0.70,
            max_value=1.00,
            value=0.90,
            step=0.01,
            key=widget_key(prefix, "highrot_min_ratio"),
        )
        selected_freq = st.selectbox("Rebalance frequency", list(REBALANCE_FREQ_MAP.keys()), index=2, key=widget_key(prefix, "highrot_rebalance"))
        params["rebalance_freq"] = REBALANCE_FREQ_MAP[selected_freq]

    elif strategy_name == "Inverse Volatility (Risk Parity Lite)":
        params["vol_window"] = st.number_input("Volatility window", min_value=10, max_value=252, value=60, step=1, key=widget_key(prefix, "invvol_window"))
        selected_freq = st.selectbox("Rebalance frequency", list(REBALANCE_FREQ_MAP.keys()), index=2, key=widget_key(prefix, "invvol_rebalance"))
        params["rebalance_freq"] = REBALANCE_FREQ_MAP[selected_freq]

    elif strategy_name == "Low Volatility Rotation":
        params["vol_window"] = st.number_input("Volatility window", min_value=10, max_value=252, value=60, step=1, key=widget_key(prefix, "lowvol_window"))
        params["top_n"] = st.number_input(
            "Number of holdings",
            min_value=1,
            max_value=max(1, len(selected_tickers)),
            value=min(5, max(1, len(selected_tickers))),
            step=1,
            key=widget_key(prefix, "lowvol_top_n"),
        )
        selected_freq = st.selectbox("Rebalance frequency", list(REBALANCE_FREQ_MAP.keys()), index=2, key=widget_key(prefix, "lowvol_rebalance"))
        params["rebalance_freq"] = REBALANCE_FREQ_MAP[selected_freq]

    return params


def render_fundamental_filter_controls(prefix: str) -> Tuple[Dict[str, Dict[str, float]], float]:
    metric_filters: Dict[str, Dict[str, float]] = {}
    enable_filter = st.toggle(
        "Filter by fundamentals (P/E, P/B, EPS)",
        value=False,
        key=widget_key(prefix, "enable_fundamental_filter"),
    )
    if not enable_filter:
        return metric_filters, DEFAULT_FUNDAMENTAL_MIN_COVERAGE

    min_coverage_pct = st.slider(
        "Min metric coverage to apply filter (%)",
        min_value=50,
        max_value=100,
        value=int(DEFAULT_FUNDAMENTAL_MIN_COVERAGE * 100),
        step=5,
        key=widget_key(prefix, "fundamental_min_coverage_pct"),
    )
    st.caption("If enabled metrics are below this coverage threshold, fundamental filtering is skipped for reliability.")

    use_trailing_pe = st.checkbox("Apply trailing P/E range", value=False, key=widget_key(prefix, "use_trailing_pe_filter"))
    if use_trailing_pe:
        trailing_pe_min = st.number_input(
            "Trailing P/E min",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=0.5,
            key=widget_key(prefix, "trailing_pe_min"),
        )
        trailing_pe_max = st.number_input(
            "Trailing P/E max",
            min_value=0.0,
            max_value=500.0,
            value=30.0,
            step=0.5,
            key=widget_key(prefix, "trailing_pe_max"),
        )
        if float(trailing_pe_max) < float(trailing_pe_min):
            trailing_pe_min, trailing_pe_max = trailing_pe_max, trailing_pe_min
        metric_filters["trailing_pe"] = {"min": float(trailing_pe_min), "max": float(trailing_pe_max)}

    use_price_to_book = st.checkbox("Apply max price/book", value=False, key=widget_key(prefix, "use_price_to_book_filter"))
    if use_price_to_book:
        price_to_book_max = st.number_input(
            "Price/Book max",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=0.1,
            key=widget_key(prefix, "price_to_book_max"),
        )
        metric_filters["price_to_book"] = {"max": float(price_to_book_max)}

    use_trailing_eps = st.checkbox("Apply min trailing EPS", value=False, key=widget_key(prefix, "use_trailing_eps_filter"))
    if use_trailing_eps:
        trailing_eps_min = st.number_input(
            "Trailing EPS min",
            min_value=-100.0,
            max_value=1000.0,
            value=0.0,
            step=0.1,
            key=widget_key(prefix, "trailing_eps_min"),
        )
        metric_filters["trailing_eps"] = {"min": float(trailing_eps_min)}

    if not metric_filters:
        st.caption("No fundamental thresholds selected yet.")

    return metric_filters, float(min_coverage_pct) / 100.0


def download_and_filter_prices(
    selected_tickers: List[str],
    start_date: date,
    end_date: date,
    exclude_illiquid: bool,
    min_median_dollar_volume: float,
    min_median_share_volume: float,
    liquidity_lookback_days: int,
    fundamental_metric_filters: Dict[str, Dict[str, float]] | None = None,
    min_fundamental_coverage: float = DEFAULT_FUNDAMENTAL_MIN_COVERAGE,
) -> Dict[str, Any]:
    fundamental_metric_filters = fundamental_metric_filters or {}
    result: Dict[str, Any] = {
        "prices": pd.DataFrame(),
        "missing_tickers": [],
        "limited_history_tickers": [],
        "illiquid_tickers": [],
        "no_volume_tickers": [],
        "liquidity_filter_applied": False,
        "fundamental_filter_requested": False,
        "fundamental_filter_applied": False,
        "fundamental_filtered_tickers": [],
        "fundamental_missing_tickers": [],
        "fundamental_low_coverage_metrics": [],
        "fundamental_metric_coverage": {},
    }
    prices, volumes = download_price_volume_data(tuple(selected_tickers), start_date.isoformat(), end_date.isoformat())
    if prices.empty:
        return result

    downloaded_before_filter = list(prices.columns)
    missing_tickers = sorted(set(selected_tickers) - set(downloaded_before_filter))
    result["missing_tickers"] = missing_tickers

    illiquid_tickers: List[str] = []
    no_volume_tickers: List[str] = []
    liquidity_filter_applied = False
    if exclude_illiquid and not prices.empty:
        prices, illiquid_tickers, no_volume_tickers, liquidity_filter_applied = filter_illiquid_stocks(
            prices=prices,
            volumes=volumes,
            min_median_dollar_volume=min_median_dollar_volume,
            min_median_share_volume=min_median_share_volume,
            lookback_days=liquidity_lookback_days,
        )
        if prices.empty:
            result["prices"] = pd.DataFrame()
            result["illiquid_tickers"] = illiquid_tickers
            result["no_volume_tickers"] = no_volume_tickers
            result["liquidity_filter_applied"] = liquidity_filter_applied
            return result

    result["illiquid_tickers"] = illiquid_tickers
    result["no_volume_tickers"] = no_volume_tickers
    result["liquidity_filter_applied"] = liquidity_filter_applied

    if fundamental_metric_filters and not prices.empty:
        result["fundamental_filter_requested"] = True
        fundamentals = download_fundamental_data(tuple(prices.columns))
        (
            prices,
            fundamental_filtered_tickers,
            fundamental_missing_tickers,
            fundamental_metric_coverage,
            fundamental_low_coverage_metrics,
            fundamental_filter_applied,
        ) = filter_stocks_by_fundamentals(
            prices=prices,
            fundamentals=fundamentals,
            metric_filters=fundamental_metric_filters,
            min_coverage=min_fundamental_coverage,
        )
        result["fundamental_filter_applied"] = bool(fundamental_filter_applied)
        result["fundamental_filtered_tickers"] = fundamental_filtered_tickers
        result["fundamental_missing_tickers"] = fundamental_missing_tickers
        result["fundamental_low_coverage_metrics"] = fundamental_low_coverage_metrics
        result["fundamental_metric_coverage"] = fundamental_metric_coverage

        if prices.empty:
            result["prices"] = pd.DataFrame()
            return result

    min_obs = min(len(prices), max(40, int(0.40 * len(prices))))
    positive_obs = (prices > 0).sum(axis=0)
    limited_history_tickers = sorted([ticker for ticker in prices.columns if int(positive_obs.get(ticker, 0)) < min_obs])
    result["prices"] = prices
    result["limited_history_tickers"] = limited_history_tickers
    return result


def build_live_signal_table(
    prices: pd.DataFrame,
    strategy_weights: pd.DataFrame,
    signal_scores: pd.DataFrame,
    execution_style: str = EXECUTION_STYLE_TARGET,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    latest_date = strategy_weights.index[-1]
    previous_date = strategy_weights.index[-2] if len(strategy_weights.index) > 1 else latest_date

    latest_weights = strategy_weights.loc[latest_date].fillna(0.0)
    previous_weights = strategy_weights.loc[previous_date].fillna(0.0)
    weight_change = latest_weights - previous_weights
    latest_scores = signal_scores.reindex(index=strategy_weights.index, columns=strategy_weights.columns).loc[latest_date]
    latest_scores = latest_scores.replace([np.inf, -np.inf], np.nan)

    latest_prices = prices.loc[latest_date]
    ret_5d = prices.pct_change(5, fill_method=None).loc[latest_date].replace([np.inf, -np.inf], np.nan)
    ret_20d = prices.pct_change(20, fill_method=None).loc[latest_date].replace([np.inf, -np.inf], np.nan)

    eps = 1e-8
    actions: List[str] = []
    for ticker in prices.columns:
        prev_w = float(previous_weights.at[ticker])
        curr_w = float(latest_weights.at[ticker])
        delta = float(weight_change.at[ticker])
        prev_abs = abs(prev_w)
        curr_abs = abs(curr_w)

        if prev_abs <= eps and curr_w > eps:
            action = "BUY"
        elif prev_abs <= eps and curr_w < -eps:
            action = "SHORT"
        elif prev_w > eps and curr_abs <= eps:
            action = "SELL"
        elif prev_w < -eps and curr_abs <= eps:
            action = "COVER"
        elif prev_w > eps and curr_w < -eps:
            action = "FLIP TO SHORT"
        elif prev_w < -eps and curr_w > eps:
            action = "FLIP TO LONG"
        elif execution_style == EXECUTION_STYLE_HOLD and curr_w > eps:
            action = "HOLD LONG"
        elif execution_style == EXECUTION_STYLE_HOLD and curr_w < -eps:
            action = "HOLD SHORT"
        elif curr_w > eps and delta > eps:
            action = "INCREASE LONG"
        elif curr_w > eps and delta < -eps:
            action = "REDUCE LONG"
        elif curr_w < -eps and abs(curr_w) > abs(prev_w) + eps:
            action = "INCREASE SHORT"
        elif curr_w < -eps and abs(curr_w) < abs(prev_w) - eps:
            action = "REDUCE SHORT"
        elif curr_w > eps:
            action = "HOLD LONG"
        elif curr_w < -eps:
            action = "HOLD SHORT"
        else:
            action = "FLAT"
        actions.append(action)

    signal_table = pd.DataFrame(
        {
            "Ticker": prices.columns,
            "Action": actions,
            "Price": [float(latest_prices.at[t]) if pd.notna(latest_prices.at[t]) else np.nan for t in prices.columns],
            "Signal Score": [float(latest_scores.at[t]) if pd.notna(latest_scores.at[t]) else np.nan for t in prices.columns],
            "Target Weight": latest_weights.values,
            "Previous Weight": previous_weights.values,
            "Weight Change": weight_change.values,
            "5D Return": ret_5d.values,
            "20D Return": ret_20d.values,
        }
    )

    signal_table["Abs Target Weight"] = signal_table["Target Weight"].abs()
    signal_table = signal_table.sort_values(["Abs Target Weight", "Signal Score"], ascending=[False, False]).reset_index(drop=True)
    return signal_table, latest_date, previous_date


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def format_rebalance_label(freq_code: str) -> str:
    return REBALANCE_FREQ_LABEL_MAP.get(freq_code, freq_code)


def strategy_explanation_markdown(strategy_name: str, params: Dict[str, float | int | str], universe_size: int) -> str:
    if strategy_name == "Buy & Hold Equal Weight":
        w = (1.0 / max(1, universe_size))
        return (
            f"Equal-weight baseline portfolio across **{universe_size} assets**.\n\n"
            f"- Target weight per asset: **{format_pct(w)}**\n"
            "- Rebalances implicitly as needed to restore equal weights."
        )

    if strategy_name in {"SMA Crossover", "EMA Crossover"}:
        ma_type = "SMA" if strategy_name == "SMA Crossover" else "EMA"
        fast = int(params["fast_window"])
        slow = int(params["slow_window"])
        band = float(params["band_pct"]) / 100.0
        return (
            f"{ma_type} trend-following crossover.\n\n"
            f"- Indicator: **{ma_type}({fast})** vs **{ma_type}({slow})**\n"
            f"- Entry rule: {ma_type}({fast}) > {ma_type}({slow}) × (1 + {format_pct(band)})\n"
            f"- Exit rule: {ma_type}({fast}) < {ma_type}({slow}) × (1 - {format_pct(band)})\n"
            "- Raw strategy signal is long-biased; position-side settings can optionally enable short exposure."
        )

    if strategy_name == "MACD Trend":
        fast = int(params["fast_window"])
        slow = int(params["slow_window"])
        signal = int(params["signal_window"])
        entry = float(params["hist_entry"])
        exit_ = float(params["hist_exit"])
        return (
            "MACD trend strategy.\n\n"
            f"- MACD line: EMA({fast}) - EMA({slow})\n"
            f"- Signal line: EMA({signal}) of MACD line\n"
            "- Histogram: MACD - Signal\n"
            f"- Entry rule: Histogram >= **{entry:.2f}**\n"
            f"- Exit rule: Histogram <= **{exit_:.2f}**\n"
            "- Raw strategy signal is long-biased; position-side settings can optionally enable short exposure."
        )

    if strategy_name == "Moving Average Reversion":
        window = int(params["ma_window"])
        entry_dev = float(params["entry_deviation_pct"]) / 100.0
        exit_mode = str(params["exit_mode"])
        stop_loss = float(params.get("stop_loss_pct", 0.0))
        lines = [
            "Moving-average pullback mean-reversion strategy.",
            "",
            f"- Baseline: SMA({window})",
            f"- Entry rule: price <= SMA × (1 - {format_pct(entry_dev)})",
            f"- Exit mode: **{exit_mode}**",
        ]
        if exit_mode in {"Signal-Based", "Hybrid"}:
            exit_dev = float(params["exit_deviation_pct"]) / 100.0
            lines.append(f"- Signal exit: price >= SMA × (1 - {format_pct(exit_dev)})")
        if exit_mode in {"Fixed Days", "Hybrid"}:
            lines.append(f"- Time exit: hold max **{int(params['max_hold_days'])}** trading days")
        if stop_loss > 0:
            lines.append(f"- Stop loss: exit if trade drops by **{format_pct(stop_loss)}** from entry")
        else:
            lines.append("- Stop loss: disabled")
        lines.append("- Raw strategy signal is long-biased; position-side settings can optionally enable short exposure.")
        return "\n".join(lines)

    if strategy_name == "RSI Mean Reversion":
        window = int(params["rsi_window"])
        oversold = float(params["oversold"])
        exit_mode = str(params["exit_mode"])
        stop_loss = float(params.get("stop_loss_pct", 0.0))
        lines = [
            "RSI mean-reversion strategy.",
            "",
            f"- RSI period: **{window}**",
            f"- Entry rule: RSI < **{oversold:.0f}**",
            f"- Exit mode: **{exit_mode}**",
        ]
        if exit_mode in {"Signal-Based", "Hybrid"}:
            lines.append(f"- Signal exit: RSI >= **{float(params['exit_rsi']):.0f}**")
        if exit_mode in {"Fixed Days", "Hybrid"}:
            lines.append(f"- Time exit: hold max **{int(params['max_hold_days'])}** trading days")
        if stop_loss > 0:
            lines.append(f"- Stop loss: exit if trade drops by **{format_pct(stop_loss)}** from entry")
        else:
            lines.append("- Stop loss: disabled")
        lines.append("- Raw strategy signal is long-biased; position-side settings can optionally enable short exposure.")
        return "\n".join(lines)

    if strategy_name == "Bollinger Mean Reversion":
        window = int(params["bb_window"])
        z_entry = float(params["z_entry"])
        exit_mode = str(params["exit_mode"])
        stop_loss = float(params.get("stop_loss_pct", 0.0))
        lines = [
            "Bollinger z-score mean-reversion strategy.",
            "",
            f"- Rolling window: **{window}**",
            f"- Entry rule: z-score <= **{-abs(z_entry):.2f}**",
            f"- Exit mode: **{exit_mode}**",
        ]
        if exit_mode in {"Signal-Based", "Hybrid"}:
            lines.append(f"- Signal exit: z-score >= **{float(params['z_exit']):.2f}**")
        if exit_mode in {"Fixed Days", "Hybrid"}:
            lines.append(f"- Time exit: hold max **{int(params['max_hold_days'])}** trading days")
        if stop_loss > 0:
            lines.append(f"- Stop loss: exit if trade drops by **{format_pct(stop_loss)}** from entry")
        else:
            lines.append("- Stop loss: disabled")
        lines.append("- Raw strategy signal is long-biased; position-side settings can optionally enable short exposure.")
        return "\n".join(lines)

    if strategy_name == "Donchian Breakout":
        entry_w = int(params["entry_window"])
        exit_w = int(params["exit_window"])
        return (
            "Donchian breakout trend strategy.\n\n"
            f"- Entry rule: close > highest close of previous **{entry_w}** days\n"
            f"- Exit rule: close < lowest close of previous **{exit_w}** days\n"
            "- Raw strategy signal is long-biased; position-side settings can optionally enable short exposure."
        )

    if strategy_name == "Time-Series Momentum":
        lookback = int(params["lookback"])
        entry = float(params["entry_threshold"])
        exit_ = float(params["exit_threshold"])
        freq = str(params["rebalance_freq"])
        return (
            "Absolute (time-series) momentum.\n\n"
            f"- Signal: trailing **{lookback}-day** return\n"
            f"- Entry rule (on rebalance dates): return >= **{format_pct(entry)}**\n"
            f"- Exit rule (checked daily): return <= **{format_pct(exit_)}**\n"
            f"- Rebalance frequency: **{freq}**"
        )

    if strategy_name == "Cross-Sectional Momentum":
        lookback = int(params["lookback"])
        top_n = int(params["top_n"])
        freq = str(params["rebalance_freq"])
        return (
            "Relative momentum ranking.\n\n"
            f"- Rank assets by **{lookback}-day** return\n"
            f"- Hold top **{top_n}** assets equal-weight\n"
            f"- Rebalance frequency: **{freq}**\n"
            "- Raw strategy signal is long-biased; position-side settings can optionally enable short exposure."
        )

    if strategy_name == "Dual Momentum":
        lookback = int(params["lookback"])
        top_n = int(params["top_n"])
        abs_thr = float(params["absolute_threshold"])
        freq = str(params["rebalance_freq"])
        return (
            "Dual momentum (relative + absolute filter).\n\n"
            f"- Rank assets by **{lookback}-day** return\n"
            f"- Absolute filter: only keep assets with return >= **{format_pct(abs_thr)}**\n"
            f"- Hold top **{top_n}** passing assets equal-weight\n"
            f"- Rebalance frequency: **{freq}**\n"
            "- If none pass the filter, portfolio stays in cash."
        )

    if strategy_name == "Volatility-Adjusted Momentum":
        lookback = int(params["lookback"])
        vol_w = int(params["vol_window"])
        top_n = int(params["top_n"])
        min_ret = float(params["min_return"])
        freq = str(params["rebalance_freq"])
        return (
            "Volatility-adjusted momentum ranking.\n\n"
            f"- Raw momentum: trailing **{lookback}-day** return\n"
            f"- Risk measure: trailing **{vol_w}-day** volatility\n"
            "- Score: momentum / volatility (higher is better)\n"
            f"- Filter: raw momentum >= **{format_pct(min_ret)}**\n"
            f"- Hold top **{top_n}** assets equal-weight\n"
            f"- Rebalance frequency: **{freq}**"
        )

    if strategy_name == "52-Week High Rotation":
        lookback = int(params["lookback"])
        top_n = int(params["top_n"])
        min_ratio = float(params["min_high_ratio"])
        freq = str(params["rebalance_freq"])
        return (
            "52-week-high proximity rotation.\n\n"
            f"- Compute trailing high over **{lookback}** days\n"
            "- Score each asset by: price / trailing high\n"
            f"- Keep assets with score >= **{min_ratio:.2f}**\n"
            f"- Hold top **{top_n}** scores equal-weight\n"
            f"- Rebalance frequency: **{freq}**"
        )

    if strategy_name == "Cross-Sectional Mean Reversion":
        lookback = int(params["lookback"])
        top_n = int(params["top_n"])
        freq = str(params["rebalance_freq"])
        return (
            "Relative mean-reversion ranking.\n\n"
            f"- Rank assets by **{lookback}-day** return\n"
            f"- Hold bottom **{top_n}** assets equal-weight\n"
            f"- Rebalance frequency: **{freq}**\n"
            "- Raw strategy signal is long-biased; position-side settings can optionally enable short exposure."
        )

    if strategy_name == "Inverse Volatility (Risk Parity Lite)":
        vol_w = int(params["vol_window"])
        freq = str(params["rebalance_freq"])
        return (
            "Inverse-volatility allocation.\n\n"
            f"- Compute trailing volatility over **{vol_w}** days\n"
            "- Weight each asset proportional to 1 / volatility\n"
            f"- Rebalance frequency: **{freq}**\n"
            "- Lower-vol assets receive higher weights."
        )

    if strategy_name == "Low Volatility Rotation":
        vol_w = int(params["vol_window"])
        top_n = int(params["top_n"])
        freq = str(params["rebalance_freq"])
        return (
            "Low-volatility rotation.\n\n"
            f"- Compute trailing volatility over **{vol_w}** days\n"
            f"- Select **{top_n}** lowest-vol assets\n"
            "- Equal-weight selected assets\n"
            f"- Rebalance frequency: **{freq}**"
        )

    return "No explanation available for this strategy."


def strategy_metric_details_markdown(strategy_name: str, params: Dict[str, float | int | str]) -> str:
    if strategy_name == "Buy & Hold Equal Weight":
        return (
            "- **Equal weight**: each asset target is `1 / number_of_assets`.\n"
            "- **Rebalance**: restores equal sizing after winners drift up and losers drift down."
        )

    if strategy_name in {"SMA Crossover", "EMA Crossover"}:
        ma_type = "SMA" if strategy_name == "SMA Crossover" else "EMA"
        return (
            f"- **{ma_type}(fast)** and **{ma_type}(slow)**: rolling trend filters with different speed.\n"
            "- **Hysteresis band**: extra buffer before flips to reduce noisy signal churn.\n"
            "- This strategy is fundamentally using **trend slope and persistence**."
        )

    if strategy_name == "MACD Trend":
        return (
            "- **MACD line**: difference between fast and slow EMA (short vs long trend).\n"
            "- **Signal line**: EMA of MACD line (smoothed momentum).\n"
            "- **Histogram**: MACD minus signal; proxy for momentum acceleration/deceleration."
        )

    if strategy_name == "Moving Average Reversion":
        window = int(params["ma_window"])
        return (
            f"- **SMA({window})**: short-term fair-value anchor.\n"
            "- **Deviation (%)**: `(price / SMA) - 1`.\n"
            "- Entry looks for large negative deviation; exits on partial/full mean normalization."
        )

    if strategy_name == "RSI Mean Reversion":
        window = int(params["rsi_window"])
        return (
            f"- **RSI({window})**: momentum oscillator from 0 to 100.\n"
            "- Lower RSI means stronger recent downside pressure vs upside pressure.\n"
            "- Oversold threshold triggers entry; recovery threshold/time/stop controls exit."
        )

    if strategy_name == "Bollinger Mean Reversion":
        window = int(params["bb_window"])
        return (
            f"- **Rolling mean/std ({window})** define a dynamic baseline and dispersion.\n"
            "- **z-score** = `(price - rolling_mean) / rolling_std`.\n"
            "- Negative extreme z-score triggers entry; normalization z-score triggers exit."
        )

    if strategy_name == "Donchian Breakout":
        entry_w = int(params["entry_window"])
        exit_w = int(params["exit_window"])
        return (
            f"- **Entry channel ({entry_w})**: highest close over prior window.\n"
            f"- **Exit channel ({exit_w})**: lowest close over prior window.\n"
            "- Breakout above channel signals trend start; break below exit channel signals failure."
        )

    if strategy_name in {"Time-Series Momentum", "Cross-Sectional Momentum", "Cross-Sectional Mean Reversion", "Dual Momentum"}:
        lookback = int(params["lookback"])
        base = [f"- **Trailing return ({lookback} days)** = `price_t / price_(t-lookback) - 1`."]
        if "rebalance_freq" in params:
            freq_code = str(params["rebalance_freq"])
            base.append(f"- **Rebalance frequency**: {format_rebalance_label(freq_code)} ({freq_code}).")
        if strategy_name == "Time-Series Momentum":
            base.append("- Uses each asset's own trailing return for entry/exit decisions.")
        elif strategy_name == "Cross-Sectional Momentum":
            base.append("- Ranks assets by trailing return and buys the strongest cohort.")
        elif strategy_name == "Cross-Sectional Mean Reversion":
            base.append("- Ranks assets by trailing return and buys the weakest cohort.")
        else:
            base.append("- Adds an absolute momentum cutoff before ranking (relative + absolute filter).")
        return "\n".join(base)

    if strategy_name == "Volatility-Adjusted Momentum":
        lookback = int(params["lookback"])
        vol_window = int(params["vol_window"])
        return (
            f"- **Raw momentum**: trailing return over {lookback} days.\n"
            f"- **Volatility**: rolling std of daily returns over {vol_window} days (annualized in score).\n"
            "- **Score**: momentum / volatility (risk-adjusted momentum quality)."
        )

    if strategy_name == "52-Week High Rotation":
        lookback = int(params["lookback"])
        return (
            f"- **Trailing high ({lookback} days)**: recent local maximum price.\n"
            "- **High-ratio**: `price / trailing_high`.\n"
            "- Higher ratio means price is closer to highs, interpreted as relative strength."
        )

    if strategy_name in {"Inverse Volatility (Risk Parity Lite)", "Low Volatility Rotation"}:
        vol_window = int(params["vol_window"])
        if strategy_name == "Inverse Volatility (Risk Parity Lite)":
            return (
                f"- **Volatility ({vol_window} days)** from daily return dispersion.\n"
                "- **Weight rule**: `w_i ∝ 1 / vol_i` then normalize to sum to 100%.\n"
                "- Lower-vol assets receive higher portfolio weight."
            )
        return (
            f"- **Volatility ({vol_window} days)** from daily return dispersion.\n"
            "- Select the lowest-volatility names, then equal-weight that subset.\n"
            "- Pure defensive selection, not trend selection."
        )

    return "- No additional metric details available."


def strategy_hypothesis_markdown(strategy_name: str) -> str:
    if strategy_name == "Buy & Hold Equal Weight":
        return (
            "**Why it might work:** captures broad equity risk premium over time.\n\n"
            "**What it tries to capture:** long-run economic growth + diversification rebalancing.\n\n"
            "**Main failure modes:** full market drawdowns, no regime filter."
        )

    if strategy_name in {"SMA Crossover", "EMA Crossover", "MACD Trend", "Donchian Breakout", "Time-Series Momentum"}:
        return (
            "**Why it might work:** prices can trend because information diffuses gradually and positioning adjusts slowly.\n\n"
            "**What it tries to capture:** medium-term continuation and sustained directional moves.\n\n"
            "**Main failure modes:** whipsaws/chop, sudden reversals after prolonged trend."
        )

    if strategy_name in {"Moving Average Reversion", "RSI Mean Reversion", "Bollinger Mean Reversion", "Cross-Sectional Mean Reversion"}:
        return (
            "**Why it might work:** short-term moves can overshoot due to panic, liquidity shocks, and crowded de-risking.\n\n"
            "**What it tries to capture:** snapback toward normal valuation/momentum after temporary dislocation.\n\n"
            "**Main failure modes:** catching persistent downtrends where 'cheap' keeps getting cheaper."
        )

    if strategy_name in {"Cross-Sectional Momentum", "Dual Momentum", "Volatility-Adjusted Momentum", "52-Week High Rotation"}:
        return (
            "**Why it might work:** relative winners often keep outperforming for a period due to underreaction and herding.\n\n"
            "**What it tries to capture:** leadership persistence across assets, optionally filtered for quality/risk.\n\n"
            "**Main failure modes:** momentum crashes when leadership rotates abruptly."
        )

    if strategy_name in {"Inverse Volatility (Risk Parity Lite)", "Low Volatility Rotation"}:
        return (
            "**Why it might work:** lower-vol assets can deliver better risk-adjusted returns and smoother compounding.\n\n"
            "**What it tries to capture:** defensive risk profile and reduced concentration in high-vol names.\n\n"
            "**Main failure modes:** underperformance during strong high-beta rallies."
        )

    return "No additional theoretical notes available."


def portfolio_construction_markdown(
    position_side: str,
    weighting_scheme: str,
    max_holdings: int,
    weighting_vol_window: int,
    max_single_position_cap_pct: float,
) -> str:
    lines = [
        f"- **Position side**: {position_side}.",
        f"- **Max holdings cap**: at most {int(max_holdings)} tickers are held simultaneously.",
        "- **Selection rule when cap binds**: the app ranks candidates by signal strength and keeps the strongest signals.",
        "- **No leverage execution**: buy orders are cash-constrained each day after accounting for same-day sells and transaction fees.",
        "- **Whole-share execution**: orders are rounded to whole shares (no fractional-share trades).",
    ]
    if position_side == "Long/Short":
        lines.append("- **Gross exposure split**: 50% long and 50% short when both sides have valid signals.")
    if weighting_scheme == "Equal Weight":
        lines.append("- **Weighting**: selected positions are equal-weighted.")
    elif weighting_scheme == "Signal Strength":
        lines.append("- **Weighting**: larger signal magnitude gets larger portfolio weight.")
    elif weighting_scheme == "Hold Until Sell (No Rebalance)":
        lines.append(
            "- **Weighting**: enter on signal, then hold shares without routine rebalancing until an exit signal appears."
        )
        lines.append(
            f"- **Single-name risk cap**: each position is capped at **{float(max_single_position_cap_pct):.1f}%** of portfolio value."
        )
    else:
        lines.append(
            f"- **Weighting**: signal magnitude adjusted by trailing volatility "
            f"(lookback = {int(weighting_vol_window)} trading days)."
        )
    return "\n".join(lines)


def render_backtest_page() -> None:
    with st.sidebar:
        st.header("1) Universe")
        source = st.radio("Universe source", ["Preset", "Index Constituents", "Custom"], horizontal=True, key=widget_key("bt", "universe_source"))

        selected_tickers: List[str]
        if source == "Preset":
            preset_name = st.selectbox("Preset universe", list(PRESET_UNIVERSES.keys()), key=widget_key("bt", "preset"))
            selected_tickers = PRESET_UNIVERSES[preset_name].copy()
            extras = parse_tickers(st.text_input("Optional extra tickers", "", key=widget_key("bt", "extras_preset")))
            selected_tickers.extend(extras)
        elif source == "Index Constituents":
            index_name = st.selectbox("Index universe", INDEX_UNIVERSE_OPTIONS, key=widget_key("bt", "index"))
            try:
                selected_tickers = load_index_universe(index_name)
            except Exception as exc:
                st.error(f"Failed to load {index_name} constituents: {exc}")
                selected_tickers = []
            extras = parse_tickers(st.text_input("Optional extra tickers", "", key=widget_key("bt", "extras_index")))
            selected_tickers.extend(extras)
        else:
            custom_text = st.text_area(
                "Tickers (comma / space separated)",
                value="AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, XOM, UNH, JNJ",
                height=110,
                key=widget_key("bt", "custom"),
            )
            selected_tickers = parse_tickers(custom_text)

        selected_tickers = normalize_ticker_list(selected_tickers)
        excluded_tickers_input = parse_tickers(
            st.text_input(
                "Tickers to exclude/ignore (comma / space separated)",
                value="",
                key=widget_key("bt", "exclude_tickers"),
            )
        )
        if excluded_tickers_input:
            exclude_set = set(excluded_tickers_input)
            selected_tickers = [ticker for ticker in selected_tickers if ticker not in exclude_set]
            st.caption(f"Manually excluded: {len(excluded_tickers_input)} ticker(s)")

        st.caption(f"Selected: {len(selected_tickers)} ticker(s)")
        if source == "Index Constituents":
            st.caption("Large universes are slower, but strategies scan the full selected universe.")

        st.header("2) Data & Costs")
        default_start = date.today() - timedelta(days=365 * 5)
        start_date = st.date_input(
            "Start date",
            value=default_start,
            min_value=EARLIEST_DATE_INPUT,
            max_value=date.today(),
            key=widget_key("bt", "start_date"),
        )
        end_date = st.date_input(
            "End date",
            value=date.today(),
            min_value=EARLIEST_DATE_INPUT,
            max_value=date.today(),
            key=widget_key("bt", "end_date"),
        )
        initial_capital = st.number_input("Initial capital ($)", min_value=1_000.0, value=100_000.0, step=1_000.0, key=widget_key("bt", "capital"))
        fee_bps = st.number_input("Transaction cost (bps per 1.0 turnover)", min_value=0.0, max_value=100.0, value=2.0, step=0.5, key=widget_key("bt", "fee_bps"))
        exclude_illiquid = st.toggle("Exclude very illiquid stocks", value=True, key=widget_key("bt", "exclude_illiquid"))
        if exclude_illiquid:
            min_median_dollar_volume_m = st.number_input(
                "Min median dollar volume ($M)",
                min_value=0.1,
                max_value=500.0,
                value=2.0,
                step=0.1,
                key=widget_key("bt", "min_dollar_vol_m"),
            )
            min_median_share_volume_m = st.number_input(
                "Min median shares traded (M)",
                min_value=0.0,
                max_value=1000.0,
                value=1.0,
                step=0.1,
                key=widget_key("bt", "min_share_vol_m"),
            )
            liquidity_lookback_days = st.number_input(
                "Liquidity lookback days",
                min_value=20,
                max_value=252,
                value=60,
                step=1,
                key=widget_key("bt", "liquidity_lookback"),
            )
        else:
            min_median_dollar_volume_m = 2.0
            min_median_share_volume_m = 1.0
            liquidity_lookback_days = 60
        fundamental_metric_filters, min_fundamental_coverage = render_fundamental_filter_controls("bt")
        st.caption("Backtest uses a 1-trading-day execution lag (signals from t-1 executed on t) to reduce lookahead bias.")
        st.caption("Safety rule: strategy signals and executions ignore NaN/zero-price assets.")
        st.caption("No leverage rule: buys are cash-constrained each day (cannot spend more than available cash after same-day sells and fees).")
        st.caption("Execution uses whole shares only (no fractional-share trading).")

        st.header("3) Strategy")
        strategy_name = st.selectbox("Strategy", list(STRATEGY_DESCRIPTIONS.keys()), key=widget_key("bt", "strategy"))
        st.caption(STRATEGY_DESCRIPTIONS[strategy_name])
        params = render_strategy_params(strategy_name, selected_tickers, prefix="bt")
        position_side = st.selectbox("Position side", POSITION_SIDE_OPTIONS, index=0, key=widget_key("bt", "position_side"))
        weighting_scheme = st.selectbox("Holding weighting", WEIGHTING_SCHEME_OPTIONS, index=0, key=widget_key("bt", "weighting_scheme"))
        if weighting_scheme == "Signal Strength / Volatility":
            weighting_vol_window = st.number_input(
                "Weighting volatility lookback (days)",
                min_value=5,
                max_value=252,
                value=20,
                step=1,
                key=widget_key("bt", "weighting_vol_window"),
            )
        else:
            weighting_vol_window = 20
        if weighting_scheme == "Hold Until Sell (No Rebalance)":
            max_single_position_cap_pct = st.slider(
                "Max single-position cap (%)",
                min_value=1.0,
                max_value=100.0,
                value=25.0,
                step=1.0,
                key=widget_key("bt", "max_single_position_cap_pct"),
            )
            st.caption("In this mode, new positions are opened on entry signals and then left untouched until exit signals, except cap trims.")
        else:
            max_single_position_cap_pct = 100.0
        max_portfolio_holdings = st.number_input(
            "Max holdings in portfolio",
            min_value=1,
            max_value=max(1, len(selected_tickers)),
            value=max(1, len(selected_tickers)),
            step=1,
            key=widget_key("bt", "max_holdings_cap"),
        )
        st.caption("Strategy evaluates the full selected universe; this cap limits how many tickers can be held at once.")
        if position_side == "Long/Short":
            st.caption("Long/Short mode allocates gross exposure 50% long and 50% short when both sides have valid signals.")

        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True, key=widget_key("bt", "run"))

    request_signature = repr(
        {
            "source": source,
            "tickers": tuple(selected_tickers),
            "excluded_tickers": tuple(excluded_tickers_input),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": float(initial_capital),
            "fee_bps": float(fee_bps),
            "exclude_illiquid": bool(exclude_illiquid),
            "min_median_dollar_volume_m": float(min_median_dollar_volume_m),
            "min_median_share_volume_m": float(min_median_share_volume_m),
            "liquidity_lookback_days": int(liquidity_lookback_days),
            "fundamental_metric_filters": tuple(
                (metric, tuple(sorted((str(k), float(v)) for k, v in bounds.items())))
                for metric, bounds in sorted(fundamental_metric_filters.items())
            ),
            "min_fundamental_coverage": float(min_fundamental_coverage),
            "position_side": position_side,
            "weighting_scheme": weighting_scheme,
            "weighting_vol_window": int(weighting_vol_window),
            "max_single_position_cap_pct": float(max_single_position_cap_pct),
            "max_portfolio_holdings": int(max_portfolio_holdings),
            "strategy_name": strategy_name,
            "params": tuple(sorted((str(k), str(v)) for k, v in params.items())),
        }
    )
    bt_result_key = "bt_last_result_bundle"
    bt_signature_key = "bt_last_request_signature"

    result_bundle = st.session_state.get(bt_result_key)
    latest_signature = st.session_state.get(bt_signature_key)

    if run_clicked:
        valid_inputs = True
        if not selected_tickers:
            st.error("Please select at least one ticker.")
            valid_inputs = False
        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
            valid_inputs = False

        if valid_inputs:
            with st.spinner("Downloading historical prices..."):
                download_result = download_and_filter_prices(
                    selected_tickers=selected_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    exclude_illiquid=exclude_illiquid,
                    min_median_dollar_volume=float(min_median_dollar_volume_m) * 1_000_000.0,
                    min_median_share_volume=float(min_median_share_volume_m) * 1_000_000.0,
                    liquidity_lookback_days=int(liquidity_lookback_days),
                    fundamental_metric_filters=fundamental_metric_filters,
                    min_fundamental_coverage=float(min_fundamental_coverage),
                )
            prices = download_result["prices"]
            missing_tickers = download_result["missing_tickers"]
            limited_history_tickers = download_result["limited_history_tickers"]
            illiquid_tickers = download_result["illiquid_tickers"]
            no_volume_tickers = download_result["no_volume_tickers"]
            liquidity_filter_applied = bool(download_result["liquidity_filter_applied"])
            fundamental_filter_requested = bool(download_result["fundamental_filter_requested"])
            fundamental_filter_applied = bool(download_result["fundamental_filter_applied"])
            fundamental_filtered_tickers = download_result["fundamental_filtered_tickers"]
            fundamental_missing_tickers = download_result["fundamental_missing_tickers"]
            fundamental_low_coverage_metrics = download_result["fundamental_low_coverage_metrics"]
            fundamental_metric_coverage = download_result["fundamental_metric_coverage"]
            if prices.empty:
                st.error(
                    "No valid price data returned after applying filters. "
                    "Try a broader universe/date range or relax liquidity/fundamental filters."
                )
                valid_inputs = False

        if valid_inputs:
            try:
                base_long_weights = build_strategy_weights(strategy_name, prices, params)
                signal_scores = build_strategy_signal_strength(strategy_name, prices, params)
                strategy_weights = construct_portfolio_weights(
                    base_long_weights=base_long_weights,
                    signal_scores=signal_scores,
                    prices=prices,
                    position_side=position_side,
                    weighting_scheme=weighting_scheme,
                    max_holdings=int(max_portfolio_holdings),
                    weighting_vol_window=int(weighting_vol_window),
                )
            except ValueError as exc:
                st.error(str(exc))
                valid_inputs = False

        if valid_inputs:
            execution_style = get_execution_style_from_weighting(weighting_scheme)
            strategy_result = run_backtest(
                prices,
                strategy_weights,
                initial_capital,
                fee_bps,
                record_trade_log=True,
                record_position_weights=True,
                execution_style=execution_style,
                max_single_position_weight=float(max_single_position_cap_pct) / 100.0,
            )
            benchmark_weights = equal_weight_positions(prices.where(prices > 0))
            benchmark_equal_result = run_backtest(
                prices,
                benchmark_weights,
                initial_capital,
                fee_bps=0.0,
                record_trade_log=False,
                record_position_weights=False,
                execution_style=EXECUTION_STYLE_TARGET,
            )
            benchmark_results: Dict[str, Dict[str, pd.Series]] = {
                "Equal-Weight Universe Benchmark": {
                    "returns": benchmark_equal_result["returns"],
                    "equity_curve": benchmark_equal_result["equity_curve"],
                }
            }
            benchmark_notes: List[str] = []

            market_cap_benchmark, market_cap_note = build_universe_market_cap_hold_benchmark(
                prices=prices,
                initial_capital=float(initial_capital),
            )
            if market_cap_benchmark is not None:
                benchmark_results["Market-Cap Universe Buy & Hold (Initial Weights)"] = market_cap_benchmark
            if market_cap_note:
                benchmark_notes.append(market_cap_note)

            benchmark_results.update(
                build_standard_benchmark_results(
                    reference_index=prices.index,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=float(initial_capital),
                )
            )
            if "S&P 500 (SPY)" not in benchmark_results or "NASDAQ-100 (QQQ)" not in benchmark_results:
                benchmark_notes.append("Some standard index benchmarks (SPY/QQQ) were unavailable for the selected range.")

            result_bundle = {
                "prices": prices,
                "missing_tickers": missing_tickers,
                "limited_history_tickers": limited_history_tickers,
                "illiquid_tickers": illiquid_tickers,
                "no_volume_tickers": no_volume_tickers,
                "liquidity_filter_applied": liquidity_filter_applied,
                "exclude_illiquid": exclude_illiquid,
                "min_median_dollar_volume_m": float(min_median_dollar_volume_m),
                "min_median_share_volume_m": float(min_median_share_volume_m),
                "liquidity_lookback_days": int(liquidity_lookback_days),
                "fundamental_metric_filters": fundamental_metric_filters,
                "min_fundamental_coverage": float(min_fundamental_coverage),
                "fundamental_filter_requested": fundamental_filter_requested,
                "fundamental_filter_applied": fundamental_filter_applied,
                "fundamental_filtered_tickers": fundamental_filtered_tickers,
                "fundamental_missing_tickers": fundamental_missing_tickers,
                "fundamental_low_coverage_metrics": fundamental_low_coverage_metrics,
                "fundamental_metric_coverage": fundamental_metric_coverage,
                "position_side": position_side,
                "weighting_scheme": weighting_scheme,
                "weighting_vol_window": int(weighting_vol_window),
                "max_single_position_cap_pct": float(max_single_position_cap_pct),
                "max_portfolio_holdings": int(max_portfolio_holdings),
                "excluded_tickers_input": excluded_tickers_input,
                "initial_capital": float(initial_capital),
                "run_metadata": {
                    "strategy_name": strategy_name,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "requested_tickers": len(selected_tickers),
                    "used_tickers": int(prices.shape[1]),
                    "position_side": position_side,
                    "weighting_scheme": weighting_scheme,
                    "weighting_vol_window": int(weighting_vol_window),
                    "execution_style": execution_style,
                    "max_single_position_cap_pct": float(max_single_position_cap_pct),
                    "max_portfolio_holdings": int(max_portfolio_holdings),
                    "fundamental_filter_requested": bool(fundamental_filter_requested),
                    "fundamental_filter_applied": bool(fundamental_filter_applied),
                    "benchmark_count": int(len(benchmark_results)),
                },
                "strategy_result": strategy_result,
                "benchmark_results": benchmark_results,
                "benchmark_notes": benchmark_notes,
            }
            st.session_state[bt_result_key] = result_bundle
            st.session_state[bt_signature_key] = request_signature
            latest_signature = request_signature

    if result_bundle is None:
        st.info("Configure inputs in the sidebar and click Run Backtest.")
        return

    prices = result_bundle["prices"]
    missing_tickers = result_bundle["missing_tickers"]
    limited_history_tickers = result_bundle["limited_history_tickers"]
    illiquid_tickers = result_bundle.get("illiquid_tickers", [])
    no_volume_tickers = result_bundle.get("no_volume_tickers", [])
    liquidity_filter_applied = bool(result_bundle.get("liquidity_filter_applied", False))
    exclude_illiquid_for_run = bool(result_bundle.get("exclude_illiquid", False))
    min_median_dollar_volume_m_for_run = float(result_bundle.get("min_median_dollar_volume_m", 0.0))
    min_median_share_volume_m_for_run = float(result_bundle.get("min_median_share_volume_m", 0.0))
    liquidity_lookback_days_for_run = int(result_bundle.get("liquidity_lookback_days", 60))
    fundamental_metric_filters_for_run = result_bundle.get("fundamental_metric_filters", {})
    min_fundamental_coverage_for_run = float(result_bundle.get("min_fundamental_coverage", DEFAULT_FUNDAMENTAL_MIN_COVERAGE))
    fundamental_filter_requested_for_run = bool(result_bundle.get("fundamental_filter_requested", False))
    fundamental_filter_applied_for_run = bool(result_bundle.get("fundamental_filter_applied", False))
    fundamental_filtered_tickers_for_run = result_bundle.get("fundamental_filtered_tickers", [])
    fundamental_missing_tickers_for_run = result_bundle.get("fundamental_missing_tickers", [])
    fundamental_low_coverage_metrics_for_run = result_bundle.get("fundamental_low_coverage_metrics", [])
    fundamental_metric_coverage_for_run = result_bundle.get("fundamental_metric_coverage", {})
    position_side_for_run = str(result_bundle.get("position_side", "Long Only"))
    weighting_scheme_for_run = str(result_bundle.get("weighting_scheme", "Equal Weight"))
    weighting_vol_window_for_run = int(result_bundle.get("weighting_vol_window", 20))
    max_single_position_cap_pct_for_run = float(result_bundle.get("max_single_position_cap_pct", 100.0))
    max_portfolio_holdings_for_run = int(result_bundle.get("max_portfolio_holdings", max(1, len(prices.columns))))
    excluded_tickers_for_run = result_bundle.get("excluded_tickers_input", [])
    initial_capital_for_run = float(result_bundle["initial_capital"])
    strategy_result = result_bundle["strategy_result"]
    strategy_returns = strategy_result["returns"]
    strategy_equity = strategy_result["equity_curve"]
    turnover = strategy_result["turnover"]
    if "benchmark_results" in result_bundle:
        benchmark_results = result_bundle["benchmark_results"]
    else:
        benchmark_results = {
            "Equal-Weight Universe Benchmark": {
                "returns": result_bundle["benchmark_returns"],
                "equity_curve": result_bundle["benchmark_equity"],
            }
        }
    run_meta = result_bundle["run_metadata"]
    benchmark_notes = result_bundle.get("benchmark_notes", [])
    execution_style_for_run = str(run_meta.get("execution_style", get_execution_style_from_weighting(weighting_scheme_for_run)))

    if request_signature != latest_signature:
        st.info("Showing last completed backtest results. Sidebar inputs changed; click Run Backtest to refresh.")

    st.caption(
        f"Last run: {run_meta['strategy_name']} from {run_meta['start_date']} to {run_meta['end_date']} "
        f"using {run_meta['used_tickers']} of {run_meta['requested_tickers']} requested tickers; "
        f"side={position_side_for_run}, weighting={weighting_scheme_for_run}, execution={execution_style_for_run}, "
        f"max holdings cap={max_portfolio_holdings_for_run}."
    )
    if benchmark_results:
        st.caption(f"Benchmarks shown: {', '.join(benchmark_results.keys())}.")
    for note in benchmark_notes:
        st.caption(note)
    if weighting_scheme_for_run == "Signal Strength / Volatility":
        st.caption(f"Signal-volatility weighting lookback: {weighting_vol_window_for_run} trading days.")
    if weighting_scheme_for_run == "Hold Until Sell (No Rebalance)":
        st.caption(f"No daily rebalancing: positions are held until exit signals; single-name cap={max_single_position_cap_pct_for_run:.1f}%.")

    st.subheader("Universe")
    st.write(f"Using {len(prices.columns)} ticker(s) with sufficient data.")
    st.code(", ".join(prices.columns), language="text")
    if excluded_tickers_for_run:
        st.caption(
            f"Manual excludes applied ({len(excluded_tickers_for_run)}): "
            f"{', '.join(excluded_tickers_for_run[:40])}{'...' if len(excluded_tickers_for_run) > 40 else ''}"
        )
    if missing_tickers:
        st.warning(f"Skipped {len(missing_tickers)} ticker(s) with no data: {', '.join(missing_tickers)}")
    if exclude_illiquid_for_run:
        if liquidity_filter_applied:
            st.caption(
                f"Illiquidity filter ON: median dollar volume >= ${min_median_dollar_volume_m_for_run:,.1f}M "
                f"OR median shares traded >= {min_median_share_volume_m_for_run:,.1f}M "
                f"over last {liquidity_lookback_days_for_run} trading days."
            )
            if illiquid_tickers:
                st.warning(
                    f"Excluded {len(illiquid_tickers)} illiquid ticker(s): "
                    f"{', '.join(illiquid_tickers[:40])}{'...' if len(illiquid_tickers) > 40 else ''}"
                )
            if no_volume_tickers:
                st.warning(
                    f"Excluded {len(no_volume_tickers)} ticker(s) with missing volume history: "
                    f"{', '.join(no_volume_tickers[:40])}{'...' if len(no_volume_tickers) > 40 else ''}"
                )
        else:
            st.info("Illiquidity filter requested, but volume data was unavailable so no liquidity-based exclusion was applied.")
    if fundamental_filter_requested_for_run:
        filter_description = format_fundamental_filter_description(fundamental_metric_filters_for_run)
        coverage_summary = format_fundamental_coverage_summary(fundamental_metric_coverage_for_run)
        if fundamental_filter_applied_for_run:
            st.caption(
                f"Fundamental filter ON ({filter_description}) with min required metric coverage "
                f"{100.0 * float(min_fundamental_coverage_for_run):.0f}%."
            )
            if coverage_summary:
                st.caption(f"Fundamental data coverage: {coverage_summary}.")
            if fundamental_filtered_tickers_for_run:
                st.warning(
                    f"Excluded {len(fundamental_filtered_tickers_for_run)} ticker(s) by fundamental thresholds: "
                    f"{', '.join(fundamental_filtered_tickers_for_run[:40])}{'...' if len(fundamental_filtered_tickers_for_run) > 40 else ''}"
                )
            if fundamental_missing_tickers_for_run:
                st.caption(
                    f"Tickers excluded due to missing fundamental values in enabled metrics: "
                    f"{', '.join(fundamental_missing_tickers_for_run[:40])}{'...' if len(fundamental_missing_tickers_for_run) > 40 else ''}"
                )
        else:
            low_coverage_text = ", ".join(
                [
                    f"{FUNDAMENTAL_METRIC_LABELS.get(metric, metric)} "
                    f"({100.0 * float(fundamental_metric_coverage_for_run.get(metric, 0.0)):.0f}%)"
                    for metric in fundamental_low_coverage_metrics_for_run
                ]
            )
            if low_coverage_text:
                st.info(
                    f"Fundamental filter skipped for reliability: coverage below "
                    f"{100.0 * float(min_fundamental_coverage_for_run):.0f}% for {low_coverage_text}."
                )
                if coverage_summary:
                    st.caption(f"Fundamental data coverage: {coverage_summary}.")
    if limited_history_tickers:
        st.info(
            f"{len(limited_history_tickers)} ticker(s) have limited positive-price history (often IPO/new listing) and were kept in the test. "
            f"Post-last-price dates are treated as price=0 only after {DELISTING_CONFIRMATION_DAYS} consecutive missing trading days "
            f"(delisting assumption): "
            f"{', '.join(limited_history_tickers[:40])}{'...' if len(limited_history_tickers) > 40 else ''}"
        )
    else:
        st.caption(
            f"Delisting assumption enabled: if a ticker has >= {DELISTING_CONFIRMATION_DAYS} consecutive missing trading days "
            "after its last observed price, later dates are treated as price=0."
        )

    st.subheader("Performance")
    render_metrics_table(strategy_returns, strategy_equity, benchmark_results)

    equity_frame = pd.DataFrame({"Strategy": strategy_equity})
    for bench_name, bench_data in benchmark_results.items():
        equity_frame[bench_name] = bench_data["equity_curve"]
    fig_equity = go.Figure()
    for col in equity_frame.columns:
        fig_equity.add_trace(go.Scatter(x=equity_frame.index, y=equity_frame[col], mode="lines", name=col))
    fig_equity.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value")
    st.plotly_chart(fig_equity, use_container_width=True)

    drawdown_frame = pd.DataFrame({"Strategy": strategy_equity / strategy_equity.cummax() - 1.0})
    for bench_name, bench_data in benchmark_results.items():
        bench_equity = bench_data["equity_curve"]
        drawdown_frame[bench_name] = bench_equity / bench_equity.cummax() - 1.0
    fig_drawdown = go.Figure()
    for col in drawdown_frame.columns:
        fig_drawdown.add_trace(go.Scatter(x=drawdown_frame.index, y=drawdown_frame[col], mode="lines", name=col))
    fig_drawdown.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown")
    st.plotly_chart(fig_drawdown, use_container_width=True)

    st.subheader("Execution")
    turnover_summary = pd.DataFrame(
        {
            "Average Daily Turnover": [f"{turnover.mean():.2f}"],
            "Median Daily Turnover": [f"{turnover.median():.2f}"],
            "Max Daily Turnover": [f"{turnover.max():.2f}"],
        }
    )
    st.dataframe(turnover_summary, use_container_width=True)

    st.subheader("Final Portfolio")
    ending_value = float(strategy_equity.iloc[-1]) if len(strategy_equity) else initial_capital_for_run
    render_final_portfolio(strategy_result["final_portfolio"], ending_value=ending_value)

    st.subheader("Trade Log")
    render_trade_log(
        trade_log=strategy_result["trade_log"],
        trade_summary=strategy_result["trade_summary"],
        total_trade_events=int(strategy_result["total_trade_events"]),
        trade_log_truncated=bool(strategy_result["trade_log_truncated"]),
    )

    st.subheader("Extreme Move Attribution")
    render_extreme_move_attribution(
        prices=prices,
        equity_curve=strategy_equity,
        fee_series=strategy_result["fees"],
        position_weights=strategy_result["position_weights"],
    )


def render_live_signals_page() -> None:
    with st.sidebar:
        st.header("1) Universe")
        source = st.radio("Universe source", ["Preset", "Index Constituents", "Custom"], horizontal=True, key=widget_key("live", "universe_source"))

        selected_tickers: List[str]
        if source == "Preset":
            preset_name = st.selectbox("Preset universe", list(PRESET_UNIVERSES.keys()), key=widget_key("live", "preset"))
            selected_tickers = PRESET_UNIVERSES[preset_name].copy()
            extras = parse_tickers(st.text_input("Optional extra tickers", "", key=widget_key("live", "extras_preset")))
            selected_tickers.extend(extras)
        elif source == "Index Constituents":
            index_name = st.selectbox("Index universe", INDEX_UNIVERSE_OPTIONS, key=widget_key("live", "index"))
            try:
                selected_tickers = load_index_universe(index_name)
            except Exception as exc:
                st.error(f"Failed to load {index_name} constituents: {exc}")
                selected_tickers = []
            extras = parse_tickers(st.text_input("Optional extra tickers", "", key=widget_key("live", "extras_index")))
            selected_tickers.extend(extras)
        else:
            custom_text = st.text_area(
                "Tickers (comma / space separated)",
                value="AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, XOM, UNH, JNJ",
                height=110,
                key=widget_key("live", "custom"),
            )
            selected_tickers = parse_tickers(custom_text)

        selected_tickers = normalize_ticker_list(selected_tickers)
        st.caption(f"Selected: {len(selected_tickers)} ticker(s)")
        if source == "Index Constituents":
            st.caption("Large universes are slower, but strategies scan the full selected universe.")

        st.header("2) Scan Window")
        as_of_date = st.date_input(
            "As-of date",
            value=date.today(),
            min_value=EARLIEST_DATE_INPUT,
            max_value=date.today(),
            key=widget_key("live", "asof_date"),
        )
        lookback_years = st.slider("History window (years)", min_value=1, max_value=15, value=5, step=1, key=widget_key("live", "lookback_years"))
        allocation_capital = st.number_input("Allocation capital ($)", min_value=1_000.0, value=100_000.0, step=1_000.0, key=widget_key("live", "capital"))
        exclude_illiquid = st.toggle("Exclude very illiquid stocks", value=True, key=widget_key("live", "exclude_illiquid"))
        if exclude_illiquid:
            min_median_dollar_volume_m = st.number_input(
                "Min median dollar volume ($M)",
                min_value=0.1,
                max_value=500.0,
                value=2.0,
                step=0.1,
                key=widget_key("live", "min_dollar_vol_m"),
            )
            min_median_share_volume_m = st.number_input(
                "Min median shares traded (M)",
                min_value=0.0,
                max_value=1000.0,
                value=1.0,
                step=0.1,
                key=widget_key("live", "min_share_vol_m"),
            )
            liquidity_lookback_days = st.number_input(
                "Liquidity lookback days",
                min_value=20,
                max_value=252,
                value=60,
                step=1,
                key=widget_key("live", "liquidity_lookback"),
            )
        else:
            min_median_dollar_volume_m = 2.0
            min_median_share_volume_m = 1.0
            liquidity_lookback_days = 60
        fundamental_metric_filters, min_fundamental_coverage = render_fundamental_filter_controls("live")
        start_date = as_of_date - timedelta(days=int(365.25 * lookback_years))
        st.caption("Safety rule: strategy signals and executions ignore NaN/zero-price assets.")
        st.caption("No leverage rule: live execution model is cash-constrained (buys cannot exceed available cash after sells and fees).")
        st.caption("Execution model uses whole shares only (no fractional-share trading).")

        st.header("3) Strategy")
        strategy_name = st.selectbox("Strategy", list(STRATEGY_DESCRIPTIONS.keys()), key=widget_key("live", "strategy"))
        st.caption(STRATEGY_DESCRIPTIONS[strategy_name])
        params = render_strategy_params(strategy_name, selected_tickers, prefix="live")
        position_side = st.selectbox("Position side", POSITION_SIDE_OPTIONS, index=0, key=widget_key("live", "position_side"))
        weighting_scheme = st.selectbox("Holding weighting", WEIGHTING_SCHEME_OPTIONS, index=0, key=widget_key("live", "weighting_scheme"))
        if weighting_scheme == "Signal Strength / Volatility":
            weighting_vol_window = st.number_input(
                "Weighting volatility lookback (days)",
                min_value=5,
                max_value=252,
                value=20,
                step=1,
                key=widget_key("live", "weighting_vol_window"),
            )
        else:
            weighting_vol_window = 20
        if weighting_scheme == "Hold Until Sell (No Rebalance)":
            max_single_position_cap_pct = st.slider(
                "Max single-position cap (%)",
                min_value=1.0,
                max_value=100.0,
                value=25.0,
                step=1.0,
                key=widget_key("live", "max_single_position_cap_pct"),
            )
            st.caption("Live mode interpretation: entries/exits are signal-driven; existing positions are not size-managed except cap trims.")
        else:
            max_single_position_cap_pct = 100.0
        max_portfolio_holdings = st.number_input(
            "Max holdings in portfolio",
            min_value=1,
            max_value=max(1, len(selected_tickers)),
            value=max(1, len(selected_tickers)),
            step=1,
            key=widget_key("live", "max_holdings_cap"),
        )
        st.caption("Strategy evaluates the full selected universe; this cap limits how many tickers can be held at once.")
        if position_side == "Long/Short":
            st.caption("Long/Short mode allocates gross exposure 50% long and 50% short when both sides have valid signals.")

        run_scan = st.button("Run Live Scan", type="primary", use_container_width=True, key=widget_key("live", "run"))

    if not run_scan:
        st.info("Configure inputs in the sidebar and click Run Live Scan.")
        return
    if not selected_tickers:
        st.error("Please select at least one ticker.")
        return

    with st.spinner("Downloading latest historical prices..."):
        download_result = download_and_filter_prices(
            selected_tickers=selected_tickers,
            start_date=start_date,
            end_date=as_of_date,
            exclude_illiquid=exclude_illiquid,
            min_median_dollar_volume=float(min_median_dollar_volume_m) * 1_000_000.0,
            min_median_share_volume=float(min_median_share_volume_m) * 1_000_000.0,
            liquidity_lookback_days=int(liquidity_lookback_days),
            fundamental_metric_filters=fundamental_metric_filters,
            min_fundamental_coverage=float(min_fundamental_coverage),
        )
    prices = download_result["prices"]
    missing_tickers = download_result["missing_tickers"]
    limited_history_tickers = download_result["limited_history_tickers"]
    illiquid_tickers = download_result["illiquid_tickers"]
    no_volume_tickers = download_result["no_volume_tickers"]
    liquidity_filter_applied = bool(download_result["liquidity_filter_applied"])
    fundamental_filter_requested = bool(download_result["fundamental_filter_requested"])
    fundamental_filter_applied = bool(download_result["fundamental_filter_applied"])
    fundamental_filtered_tickers = download_result["fundamental_filtered_tickers"]
    fundamental_missing_tickers = download_result["fundamental_missing_tickers"]
    fundamental_low_coverage_metrics = download_result["fundamental_low_coverage_metrics"]
    fundamental_metric_coverage = download_result["fundamental_metric_coverage"]
    if prices.empty:
        st.error(
            "No valid price data returned after applying filters. "
            "Try a broader universe/date range or relax liquidity/fundamental filters."
        )
        return

    st.subheader("Universe")
    st.write(f"Using {len(prices.columns)} ticker(s) with sufficient data.")
    st.code(", ".join(prices.columns), language="text")
    if missing_tickers:
        st.warning(f"Skipped {len(missing_tickers)} ticker(s) with no data: {', '.join(missing_tickers)}")
    if exclude_illiquid:
        if liquidity_filter_applied:
            st.caption(
                f"Illiquidity filter ON: median dollar volume >= ${float(min_median_dollar_volume_m):,.1f}M "
                f"OR median shares traded >= {float(min_median_share_volume_m):,.1f}M "
                f"over last {int(liquidity_lookback_days)} trading days."
            )
            if illiquid_tickers:
                st.warning(
                    f"Excluded {len(illiquid_tickers)} illiquid ticker(s): "
                    f"{', '.join(illiquid_tickers[:40])}{'...' if len(illiquid_tickers) > 40 else ''}"
                )
            if no_volume_tickers:
                st.warning(
                    f"Excluded {len(no_volume_tickers)} ticker(s) with missing volume history: "
                    f"{', '.join(no_volume_tickers[:40])}{'...' if len(no_volume_tickers) > 40 else ''}"
                )
        else:
            st.info("Illiquidity filter requested, but volume data was unavailable so no liquidity-based exclusion was applied.")
    if fundamental_filter_requested:
        filter_description = format_fundamental_filter_description(fundamental_metric_filters)
        coverage_summary = format_fundamental_coverage_summary(fundamental_metric_coverage)
        if fundamental_filter_applied:
            st.caption(
                f"Fundamental filter ON ({filter_description}) with min required metric coverage "
                f"{100.0 * float(min_fundamental_coverage):.0f}%."
            )
            if coverage_summary:
                st.caption(f"Fundamental data coverage: {coverage_summary}.")
            if fundamental_filtered_tickers:
                st.warning(
                    f"Excluded {len(fundamental_filtered_tickers)} ticker(s) by fundamental thresholds: "
                    f"{', '.join(fundamental_filtered_tickers[:40])}{'...' if len(fundamental_filtered_tickers) > 40 else ''}"
                )
            if fundamental_missing_tickers:
                st.caption(
                    f"Tickers excluded due to missing fundamental values in enabled metrics: "
                    f"{', '.join(fundamental_missing_tickers[:40])}{'...' if len(fundamental_missing_tickers) > 40 else ''}"
                )
        else:
            low_coverage_text = ", ".join(
                [
                    f"{FUNDAMENTAL_METRIC_LABELS.get(metric, metric)} "
                    f"({100.0 * float(fundamental_metric_coverage.get(metric, 0.0)):.0f}%)"
                    for metric in fundamental_low_coverage_metrics
                ]
            )
            if low_coverage_text:
                st.info(
                    f"Fundamental filter skipped for reliability: coverage below "
                    f"{100.0 * float(min_fundamental_coverage):.0f}% for {low_coverage_text}."
                )
                if coverage_summary:
                    st.caption(f"Fundamental data coverage: {coverage_summary}.")
    if limited_history_tickers:
        st.info(
            f"{len(limited_history_tickers)} ticker(s) have limited positive-price history (often IPO/new listing). "
            f"For scan continuity, post-last-price dates are treated as price=0 only after "
            f"{DELISTING_CONFIRMATION_DAYS} consecutive missing trading days: "
            f"{', '.join(limited_history_tickers[:40])}{'...' if len(limited_history_tickers) > 40 else ''}"
        )

    try:
        base_long_weights = build_strategy_weights(strategy_name, prices, params)
        signal_scores = build_strategy_signal_strength(strategy_name, prices, params)
        strategy_weights = construct_portfolio_weights(
            base_long_weights=base_long_weights,
            signal_scores=signal_scores,
            prices=prices,
            position_side=position_side,
            weighting_scheme=weighting_scheme,
            max_holdings=int(max_portfolio_holdings),
            weighting_vol_window=int(weighting_vol_window),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    execution_style = get_execution_style_from_weighting(weighting_scheme)
    signal_table, latest_date, previous_date = build_live_signal_table(
        prices,
        strategy_weights,
        signal_scores,
        execution_style=execution_style,
    )
    signal_table["Target Value"] = signal_table["Target Weight"] * allocation_capital

    st.subheader("Live Strategy Signals")
    st.write(
        f"Signals as of **{latest_date.strftime('%Y-%m-%d')}** "
        f"(previous state: {previous_date.strftime('%Y-%m-%d')})."
    )
    st.caption(
        f"Side={position_side}; weighting={weighting_scheme}; max holdings cap={int(max_portfolio_holdings)}."
    )
    if weighting_scheme == "Signal Strength / Volatility":
        st.caption(f"Signal-volatility weighting lookback: {int(weighting_vol_window)} trading days.")
    if weighting_scheme == "Hold Until Sell (No Rebalance)":
        st.caption(f"Hold-until-sell mode active; single-name cap used for execution={float(max_single_position_cap_pct):.1f}%.")
    if latest_date.date() < as_of_date:
        st.warning(
            f"Latest available market data is from {latest_date.strftime('%Y-%m-%d')}; "
            f"selected as-of date is {as_of_date.strftime('%Y-%m-%d')}."
        )

    add_actions = ["BUY", "SHORT", "INCREASE LONG", "INCREASE SHORT", "FLIP TO LONG", "FLIP TO SHORT"]
    reduce_actions = ["SELL", "COVER", "REDUCE LONG", "REDUCE SHORT"]
    buy_now = signal_table[signal_table["Action"].isin(add_actions)].copy()
    exits = signal_table[signal_table["Action"].isin(reduce_actions)].copy()
    current_holdings = signal_table[signal_table["Abs Target Weight"] > 0].copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Add/Enter", f"{len(buy_now):,}")
    with c2:
        st.metric("Reduce/Exit", f"{len(exits):,}")
    with c3:
        st.metric("Open Targets", f"{len(current_holdings):,}")
    with c4:
        st.metric("Gross Target Exposure", f"{current_holdings['Abs Target Weight'].sum():.2%}")

    if execution_style == EXECUTION_STYLE_HOLD:
        st.info(
            "Action legend (hold-until-sell): `BUY`/`SHORT` open new positions, `SELL`/`COVER` close positions, "
            "`FLIP TO LONG/SHORT` changes side, `HOLD LONG/SHORT` keeps the existing position unchanged, `FLAT` stays uninvested."
        )
    else:
        st.info(
            "Action legend: `BUY` enters a new long, `SHORT` enters a new short, "
            "`INCREASE LONG/SHORT` increases same-side exposure, `REDUCE LONG/SHORT` trims same-side exposure, "
            "`SELL` exits a long, `COVER` exits a short, `FLIP TO LONG/SHORT` crosses directly through zero, "
            "`HOLD LONG/SHORT` keeps exposure roughly unchanged, `FLAT` stays uninvested."
        )

    tab_buy, tab_hold, tab_all = st.tabs(["Entries & Size Ups", "Current Target Portfolio", "All Signals"])
    with tab_buy:
        if buy_now.empty:
            st.info("No new entry or size-up signals right now for this strategy/universe.")
        else:
            buy_now = buy_now.sort_values(["Action", "Abs Target Weight"], ascending=[True, False])
            st.dataframe(
                buy_now.style.format(
                    {
                        "Price": "${:,.2f}",
                        "Signal Score": "{:+.4f}",
                        "Target Weight": "{:.2%}",
                        "Abs Target Weight": "{:.2%}",
                        "Previous Weight": "{:.2%}",
                        "Weight Change": "{:.2%}",
                        "Target Value": "${:,.2f}",
                        "5D Return": "{:.2%}",
                        "20D Return": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )
            st.download_button(
                "Download Entry/Increase List (CSV)",
                data=buy_now.to_csv(index=False).encode("utf-8"),
                file_name=f"entry_signals_{latest_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    with tab_hold:
        if current_holdings.empty:
            st.info("No target holdings for this strategy right now.")
        else:
            holdings = current_holdings.sort_values("Abs Target Weight", ascending=False)
            top = holdings.head(20).sort_values("Target Weight", ascending=True)
            fig_top = go.Figure()
            fig_top.add_trace(
                go.Bar(
                    x=top["Target Weight"],
                    y=top["Ticker"],
                    orientation="h",
                    marker_color=np.where(top["Target Weight"] >= 0, "#2ca02c", "#d62728"),
                    name="Target Weight",
                )
            )
            fig_top.update_layout(title="Top Target Weights (Signed)", xaxis_title="Weight", yaxis_title="Ticker")
            st.plotly_chart(fig_top, use_container_width=True)
            st.dataframe(
                holdings.style.format(
                    {
                        "Price": "${:,.2f}",
                        "Signal Score": "{:+.4f}",
                        "Target Weight": "{:.2%}",
                        "Abs Target Weight": "{:.2%}",
                        "Previous Weight": "{:.2%}",
                        "Weight Change": "{:.2%}",
                        "Target Value": "${:,.2f}",
                        "5D Return": "{:.2%}",
                        "20D Return": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

    with tab_all:
        action_options = sorted(signal_table["Action"].dropna().unique().tolist())
        default_actions = [action for action in action_options if action != "FLAT"]
        selected_actions = st.multiselect(
            "Filter actions",
            options=action_options,
            default=default_actions,
            key=widget_key("live", "action_filter"),
        )
        row_limit = st.select_slider(
            "Rows to display",
            options=[50, 100, 250, 500, 1000, "All"],
            value=250,
            key=widget_key("live", "row_limit"),
        )
        filtered = signal_table[signal_table["Action"].isin(selected_actions)].copy()
        filtered = filtered.sort_values(["Abs Target Weight", "Weight Change"], ascending=[False, False])
        if row_limit != "All":
            filtered = filtered.head(int(row_limit))

        st.caption(f"Showing {len(filtered):,} rows.")
        st.dataframe(
            filtered.style.format(
                {
                    "Price": "${:,.2f}",
                    "Signal Score": "{:+.4f}",
                    "Target Weight": "{:.2%}",
                    "Abs Target Weight": "{:.2%}",
                    "Previous Weight": "{:.2%}",
                    "Weight Change": "{:.2%}",
                    "Target Value": "${:,.2f}",
                    "5D Return": "{:.2%}",
                    "20D Return": "{:.2%}",
                }
            ),
            use_container_width=True,
        )
        st.download_button(
            "Download Signals (CSV)",
            data=signal_table.to_csv(index=False).encode("utf-8"),
            file_name=f"strategy_signals_{latest_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


def render_strategy_explainer_page() -> None:
    with st.sidebar:
        st.header("1) Universe Context")
        source = st.radio("Universe source", ["Preset", "Index Constituents", "Custom"], horizontal=True, key=widget_key("exp", "universe_source"))

        selected_tickers: List[str]
        if source == "Preset":
            preset_name = st.selectbox("Preset universe", list(PRESET_UNIVERSES.keys()), key=widget_key("exp", "preset"))
            selected_tickers = PRESET_UNIVERSES[preset_name].copy()
            extras = parse_tickers(st.text_input("Optional extra tickers", "", key=widget_key("exp", "extras_preset")))
            selected_tickers.extend(extras)
        elif source == "Index Constituents":
            index_name = st.selectbox("Index universe", INDEX_UNIVERSE_OPTIONS, key=widget_key("exp", "index"))
            try:
                selected_tickers = load_index_universe(index_name)
            except Exception as exc:
                st.error(f"Failed to load {index_name} constituents: {exc}")
                selected_tickers = []
            extras = parse_tickers(st.text_input("Optional extra tickers", "", key=widget_key("exp", "extras_index")))
            selected_tickers.extend(extras)
        else:
            custom_text = st.text_area(
                "Tickers (comma / space separated)",
                value="AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, XOM, UNH, JNJ",
                height=110,
                key=widget_key("exp", "custom"),
            )
            selected_tickers = parse_tickers(custom_text)

        selected_tickers = normalize_ticker_list(selected_tickers)
        max_selectable = max(1, min(800, len(selected_tickers)))
        default_count = min(max_selectable, max(1, min(250, len(selected_tickers))))
        max_tickers = st.slider("Context universe size", min_value=1, max_value=max_selectable, value=default_count, step=1, key=widget_key("exp", "max_tickers"))
        selected_tickers = selected_tickers[:max_tickers]
        st.caption(f"Context: {len(selected_tickers)} ticker(s)")

        st.header("2) Strategy")
        strategy_name = st.selectbox("Strategy", list(STRATEGY_DESCRIPTIONS.keys()), key=widget_key("exp", "strategy"))
        st.caption(STRATEGY_DESCRIPTIONS[strategy_name])
        params = render_strategy_params(strategy_name, selected_tickers, prefix="exp")
        position_side = st.selectbox("Position side", POSITION_SIDE_OPTIONS, index=0, key=widget_key("exp", "position_side"))
        weighting_scheme = st.selectbox("Holding weighting", WEIGHTING_SCHEME_OPTIONS, index=0, key=widget_key("exp", "weighting_scheme"))
        if weighting_scheme == "Signal Strength / Volatility":
            weighting_vol_window = st.number_input(
                "Weighting volatility lookback (days)",
                min_value=5,
                max_value=252,
                value=20,
                step=1,
                key=widget_key("exp", "weighting_vol_window"),
            )
        else:
            weighting_vol_window = 20
        if weighting_scheme == "Hold Until Sell (No Rebalance)":
            max_single_position_cap_pct = st.slider(
                "Max single-position cap (%)",
                min_value=1.0,
                max_value=100.0,
                value=25.0,
                step=1.0,
                key=widget_key("exp", "max_single_position_cap_pct"),
            )
        else:
            max_single_position_cap_pct = 100.0
        max_holdings = st.number_input(
            "Max holdings in portfolio",
            min_value=1,
            max_value=max(1, len(selected_tickers)),
            value=min(25, max(1, len(selected_tickers))),
            step=1,
            key=widget_key("exp", "max_holdings_cap"),
        )

    st.subheader("Plain-English Strategy Logic")
    st.markdown(strategy_explanation_markdown(strategy_name, params, universe_size=len(selected_tickers)))

    st.subheader("Metric Definitions")
    st.markdown(strategy_metric_details_markdown(strategy_name, params))

    st.subheader("Economic Rationale")
    st.markdown(strategy_hypothesis_markdown(strategy_name))

    st.subheader("Portfolio Construction Layer")
    st.markdown(
        portfolio_construction_markdown(
            position_side=position_side,
            weighting_scheme=weighting_scheme,
            max_holdings=int(max_holdings),
            weighting_vol_window=int(weighting_vol_window),
            max_single_position_cap_pct=float(max_single_position_cap_pct),
        )
    )

    st.subheader("Current Action Definitions")
    st.markdown(
        "- `BUY`: previously flat, now opening a long position.\n"
        "- `SHORT`: previously flat, now opening a short position.\n"
        "- `INCREASE LONG` / `REDUCE LONG`: adjusting an existing long without flipping side.\n"
        "- `INCREASE SHORT` / `REDUCE SHORT`: adjusting an existing short without flipping side.\n"
        "- `SELL`: long position goes to flat.\n"
        "- `COVER`: short position goes to flat.\n"
        "- `FLIP TO LONG` / `FLIP TO SHORT`: direct cross from short to long, or long to short.\n"
        "- `HOLD LONG` / `HOLD SHORT`: side and size stay effectively unchanged.\n"
        "- `FLAT`: not held previously and still not held."
    )

    if selected_tickers:
        preview = ", ".join(selected_tickers[:20])
        suffix = "..." if len(selected_tickers) > 20 else ""
        st.caption(f"Universe preview: {preview}{suffix}")

    st.subheader("Parameter Snapshot")
    snapshot_rows = [{"Parameter": key, "Value": value} for key, value in params.items()]
    snapshot_rows.extend(
        [
            {"Parameter": "position_side", "Value": position_side},
            {"Parameter": "weighting_scheme", "Value": weighting_scheme},
            {"Parameter": "weighting_vol_window", "Value": int(weighting_vol_window)},
            {"Parameter": "max_single_position_cap_pct", "Value": float(max_single_position_cap_pct)},
            {"Parameter": "max_holdings_cap", "Value": int(max_holdings)},
        ]
    )
    if snapshot_rows:
        st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True)
    else:
        st.info("This strategy has no tunable parameters on this page.")


def main() -> None:
    st.set_page_config(page_title="Quant Strategy Backtester", layout="wide")
    with st.sidebar:
        page = st.radio("Page", ["Backtest", "Live Signals", "Strategy Explainer"], key="app_page")

    if page == "Backtest":
        st.title("Quant Strategy Backtester")
        st.caption("Backtest quantitative strategies with configurable long-only, short-only, or long/short portfolio construction.")
        render_backtest_page()
    elif page == "Live Signals":
        st.title("Quant Strategy Live Signals")
        st.caption("Run a strategy on the selected universe and find what to buy/sell now based on current signals.")
        render_live_signals_page()
    else:
        st.title("Strategy Explainer")
        st.caption("Parameter-aware explanations of how each strategy enters, exits, and allocates capital.")
        render_strategy_explainer_page()


if __name__ == "__main__":
    main()
