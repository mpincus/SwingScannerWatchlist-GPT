#!/usr/bin/env python3
"""
Build extremes.csv (RSI-extremes only) from the FULL US watchlist.
- Input (optional): combined_watchlist.csv with column 'Ticker'
  If missing, auto-build US universe = S&P 500 + Nasdaq-100 + Dow 30.
- Output:
  1) extremes.csv  columns: Ticker,RSI14,Side,Close,AsOf,ATR20,MA50,MA200
  2) extremes.txt  one ticker per line
  3) missed_tickers.txt  any symbols that failed to fetch

US only. ETFs allowed. No sampling.
"""

import os
import sys
import time
import pathlib
from typing import List, Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- Config ----------------
RSI_PERIOD = 14
ATR_PERIOD = 20
MA1, MA2 = 50, 200
OVERSOLD = 30.0
OVERBOUGHT = 70.0

CHUNK = int(os.getenv("CHUNK", "120"))     # symbols per yfinance call
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_SLEEP = int(os.getenv("RETRY_SLEEP", "3"))

ROOT = pathlib.Path(".").resolve()
ASOF = pd.Timestamp.utcnow().tz_convert("UTC").tz_localize(None).date().isoformat()

# -------------- Helpers ---------------

def _clean_symbol(t: str) -> str:
    return (
        str(t).strip().upper()
        .replace(" ", "")
        .replace("/", "-")
        .replace(".", "-")
    )

def _read_html_table(url: str, match: str | None = None) -> pd.DataFrame:
    ua = {"User-Agent": "Mozilla/5.0 (CI/Scanner)"}
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=ua, timeout=20)
            r.raise_for_status()
            dfs = pd.read_html(r.text, match=match)
            if dfs:
                return dfs[0]
        except Exception:
            if i == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP * (i + 1))
    raise RuntimeError(f"Failed to read table: {url}")

def load_universe_from_wiki() -> List[str]:
    # S&P 500
    sp = _read_html_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp_col = "Symbol" if "Symbol" in sp.columns else sp.columns[0]
    sp_syms = {_clean_symbol(x) for x in sp[sp_col].dropna().astype(str)}

    # Nasdaq-100
    ndx = _read_html_table("https://en.wikipedia.org/wiki/Nasdaq-100", match="Ticker|Symbol")
    ndx_col = "Ticker" if "Ticker" in ndx.columns else ("Symbol" if "Symbol" in ndx.columns else ndx.columns[0])
    ndx_syms = {_clean_symbol(x) for x in ndx[ndx_col].dropna().astype(str)}

    # Dow 30
    dow = _read_html_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", match="Symbol|Ticker")
    if "Symbol" in dow.columns:
        dcol = "Symbol"
    elif "Ticker" in dow.columns:
        dcol = "Ticker"
    else:
        dcol = dow.columns[0]
    dow_syms = {_clean_symbol(x) for x in dow[dcol].dropna().astype(str)}

    syms = sorted(sp_syms | ndx_syms | dow_syms)
    return syms

def load_watchlist_csv(path: pathlib.Path) -> List[str]:
    df = pd.read_csv(path)
    # find the ticker-like column
    candidates = [c for c in df.columns if c.lower() == "ticker" or "ticker" in c.lower() or "symbol" in c.lower()]
    if not candidates:
        raise ValueError("No Ticker/Symbol column found in combined_watchlist.csv")
    col = candidates[0]
    syms = sorted({_clean_symbol(x) for x in df[col].dropna().astype(str) if str(x).strip()})
    return syms

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    # df columns: Open, High, Low, Close
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def yf_download_batch(symbols: List[str]) -> pd.DataFrame:
    tries = 0
    last_err = None
    size = CHUNK
    while tries < MAX_RETRIES:
        try:
            data = yf.download(
                symbols,
                period="12mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            return data
        except Exception as e:
            last_err = e
            tries += 1
            time.sleep(RETRY_SLEEP * tries)
    raise RuntimeError(f"yfinance batch failed: {last_err}")

# -------------- Scan ---------------

def scan_extremes(symbols: List[str]) -> tuple[pd.DataFrame, List[str]]:
    rows: list[dict] = []
    misses: list[str] = []

    for i in range(0, len(symbols), CHUNK):
        batch = symbols[i:i + CHUNK]
        # Try bulk first
        bulk_ok = True
        try:
            data = yf_download_batch(batch)
        except Exception:
            bulk_ok = False

        if bulk_ok and isinstance(data.columns, pd.MultiIndex):
            # multi-symbol frame
            names = sorted({t for t, _ in data.columns})
            for sym in names:
                try:
                    df = data[sym].dropna()
                    if df.empty or len(df) < max(RSI_PERIOD, ATR_PERIOD) + 1:
                        continue
                    rsi = compute_rsi(df["Close"]).iloc[-1]
                    if np.isnan(rsi):
                        continue
                    if rsi <= OVERSOLD or rsi >= OVERBOUGHT:
                        side = "long" if rsi <= OVERSOLD else "short"
                        atr = compute_atr(df).iloc[-1]
                        ma50 = df["Close"].rolling(MA1, min_periods=MA1).mean().iloc[-1]
                        ma200 = df["Close"].rolling(MA2, min_periods=MA2).mean().iloc[-1]
                        rows.append({
                            "Ticker": sym,
                            "RSI14": round(float(rsi), 2),
                            "Side": side,
                            "Close": round(float(df["Close"].iloc[-1]), 2),
                            "AsOf": ASOF,
                            "ATR20": round(float(atr), 2) if pd.notna(atr) else np.nan,
                            "MA50": round(float(ma50), 2) if pd.notna(ma50) else np.nan,
                            "MA200": round(float(ma200), 2) if pd.notna(ma200) else np.nan,
                        })
                except Exception:
                    misses.append(sym)
        else:
            # per-ticker salvage
            for sym in batch:
                try:
                    df = yf.download(sym, period="12mo", interval="1d", auto_adjust=False, progress=False)
                    if df.empty or len(df) < max(RSI_PERIOD, ATR_PERIOD) + 1:
                        continue
                    rsi = compute_rsi(df["Close"]).iloc[-1]
                    if np.isnan(rsi):
                        continue
                    if rsi <= OVERSOLD or rsi >= OVERBOUGHT:
                        side = "long" if rsi <= OVERSOLD else "short"
                        atr = compute_atr(df).iloc[-1]
                        ma50 = df["Close"].rolling(MA1, min_periods=MA1).mean().iloc[-1]
                        ma200 = df["Close"].rolling(MA2, min_periods=MA2).mean().iloc[-1]
                        rows.append({
                            "Ticker": sym,
                            "RSI14": round(float(rsi), 2),
                            "Side": side,
                            "Close": round(float(df["Close"].iloc[-1]), 2),
                            "AsOf": ASOF,
                            "ATR20": round(float(atr), 2) if pd.notna(atr) else np.nan,
                            "MA50": round(float(ma50), 2) if pd.notna(ma50) else np.nan,
                            "MA200": round(float(ma200), 2) if pd.notna(ma200) else np.nan,
                        })
                except Exception:
                    misses.append(sym)

    out = pd.DataFrame(rows, columns=["Ticker","RSI14","Side","Close","AsOf","ATR20","MA50","MA200"])
    out = out.dropna(subset=["RSI14"]).sort_values(["Side","Ticker"]).reset_index(drop=True)
    return out, sorted(set(misses))

# -------------- Main ---------------

def main():
    # Resolve universe
    wl = ROOT / "combined_watchlist.csv"
    if wl.exists():
        symbols = load_watchlist_csv(wl)
    else:
        symbols = load_universe_from_wiki()

    # Full scan
    df, misses = scan_extremes(symbols)

    # Outputs
    df.to_csv(ROOT / "extremes.csv", index=False)
    df["Ticker"].to_csv(ROOT / "extremes.txt", index=False, header=False)

    # Misses file for auditing
    pd.Series(misses, dtype=str).to_csv(ROOT / "missed_tickers.txt", index=False, header=False)

    # Console summary
    print(f"Watchlist count: {len(symbols)}")
    print(f"Extremes found: {len(df)}")
    print(f"Misses: {len(misses)}  -> missed_tickers.txt")

if __name__ == "__main__":
    sys.exit(main())