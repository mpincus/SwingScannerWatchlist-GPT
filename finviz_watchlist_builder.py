#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build RSI-extremes watchlist (US only, ETFs allowed) and write:
- combined_watchlist.csv  (columns: Ticker,List where List in {oversold, overbought})
- docs/combined_watchlist.txt  (tickers only)
Also writes diagnostics:
- extremes.csv, extremes.txt, missed_tickers.txt
Creates empty pullbacks.csv and breakouts.csv for workflow compatibility.

Rules: RSI(14) long if RSI <= 30, short if RSI >= 70.
Never hard-fail CI; always emit artifacts.
"""

import os
import sys
import time
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---- Config ----
RSI_PERIOD = 14
OVERSOLD = 30.0
OVERBOUGHT = 70.0
CHUNK = int(os.getenv("CHUNK", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_SLEEP = int(os.getenv("RETRY_SLEEP", "3"))
HTTP_TIMEOUT = 20
UA = {"User-Agent": "Mozilla/5.0 (Scanner/CI)"}

ROOT = pathlib.Path(".").resolve()
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
ASOF = pd.Timestamp.utcnow().tz_localize(None).date().isoformat()

# ---- Helpers ----
def clean_symbol(t: str) -> str:
    return (
        str(t).strip().upper()
        .replace(" ", "")
        .replace("/", "-")
        .replace(".", "-")
    )

def read_html_table(url: str, match: str | None = None) -> pd.DataFrame:
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=UA, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            dfs = pd.read_html(r.text, match=match)
            if dfs:
                return dfs[0]
        except Exception:
            if i == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP * (i + 1))
    raise RuntimeError("Failed to read table")

def load_universe_us() -> List[str]:
    # S&P 500
    sp = read_html_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp_col = "Symbol" if "Symbol" in sp.columns else sp.columns[0]
    # Nasdaq-100
    nd = read_html_table("https://en.wikipedia.org/wiki/Nasdaq-100", match="Ticker|Symbol")
    nd_col = "Ticker" if "Ticker" in nd.columns else ("Symbol" if "Symbol" in nd.columns else nd.columns[0])
    # Dow 30
    dj = read_html_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", match="Symbol|Ticker")
    dj_col = "Symbol" if "Symbol" in dj.columns else ("Ticker" if "Ticker" in dj.columns else dj.columns[0])
    syms = pd.concat([sp[sp_col], nd[nd_col], dj[dj_col]], ignore_index=True)
    return sorted({clean_symbol(x) for x in syms.dropna().astype(str)})

def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))

def dl_batch(symbols: List[str]) -> pd.DataFrame:
    tries = 0
    last = None
    while tries < MAX_RETRIES:
        try:
            return yf.download(
                symbols,
                period="12mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            last = e
            tries += 1
            time.sleep(RETRY_SLEEP * tries)
    raise RuntimeError(f"yfinance batch failed: {last}")

# ---- Scan ----
def scan_extremes(symbols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    misses: List[str] = []
    need = RSI_PERIOD + 1

    for i in range(0, len(symbols), CHUNK):
        batch = symbols[i:i + CHUNK]
        data = None
        try:
            data = dl_batch(batch)
        except Exception:
            data = None

        if data is not None and isinstance(data.columns, pd.MultiIndex):
            names = sorted({t for t, _ in data.columns})
            for sym in names:
                try:
                    df = data[sym].dropna()
                    if df.empty or len(df) < need:  # not enough history
                        continue
                    rv = rsi(df["Close"]).iloc[-1]
                    if np.isnan(rv) or (OVERSOLD < rv < OVERBOUGHT):
                        continue
                    side = "long" if rv <= OVERSOLD else "short"
                    rows.append({
                        "Ticker": sym,
                        "RSI14": round(float(rv), 2),
                        "Side": side,
                        "Close": round(float(df["Close"].iloc[-1]), 2),
                        "AsOf": ASOF,
                    })
                except Exception:
                    misses.append(sym)
        else:
            # salvage per ticker
            for sym in batch:
                try:
                    df = yf.download(sym, period="12mo", interval="1d", auto_adjust=False, progress=False)
                    if df.empty or len(df) < need:
                        continue
                    rv = rsi(df["Close"]).iloc[-1]
                    if np.isnan(rv) or (OVERSOLD < rv < OVERBOUGHT):
                        continue
                    side = "long" if rv <= OVERSOLD else "short"
                    rows.append({
                        "Ticker": sym,
                        "RSI14": round(float(rv), 2),
                        "Side": side,
                        "Close": round(float(df["Close"].iloc[-1]), 2),
                        "AsOf": ASOF,
                    })
                except Exception:
                    misses.append(sym)

    out = pd.DataFrame(rows, columns=["Ticker", "RSI14", "Side", "Close", "AsOf"])
    out = out.dropna(subset=["RSI14"]).sort_values(["Side", "Ticker"]).reset_index(drop=True)
    return out, sorted(set(misses))

# ---- Outputs ----
def write_outputs(df: pd.DataFrame, misses: List[str]) -> None:
    combined = df.assign(
        List=np.where(df["Side"].str.lower() == "long", "oversold", "overbought")
    )[["Ticker", "List"]]
    combined.to_csv(ROOT / "combined_watchlist.csv", index=False)
    combined["Ticker