#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build RSI-extremes-only watchlist and write to combined_watchlist.csv.
Also writes:
- docs/combined_watchlist.txt (tickers only)
- extremes.csv and extremes.txt (diagnostics)
- missed_tickers.txt (symbols we failed to fetch)
- pullbacks.csv and breakouts.csv (empty stubs for workflow compatibility)

Rules
- US only. ETFs allowed.
- Universe source: S&P 500 + Nasdaq-100 + Dow 30 (from Wikipedia) every run.
- RSI(14) extremes: long if RSI <= 30, short if RSI >= 70.
- Never fail CI: always emit artifacts even on errors.

Requirements: numpy, pandas, yfinance, requests, lxml
"""

from __future__ import annotations

import sys
import os
import time
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- Config ----------------
RSI_PERIOD = 14
ATR_PERIOD = 20
MA50 = 50
MA200 = 200
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

# ---------------- Helpers ----------------

def _clean_symbol(t: str) -> str:
    return (
        str(t).strip().upper()
        .replace(" ", "")
        .replace("/", "-")
        .replace(".", "-")
    )

def _read_html_table(url: str, match: str | None = None) -> pd.DataFrame:
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

def _load_universe_from_indices() -> List[str]:
    sp = _read_html_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp_col = "Symbol" if "Symbol" in sp.columns else sp.columns[0]

    ndx = _read_html_table("https://en.wikipedia.org/wiki/Nasdaq-100", match="Ticker|Symbol")
    if "Ticker" in ndx.columns:
        ndx_col = "Ticker"
    elif "Symbol" in ndx.columns:
        ndx_col = "Symbol"
    else:
        ndx_col = ndx.columns[0]

    dow = _read_html_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", match="Symbol|Ticker")
    if "Symbol" in dow.columns:
        dow_col = "Symbol"
    elif "Ticker" in dow.columns:
        dow_col = "Ticker"
    else:
        dow_col = dow.columns[0]

    syms = sorted({
        _clean_symbol(x)
        for x in pd.concat([sp[sp_col], ndx[ndx_col], dow[dow_col]]).dropna().astype(str)
    })
    return syms

# ---------------- Indicators ----------------

def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))

def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

# ---------------- Data fetch ----------------

def _download_batch(symbols: List[str]) -> pd.DataFrame:
    tries = 0
    last_err = None
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
            last_err = e
            tries += 1
            time.sleep(RETRY_SLEEP * tries)
    raise RuntimeError(f"yfinance batch failed: {last_err}")

# ---------------- Scan ----------------

def scan_extremes(symbols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    rows: list[dict] = []
    misses: list[str] = []
    need = max(RSI_PERIOD, ATR_PERIOD) + 1

    for i in range(0, len(symbols), CHUNK):
        batch = symbols[i:i + CHUNK]
        data = None
        try:
            data = _download_batch(batch)
        except Exception:
            data = None

        if data is not None and isinstance(data.columns, pd.MultiIndex):
            names = sorted({t for t, _ in data.columns})
            for sym in names:
                try:
                    df = data[sym].dropna()
                    if df.empty or len(df) < need:
                        continue
                    r = _rsi(df["Close"]).iloc[-1]
                    if np.isnan(r) or (OVERSOLD < r < OVERBOUGHT):
                        continue
                    side = "long" if r <= OVERSOLD else "short"
                    atr = _atr(df).iloc[-1]
                    ma50