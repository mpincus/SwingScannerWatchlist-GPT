#!/usr/bin/env python3 """ Build RSI-extremes-only watchlist and write it to combined_watchlist.csv for consistency. Also writes:

docs/combined_watchlist.txt (tickers only)

extremes.csv and extremes.txt (diagnostics, optional consumers)

missed_tickers.txt (any symbols we failed to fetch)

pullbacks.csv and breakouts.csv (empty stubs for workflow compatibility)


Rules

US only. ETFs allowed.

Universe: if combined_watchlist.csv exists at start, use its Ticker column as the universe. Otherwise auto-build SP500 + Nasdaq-100 + Dow 30 from Wikipedia.

RSI(14) extremes: longs if RSI ≤ 30, shorts if RSI ≥ 70.

Never fail CI: always emit artifacts even on errors.


Requirements: numpy, pandas, yfinance, requests, lxml """ from future import annotations import sys import os import time import pathlib from typing import List, Dict, Tuple

import numpy as np import pandas as pd import requests import yfinance as yf

---------------- Config ----------------

RSI_PERIOD = 14 ATR_PERIOD = 20 MA50 = 50 MA200 = 200 OVERSOLD = 30.0 OVERBOUGHT = 70.0

CHUNK = int(os.getenv("CHUNK", "120")) MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3")) RETRY_SLEEP = int(os.getenv("RETRY_SLEEP", "3")) HTTP_TIMEOUT = 20 UA = {"User-Agent": "Mozilla/5.0 (Scanner/CI)"}

ROOT = pathlib.Path(".").resolve() DOCS = ROOT / "docs" DOCS.mkdir(exist_ok=True) ASOF = pd.Timestamp.utcnow().tz_localize(None).date().isoformat()

---------------- Helpers ----------------

def _clean_symbol(t: str) -> str: return ( str(t).strip().upper() .replace(" ", "") .replace("/", "-") .replace(".", "-") )

def _read_html_table(url: str, match: str | None = None) -> pd.DataFrame: for i in range(MAX_RETRIES): try: r = requests.get(url, headers=UA, timeout=HTTP_TIMEOUT) r.raise_for_status() dfs = pd.read_html(r.text, match=match) if dfs: return dfs[0] except Exception: if i == MAX_RETRIES - 1: raise time.sleep(RETRY_SLEEP * (i + 1)) raise RuntimeError(f"Failed to read table: {url}")

def _load_universe() -> List[str]: """If combined_watchlist.csv exists, use it as the universe source. Else build SP500 + NDX + DOW from Wikipedia. """ src = ROOT / "combined_watchlist.csv" if src.exists(): try: df = pd.read_csv(src) candidates = [c for c in df.columns if c.lower() == "ticker" or "ticker" in c.lower() or "symbol" in c.lower()] col = candidates[0] if candidates else df.columns[0] syms = sorted({_clean_symbol(x) for x in df[col].dropna().astype(str) if str(x).strip()}) return syms except Exception: pass

# Auto-build US indices
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

syms = sorted({_clean_symbol(x) for x in pd.concat([sp[sp_col], ndx[ndx_col], dow[dow_col]]).dropna().astype(str)})

# Write baseline universe for transparency
pd.DataFrame({"Ticker": syms}).to_csv(ROOT / "combined_watchlist.csv", index=False)
pd.Series(syms, dtype=str).to_csv(DOCS / "combined_watchlist.txt", index=False, header=False)
return syms

---------------- Indicators ----------------

def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series: d = close.diff() up = d.clip(lower=0.0) down = -d.clip(upper=0.0) roll_up = up.ewm(alpha=1/period, adjust=False).mean() roll_down = down.ewm(alpha=1/period, adjust=False).mean() rs = roll_up / roll_down return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series: h, l, c = df["High"], df["Low"], df["Close"] pc = c.shift(1) tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1) return tr.rolling(period, min_periods=period).mean()

---------------- Data fetch ----------------

def _download_batch(symbols: List[str]) -> pd.DataFrame: tries = 0 last = None while tries < MAX_RETRIES: try: return yf.download( symbols, period="12mo", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True, ) except Exception as e: last = e tries += 1 time.sleep(RETRY_SLEEP * tries) raise RuntimeError(f"yfinance batch failed: {last}")

---------------- Scan ----------------

def scan_extremes(symbols: List[str]) -> Tuple[pd.DataFrame, List[str]]: rows: list[dict] = [] misses: list[str] = [] need = max(RSI_PERIOD, ATR_PERIOD) + 1

for i in range(0, len(symbols), CHUNK):
    batch = symbols[i:i+CHUNK]
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
                ma50 = df["Close"].rolling(MA50, min_periods=MA50).mean().iloc[-1]
                ma200 = df["Close"].rolling(MA200, min_periods=MA200).mean().iloc[-1]
                rows.append({
                    "Ticker": sym,
                    "RSI14": round(float(r), 2),
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
                if df.empty or len(df) < need:
                    continue
                r = _rsi(df["Close"]).iloc[-1]
                if np.isnan(r) or (OVERSOLD < r < OVERBOUGHT):
                    continue
                side = "long" if r <= OVERSOLD else "short"
                atr = _atr(df).iloc[-1]
                ma50 = df["Close"].rolling(MA50, min_periods=MA50).mean().iloc[-1]
                ma200 = df["Close"].rolling(MA200, min_periods=MA200).mean().iloc[-1]
                rows.append({
                    "Ticker": sym,
                    "RSI14": round(float(r), 2),
                    "Side": side,
                    "Close": round(float(df["Close"].iloc[-1]), 2),
                    "AsOf": ASOF,
                    "ATR20": round(float(atr), 2) if pd.notna(atr) else np.nan,
                    "MA50": round(float(ma50), 2) if pd.notna(ma50) else np.nan,
                    "MA200": round(float(ma200), 2) if pd.notna(ma200) else np.nan,
                })
            except Exception:
                misses.append(sym)

out = pd.DataFrame(rows, columns=[
    "Ticker","RSI14","Side","Close","AsOf","ATR20","MA50","MA200"
])
out = out.dropna(subset=["RSI14"]).sort_values(["Side","Ticker"]).reset_index(drop=True)
return out, sorted(set(misses))

---------------- Outputs ----------------

def write_outputs(df: pd.DataFrame, misses: List[str]) -> None: # Primary output: combined_watchlist.csv with oversold/overbought labels combined = df.assign( List=np.where(df["Side"].str.lower() == "long", "oversold", "overbought") )[["Ticker","List"]]

combined.to_csv(ROOT / "combined_watchlist.csv", index=False)
combined["Ticker"].to_csv(DOCS / "combined_watchlist.txt", index=False, header=False)

# Diagnostics/legacy
df.to_csv(ROOT / "extremes.csv", index=False)
df["Ticker"].to_csv(ROOT / "extremes.txt", index=False, header=False)
pd.Series(misses, dtype=str).to_csv(ROOT / "missed_tickers.txt", index=False, header=False)

# Legacy stubs to avoid workflow breakage
pd.DataFrame({"Ticker": []}).to_csv(ROOT / "pullbacks.csv", index=False)
pd.DataFrame({"Ticker": []}).to_csv(ROOT / "breakouts.csv", index=False)

---------------- Main ----------------

def main() -> int: try: universe = _load_universe() if not universe: # still emit empty artifacts write_outputs(pd.DataFrame(columns=[ "Ticker","RSI14","Side","Close","AsOf","ATR20","MA50","MA200" ]), []) print("Watchlist empty. Outputs written.") return 0 df, misses = scan_extremes(universe) write_outputs(df, misses) print(f"Universe: {len(universe)} | Extremes: {len(df)} | Misses: {len(misses)}") return 0 except Exception as e: # Never fail CI; emit empty artifacts try: write_outputs(pd.DataFrame(columns=[ "Ticker","RSI14","Side","Close","AsOf","ATR20","MA50","MA200" ]), [str(e)]) except Exception: pass print(f"WARN: {e}") return 0

if name == "main": sys.exit(main())

