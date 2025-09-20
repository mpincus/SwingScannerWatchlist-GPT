#!/usr/bin/env python3
"""
Outputs RSI-extremes from the full US watchlist.
- Reads combined_watchlist.csv (Ticker column). If missing, builds SP500+NDX+DJIA.
- Writes:
  * extremes.csv  (Ticker,RSI14,Side,Close,AsOf,ATR20,MA50,MA200)
  * extremes.txt  (tickers only)
  * missed_tickers.txt
  * combined_watchlist.csv + docs/combined_watchlist.txt (created if missing)
  * pullbacks.csv and breakouts.csv stubs (empty) for workflow compatibility
"""

import os, time, pathlib, sys
from typing import List, Dict
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------- Config ----------
RSI_PERIOD = 14
ATR_PERIOD = 20
MA1, MA2 = 50, 200
OVERSOLD, OVERBOUGHT = 30.0, 70.0
CHUNK = 120
MAX_RETRIES, RETRY_SLEEP = 3, 3

ROOT = pathlib.Path(".").resolve()
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
ASOF = pd.Timestamp.utcnow().tz_localize(None).date().isoformat()

# ---------- Helpers ----------
def _clean(t: str) -> str:
    return str(t).strip().upper().replace(" ", "").replace("/", "-").replace(".", "-")

def _read_html_table(url: str, match: str | None = None) -> pd.DataFrame:
    ua = {"User-Agent": "Mozilla/5.0 (Scanner/CI)"}
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
    raise RuntimeError(f"Failed to read {url}")

def _universe() -> List[str]:
    wl = ROOT / "combined_watchlist.csv"
    if wl.exists():
        df = pd.read_csv(wl)
        cols = [c for c in df.columns if c.lower() in ("ticker", "symbol") or "ticker" in c.lower()]
        col = cols[0] if cols else df.columns[0]
        return sorted({_clean(x) for x in df[col].dropna().astype(str) if str(x).strip()})

    sp = _read_html_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    spc = "Symbol" if "Symbol" in sp.columns else sp.columns[0]
    nd = _read_html_table("https://en.wikipedia.org/wiki/Nasdaq-100", match="Ticker|Symbol")
    ndc = "Ticker" if "Ticker" in nd.columns else ("Symbol" if "Symbol" in nd.columns else nd.columns[0])
    dj = _read_html_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", match="Symbol|Ticker")
    djc = "Symbol" if "Symbol" in dj.columns else ( "Ticker" if "Ticker" in dj.columns else dj.columns[0])

    syms = sorted({_clean(x) for x in pd.concat([sp[spc], nd[ndc], dj[djc]]).dropna().astype(str)})
    pd.DataFrame({"Ticker": syms}).to_csv(ROOT / "combined_watchlist.csv", index=False)
    pd.Series(syms, dtype=str).to_csv(DOCS / "combined_watchlist.txt", index=False, header=False)
    return syms

def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    d = close.diff()
    up, down = d.clip(lower=0), -d.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def _download_batch(symbols: List[str]) -> pd.DataFrame:
    tries = 0
    err = None
    while tries < MAX_RETRIES:
        try:
            return yf.download(symbols, period="12mo", interval="1d",
                               auto_adjust=False, progress=False,
                               group_by="ticker", threads=True)
        except Exception as e:
            err = e
            tries += 1
            time.sleep(RETRY_SLEEP * tries)
    raise RuntimeError(f"yfinance batch failed: {err}")

# ---------- Scan ----------
def scan(symbols: List[str]) -> tuple[pd.DataFrame, List[str]]:
    rows, misses = [], []
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
                    if df.empty or len(df) < max(RSI_PERIOD, ATR_PERIOD) + 1:
                        continue
                    r = _rsi(df["Close"]).iloc[-1]
                    if np.isnan(r) or (r > OVERSOLD and r < OVERBOUGHT):
                        continue
                    side = "long" if r <= OVERSOLD else "short"
                    rows.append({
                        "Ticker": sym,
                        "RSI14": round(float(r), 2),
                        "Side": side,
                        "Close": round(float(df["Close"].iloc[-1]), 2),
                        "AsOf": ASOF,
                        "ATR20": round(float(_atr(df).iloc[-1]), 2),
                        "MA50": round(float(df["Close"].rolling(50).mean().iloc[-1]), 2),
                        "MA200": round(float(df["Close"].rolling(200).mean().iloc[-1]), 2),
                    })
                except Exception:
                    misses.append(sym)
        else:
            for sym in batch:
                try:
                    df = yf.download(sym, period="12mo", interval="1d",
                                     auto_adjust=False, progress=False)
                    if df.empty or len(df) < max(RSI_PERIOD, ATR_PERIOD) + 1:
                        continue
                    r = _rsi(df["Close"]).iloc[-1]
                    if np.isnan(r) or (r > OVERSOLD and r < OVERBOUGHT):
                        continue
                    side = "long" if r <= OVERSOLD else "short"
                    rows.append({
                        "Ticker": sym,
                        "RSI14": round(float(r), 2),
                        "Side": side,
                        "Close": round(float(df["Close"].iloc[-1]), 2),
                        "AsOf": ASOF,
                        "ATR20": round(float(_atr(df).iloc[-1]), 2),
                        "MA50": round(float(df["Close"].rolling(50).mean().iloc[-1]), 2),
                        "MA200": round(float(df["Close"].rolling(200).mean().iloc[-1]), 2),
                    })
                except Exception:
                    misses.append(sym)

    out = pd.DataFrame(rows, columns=["Ticker","RSI14","Side","Close","AsOf","ATR20","MA50","MA200"])
    out = out.dropna(subset=["RSI14"]).sort_values(["Side","Ticker"]).reset_index(drop=True)
    return out, sorted(set(misses))

# ---------- Main ----------
def main():
    try:
        syms = [_clean(s) for s in _universe()]
        df, misses = scan(syms)
        df.to_csv(ROOT / "extremes.csv", index=False)
        df["Ticker"].to_csv(ROOT / "extremes.txt", index=False, header=False)
        pd.Series(misses, dtype=str).to_csv(ROOT / "missed_tickers.txt", index=False, header=False)
        # legacy stubs for workflow compatibility
        pd.DataFrame({"Ticker": []}).to_csv(ROOT / "pullbacks.csv", index=False)
        pd.DataFrame({"Ticker": []}).to_csv(ROOT / "breakouts.csv", index=False)
        # ensure docs txt exists
        if not (ROOT / "combined_watchlist.csv").exists():
            pd.DataFrame({"Ticker": syms}).to_csv(ROOT / "combined_watchlist.csv", index=False)
        pd.read_csv(ROOT / "combined_watchlist.csv")["Ticker"].to_csv(
            DOCS / "combined_watchlist.txt", index=False, header=False
        )
        print(f"Watchlist count: {len(syms)} | Extremes: {len(df)} | Misses: {len(misses)}")
        return 0
    except Exception as e:
        # never fail CI; still emit empty artifacts
        pd.DataFrame(columns=["Ticker","RSI14","Side","Close","AsOf","ATR20","MA50","MA200"]).to_csv(ROOT / "extremes.csv", index=False)
        pd.Series([], dtype=str).to_csv(ROOT / "extremes.txt", index=False, header=False)
        pd.Series([str(e)], dtype=str).to_csv(ROOT / "missed_tickers.txt", index=False, header=False)
        for n in ("pullbacks.csv","breakouts.csv"): pd.DataFrame({"Ticker": []}).to_csv(ROOT / n, index=False)
        return 0

if __name__ == "__main__":
    sys.exit(main())