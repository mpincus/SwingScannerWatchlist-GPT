# scripts/trigger_scan.py
# Requires: pandas, numpy
import pandas as pd
from pathlib import Path

DATA = Path("data/combined.csv")
OUT  = Path("data/signals.csv")

def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-12)
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def compute_triggers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker","Date"]).copy()
    # RSI
    df["RSI14"] = df.groupby("Ticker")["Close"].transform(rsi14)

    # Prior 3-day levels (exclude today)
    df["H3"] = df.groupby("Ticker")["High"].transform(lambda s: s.shift(1).rolling(3).max())
    df["L3"] = df.groupby("Ticker")["Low" ].transform(lambda s: s.shift(1).rolling(3).min())

    # Previous OHLC for engulfing
    g = df.groupby("Ticker")
    prev_o = g["Open"].shift(1)
    prev_c = g["Close"].shift(1)

    bullish_engulf = (df["Close"]>df["Open"]) & (prev_c<prev_o) & (df["Close"]>=prev_o) & (df["Open"]<=prev_c)
    bearish_engulf = (df["Close"]<df["Open"]) & (prev_c>prev_o) & (df["Close"]<=prev_o) & (df["Open"]>=prev_c)

    # Triggers
    long_rsi_cross  = (df["Group"].eq("oversold"))   & (df["RSI14"].shift(1)<=30) & (df["RSI14"]>30)
    short_rsi_cross = (df["Group"].eq("overbought")) & (df["RSI14"].shift(1)>=70) & (df["RSI14"]<70)

    long_breakout   = df["Group"].eq("oversold")   & (df["Close"]>df["H3"])
    short_breakdown = df["Group"].eq("overbought") & (df["Close"]<df["L3"])

    long_candle  = df["Group"].eq("oversold")   & bullish_engulf
    short_candle = df["Group"].eq("overbought") & bearish_engulf

    # Build trigger labels
    labels = []
    for lr, lb, lc, sr, sd, sc in zip(long_rsi_cross, long_breakout, long_candle,
                                      short_rsi_cross, short_breakdown, short_candle):
        t = []
        if lr: t.append("LONG_RSI_CROSS")
        if lb: t.append("LONG_3DAY_BREAKOUT")
        if lc: t.append("LONG_ENGULF")
        if sr: t.append("SHORT_RSI_CROSS")
        if sd: t.append("SHORT_3DAY_BREAKDOWN")
        if sc: t.append("SHORT_ENGULF")
        labels.append("|".join(t))
    df["Trigger"] = labels

    # Side
    df["Side"] = ""
    df.loc[df["Trigger"].str.contains("LONG_", na=False) & ~df["Trigger"].str.contains("SHORT_", na=False), "Side"] = "long"
    df.loc[df["Trigger"].str.contains("SHORT_", na=False) & ~df["Trigger"].str.contains("LONG_", na=False), "Side"] = "short"
    df.loc[(df["Trigger"].str.contains("LONG_", na=False)) & (df["Trigger"].str.contains("SHORT_", na=False)), "Side"] = "both"

    cols = ["Date","Ticker","Group","Side","Trigger","Open","High","Low","Close","RSI14","H3","L3"]
    sig = df.loc[df["Trigger"]!="", cols].copy()
    sig["Date"] = pd.to_datetime(sig["Date"]).dt.date
    return sig.sort_values(["Date","Ticker"])

def main():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    sig = compute_triggers(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sig.to_csv(OUT, index=False)
    print(f"signals: {len(sig)} rows -> {OUT}")

if __name__ == "__main__":
    main()