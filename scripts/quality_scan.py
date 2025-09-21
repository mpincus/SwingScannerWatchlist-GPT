# scripts/quality_scan.py

from pathlib import Path
import pandas as pd
import numpy as np

COMBINED = Path("data/combined.csv")
SIGNALS  = Path("data/signals.csv")
OUT      = Path("data/signals_quality.csv")

def rsi14(close: pd.Series) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-12)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def main():
    # Load OHLC and compute RSI
    df = pd.read_csv(COMBINED, parse_dates=["Date"]).sort_values(["Ticker","Date"])
    df["RSI14"] = df.groupby("Ticker")["Close"].transform(rsi14)

    # Previous candle + prior 3-day high/low
    g = df.groupby("Ticker", group_keys=False)
    df["PrevOpen"]  = g["Open"].shift(1)
    df["PrevClose"] = g["Close"].shift(1)
    df["H3"] = g["High"].shift(1).rolling(3).max()
    df["L3"] = g["Low"].shift(1).rolling(3).min()

    # Load signals
    sig = pd.read_csv(SIGNALS, parse_dates=["Date"])

    # Merge signals with context
    ctx_cols = ["Date","Ticker","Group","Open","High","Low","Close","RSI14","PrevOpen","PrevClose","H3","L3"]
    m = pd.merge(sig, df[ctx_cols], on=["Date","Ticker","Group"], how="left")

    # Drop missing data
    m = m.dropna(subset=["Open","Close","RSI14","PrevOpen","PrevClose","H3","L3"])

    # Engulfing checks
    bull = (m["Close"] > m["Open"]) & (m["PrevClose"] < m["PrevOpen"]) & (m["Close"] >= m["PrevOpen"]) & (m["Open"] <= m["PrevClose"])
    bear = (m["Close"] < m["Open"]) & (m["PrevClose"] > m["PrevOpen"]) & (m["Close"] <= m["PrevOpen"]) & (m["Open"] >= m["PrevClose"])

    # RSI gates
    rsi_long_ok  = m["RSI14"] <= 30
    rsi_short_ok = m["RSI14"] >= 70

    # Risk/Reward
    risk_long  = m["Close"] - m["L3"]
    risk_short = m["H3"] - m["Close"]

    rr_long  = (1.25 * risk_long)  / risk_long
    rr_short = (1.25 * risk_short) / risk_short
    rr_long  = rr_long.where(risk_long > 0)
    rr_short = rr_short.where(risk_short > 0)

    # Masks
    long_mask  = (m["Side"]=="long")  & rsi_long_ok  & bull & (rr_long  >= 1.25)
    short_mask = (m["Side"]=="short") & rsi_short_ok & bear & (rr_short >= 1.25)

    passed = m[long_mask | short_mask].copy()

    if passed.empty:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        passed.to_csv(OUT, index=False)
        print("0 quality signals -> data/signals_quality.csv")
        return

    # Stops / targets
    passed.loc[long_mask,  "Stop"]   = passed.loc[long_mask,  "L3"]
    passed.loc[long_mask,  "Target"] = passed.loc[long_mask,  "Close"] + 1.25 * (passed.loc[long_mask,  "Close"] - passed.loc[long_mask,  "L3"])
    passed.loc[short_mask, "Stop"]   = passed.loc[short_mask, "H3"]
    passed.loc[short_mask, "Target"] = passed.loc[short_mask, "Close"] - 1.25 * (passed.loc[short_mask, "H3"] - passed.loc[short_mask, "Close"])

    passed["R_R"] = 1.25

    out_cols = ["Date","Ticker","Group","Side","Trigger","Open","High","Low","Close","RSI14","H3","L3","Stop","Target","R_R"]
    passed = passed[out_cols].sort_values(["Date","Ticker"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    passed.to_csv(OUT, index=False)
    print(f"{len(passed)} quality signals -> {OUT}")

if __name__ == "__main__":
    main()