# Generates data/signals.csv using RSI-up and RSI-down triggers.
# Inputs:  data/combined.csv  (must contain: Date,Ticker,Group,Open,High,Low,Close,Volume)
# Output:  data/signals.csv   (columns kept compatible with prior workflow)

import pandas as pd
from pathlib import Path

COMBINED = Path("data/combined.csv")
OUT = Path("data/signals.csv")

def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-12)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def main():
    df = pd.read_csv(COMBINED, parse_dates=["Date"])
    df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)

    # Compute RSI(14) per ticker
    df["RSI14"] = df.groupby("Ticker")["Close"].transform(rsi14)
    df["RSI14_prev"] = df.groupby("Ticker")["RSI14"].shift(1)

    signals = []

    # RSI Up -> long
    sig_up = df[(df["RSI14"] > df["RSI14_prev"]) & df["RSI14"].notna() & df["RSI14_prev"].notna()].copy()
    sig_up["Side"] = "long"
    sig_up["Trigger"] = "RSI_UP"
    signals.append(sig_up)

    # RSI Down -> short
    sig_down = df[(df["RSI14"] < df["RSI14_prev"]) & df["RSI14"].notna() & df["RSI14_prev"].notna()].copy()
    sig_down["Side"] = "short"
    sig_down["Trigger"] = "RSI_DOWN"
    signals.append(sig_down)

    sig = pd.concat(signals, ignore_index=True)

    # Optional ref levels for schema compatibility
    sig["H3"] = sig.groupby("Ticker")["High"].shift(1).rolling(3).max().reset_index(level=0, drop=True)
    sig["L3"] = sig.groupby("Ticker")["Low"].shift(1).rolling(3).min().reset_index(level=0, drop=True)

    cols = ["Date","Ticker","Group","Side","Trigger","Open","High","Low","Close","RSI14","H3","L3"]
    sig = sig[cols].sort_values(["Date","Ticker"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sig.to_csv(OUT, index=False)
    print(f"signals: {len(sig)} rows -> {OUT}")

if __name__ == "__main__":
    main()