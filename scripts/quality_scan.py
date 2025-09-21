# scripts/quality_scan.py
# Build data/signals_quality.csv by applying trade-quality gates to signals.csv
# Gates: RSI extreme (<=30 long, >=70 short) + engulfing reversal + R/R >= 1.25 using prior 3-day S/R
# Inputs:  data/combined.csv  (Date,Ticker,Group,Open,High,Low,Close,Volume)
#          data/signals.csv   (Date,Ticker,Group,Side,Trigger,Open,High,Low,Close,RSI14 optional)
# Output:  data/signals_quality.csv

from pathlib import Path
import pandas as pd

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
    df = pd.read_csv(COMBINED, parse_dates=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
    sig = pd.read_csv(SIGNALS,  parse_dates=["Date"])

    # Compute RSI14 in-case combined.csv doesn't have it
    df["RSI14"] = df.groupby("Ticker")["Close"].transform(rsi14)

    # Precompute previous candle and prior 3-day S/R (exclude today with shift(1))
    g = df.groupby("Ticker", group_keys=False)
    df["PrevOpen"]  = g["Open"].shift(1)
    df["PrevClose"] = g["Close"].shift(1)
    df["H3"] = g["High"].shift(1).rolling(3).max()
    df["L3"] = g["Low" ].shift(1).rolling(3).min()

    # Merge signals with needed context
    cols_ctx = ["Date","Ticker","Group","Open","High","Low","Close","RSI14","PrevOpen","PrevClose","H3","L3"]
    m = pd.merge(sig, df[cols_ctx], on=["Date","Ticker","Group"], how="left")

    # Drop rows without required context
    m = m.dropna(subset=["Open_x","Close_x","RSI14","PrevOpen","PrevClose"])
    # Use signal's OHLC as entry candle
    m = m.rename(columns={"Open_x":"Open","High_x":"High","Low_x":"Low","Close_x":"Close"})

    # Engulfing checks
    bull_engulf = (m["Close"] > m["Open"]) & (m["PrevClose"] < m["PrevOpen"]) & (m["Close"] >= m["PrevOpen"]) & (m["Open"] <= m["PrevClose"])
    bear_engulf = (m["Close"] < m["Open"]) & (m["PrevClose"] > m["PrevOpen"]) & (m["Close"] <= m["PrevOpen"]) & (m["Open"] >= m["PrevClose"])

    # RSI extreme gates
    rsi_long_ok  = m["RSI14"] <= 30
    rsi_short_ok = m["RSI14"] >= 70

    # Risk/Reward using prior 3-day levels
    # Long: stop=L3, risk=Close-L3, target=Close + 1.25*risk
    # Short: stop=H3, risk=H3-Close, target=Close - 1.25*risk
    m["Risk_long"]  = m["Close"] - m["L3"]
    m["Risk_short"] = m["H3"] - m["Close"]

    rr_long  = (1.25 * m["Risk_long"])  / m["Risk_long"]
    rr_short = (1.25 * m["Risk_short"]) / m["Risk_short"]
    # Guard divide-by-zero or negative risk
    rr_long  = rr_long.where(m["Risk_long"]  > 0)
    rr_short = rr_short.where(m["Risk_short"] > 0)

    # Build masks
    long_mask  = (m["Side"]=="long")  & rsi_long_ok  & bull_engulf  & (rr_long  >= 1.25)
    short_mask = (m["Side"]=="short") & rsi_short_ok & bear_engulf  & (rr_short >= 1.25)

    passed = m[long_mask | short_mask].copy()
    if passed.empty:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        passed.to_csv(OUT, index=False)
        print("0 quality signals -> data/signals_quality.csv")
        return

    # Compute Stop/Target and R_R for output
    passed.loc[long_mask,  "Stop"]   = passed.loc[long_mask,  "L3"]
    passed.loc[long_mask,  "Target"] = passed.loc[long_mask,  "Close"] + 1.25 * (passed.loc[long_mask,  "Close"] - passed.loc[long_mask,  "L3"])
    passed.loc[short_mask, "Stop"]   = passed.loc[short_mask, "H3"]
    passed.loc[short_mask, "Target"] = passed.loc[short_mask, "Close"] - 1.25 * (passed.loc[short_mask, "H3"] - passed.loc[short_mask, "Close"])

    passed["R_R"] = 1.25  # by construction with this simple S/R model

    # Align columns with prior schema + extras
    out_cols = ["Date","Ticker","Group","Side","Trigger",
                "Open","High","Low","Close","RSI14","H3","L3",
                "Stop","Target","R_R"]
    # Pull Trigger/Side from signals (already in 'm')
    passed = passed.rename(columns={"Trigger":"Trigger", "Side":"Side"})
    passed = passed[out_cols].sort_values(["Date","Ticker"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    passed.to_csv(OUT, index=False)
    print(f"{len(passed)} quality signals -> {OUT}")

if __name__ == "__main__":
    main()