# Build data/quality_today.csv directly from data/combined.csv
# Gates (same as before):
#   Longs: Group==oversold AND RSI<=30 AND bullish engulfing AND R/R ≥ 1.25 (stop = prior 3-day low)
#   Shorts: Group==overbought AND RSI>=70 AND bearish engulfing AND R/R ≥ 1.25 (stop = prior 3-day high)

from pathlib import Path
import pandas as pd

COMBINED = Path("data/combined.csv")
OUT = Path("data/quality_today.csv")

def rsi14(s: pd.Series) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-12)
    rs = ag / al
    return 100 - (100 / (1 + rs))

def grade(rr):
    if rr >= 1.75: return "A+"
    if rr >= 1.50: return "A"
    if rr >= 1.25: return "B+"
    return "REJECT"

def main():
    df = pd.read_csv(COMBINED, parse_dates=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)

    # Compute context
    df["RSI14"] = df.groupby("Ticker")["Close"].transform(rsi14)
    g = df.groupby("Ticker", group_keys=False)
    df["PrevOpen"]  = g["Open"].shift(1)
    df["PrevClose"] = g["Close"].shift(1)
    df["H3"] = g["High"].shift(1).rolling(3).max()
    df["L3"] = g["Low"].shift(1).rolling(3).min()

    # Use latest date available
    last_date = df["Date"].max()
    today = df[df["Date"] == last_date].dropna(subset=["PrevOpen","PrevClose","H3","L3","RSI14"]).copy()

    # Engulfing
    bull = (today["Close"] > today["Open"]) & (today["PrevClose"] < today["PrevOpen"]) & \
           (today["Close"] >= today["PrevOpen"]) & (today["Open"] <= today["PrevClose"])
    bear = (today["Close"] < today["Open"]) & (today["PrevClose"] > today["PrevOpen"]) & \
           (today["Close"] <= today["PrevOpen"]) & (today["Open"] >= today["PrevClose"])

    # Longs from oversold
    longs = today[(today["Group"] == "oversold") & (today["RSI14"] <= 30) & bull].copy()
    longs["Stop"]   = longs["L3"]
    longs["Risk"]   = (longs["Close"] - longs["Stop"]).clip(lower=1e-6)
    longs["Target"] = longs["Close"] + 1.25 * longs["Risk"]
    longs["R_R"]    = 1.25
    longs["Side"]   = "long"
    longs["Trigger"]= "QUALITY_REVERSAL"

    # Shorts from overbought
    shorts = today[(today["Group"] == "overbought") & (today["RSI14"] >= 70) & bear].copy()
    shorts["Stop"]   = shorts["H3"]
    shorts["Risk"]   = (shorts["Stop"] - shorts["Close"]).clip(lower=1e-6)
    shorts["Target"] = shorts["Close"] - 1.25 * shorts["Risk"]
    shorts["R_R"]    = 1.25
    shorts["Side"]   = "short"
    shorts["Trigger"]= "QUALITY_REVERSAL"

    out = pd.concat([longs, shorts], ignore_index=True)
    if out.empty:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT, index=False)
        print(f"0 quality setups for {last_date.date()} -> {OUT}")
        return

    out["Grade"] = out["R_R"].apply(grade)
    out = out[out["Grade"].isin(["A+","A","B+"])]

    cols = ["Date","Ticker","Group","Side","Trigger","Open","High","Low","Close",
            "RSI14","H3","L3","Stop","Target","R_R","Grade"]
    out = out[cols].sort_values(["Side","Grade","Ticker"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"{len(out)} quality setups for {last_date.date()} -> {OUT}")

if __name__ == "__main__":
    main()