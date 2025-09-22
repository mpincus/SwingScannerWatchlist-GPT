# scripts/quality.py
# Input:  data/combined.csv  (Date,Ticker,Group,Open,High,Low,Close,Volume)
# Output: data/quality_today.csv  (with WinRate10, Samples10, Grade based on likelihood)

from pathlib import Path
import pandas as pd
import numpy as np

COMBINED = Path("data/combined.csv")
OUT = Path("data/quality_today.csv")

# ---- Config ----
LOOKBACK_DAYS = 180      # rolling history window
HORIZON = 10             # profit horizon (days)
MIN_SAMPLES = 80         # minimum samples for pattern; else fallback to global
HALF_LIFE = 60           # days for exponential decay weighting
WIN_THRESH = {           # likelihood-based grading thresholds
    "A+": 0.65,
    "A":  0.55,
    "B+": 0.50
}

def rsi14(close: pd.Series) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-12)
    rs = ag / al
    return 100 - (100 / (1 + rs))

def grade_from_winrate(p: float) -> str:
    if p is None: return ""
    if p >= WIN_THRESH["A+"]: return "A+"
    if p >= WIN_THRESH["A"]:  return "A"
    if p >= WIN_THRESH["B+"]: return "B+"
    return ""

def main():
    df = pd.read_csv(COMBINED, parse_dates=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
    if df.empty:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(OUT, index=False)
        print("combined.csv empty")
        return

    # Context
    g = df.groupby("Ticker", group_keys=False)
    df["RSI14"] = g["Close"].transform(rsi14)
    df["PrevOpen"]  = g["Open"].shift(1)
    df["PrevClose"] = g["Close"].shift(1)
    df["H3"] = g["High"].shift(1).rolling(3).max()
    df["L3"] = g["Low" ].shift(1).rolling(3).min()
    for d in (HORIZON,):
        df[f"Close_fut{d}"] = g["Close"].shift(-d)

    last_date = df["Date"].max()
    cutoff = last_date - pd.Timedelta(days=LOOKBACK_DAYS)

    # Engulfing
    bull = (df["Close"] > df["Open"]) & (df["PrevClose"] < df["PrevOpen"]) & \
           (df["Close"] >= df["PrevOpen"]) & (df["Open"] <= df["PrevClose"])
    bear = (df["Close"] < df["Open"]) & (df["PrevClose"] > df["PrevOpen"]) & \
           (df["Close"] <= df["PrevOpen"]) & (df["Open"] >= df["PrevClose"])

    # Quality definitions
    long_q  = (df["Group"]=="oversold")   & (df["RSI14"]<=30) & bull & df["L3"].notna() & ((df["Close"]-df["L3"])>0)
    short_q = (df["Group"]=="overbought") & (df["RSI14"]>=70) & bear & df["H3"].notna() & ((df["H3"]-df["Close"])>0)

    df["Pattern"] = np.where(long_q, "OversoldLongRev",
                       np.where(short_q, "OverboughtShortRev", ""))

    # --- Build historical sample for win-rate ---
    hist = df[(df["Date"]>=cutoff) & (df["Date"]<last_date) & (df["Pattern"]!="")].copy()
    if not hist.empty:
        # Returns at horizon, signed by side
        hist["Side"] = np.where(hist["Pattern"]=="OversoldLongRev", "long", "short")
        hist[f"ret{HORIZON}"] = (hist[f"Close_fut{HORIZON}"] - hist["Close"]) / hist["Close"]
        hist.loc[hist["Side"]=="short", f"ret{HORIZON}"] *= -1
        hist["win"] = hist[f"ret{HORIZON}"] > 0

        # Exponential decay weights by recency
        age_days = (last_date - hist["Date"]).dt.days.clip(lower=0).astype(float)
        lam = np.log(2) / HALF_LIFE
        hist["w"] = np.exp(-lam * age_days)

        # Pattern-level weighted win rate
        wstats = (hist.groupby("Pattern")
                       .apply(lambda x: pd.Series({
                           "Samples10": int(x["win"].notna().sum()),
                           "WinRate10": float((x["win"]*x["w"]).sum() / x["w"].sum()) if x["w"].sum()>0 else np.nan
                       }))
                 ).reset_index()
    else:
        wstats = pd.DataFrame(columns=["Pattern","Samples10","WinRate10"])

    # Global fallback
    global_wr = None
    if not wstats.empty and wstats["WinRate10"].notna().any():
        # Weighted global across patterns
        mask = hist["win"].notna()
        if mask.any():
            global_wr = float((hist.loc[mask,"win"]*hist.loc[mask,"w"]).sum() / hist.loc[mask,"w"].sum())

    # --- Today’s candidates ---
    today = df[(df["Date"]==last_date) & (df["Pattern"]!="")].copy()
    if today.empty:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        today.to_csv(OUT, index=False)
        print(f"No quality setups for {last_date.date()}")
        return

    # Stops/targets (unchanged, used for execution not grading)
    longs = today["Pattern"]=="OversoldLongRev"
    shorts = today["Pattern"]=="OverboughtShortRev"

    today.loc[longs,  "Side"]   = "long"
    today.loc[shorts, "Side"]   = "short"
    today.loc[longs,  "Stop"]   = today.loc[longs,  "L3"]
    today.loc[longs,  "Target"] = today.loc[longs,  "Close"] + 1.25*(today.loc[longs,  "Close"] - today.loc[longs,  "L3"])
    today.loc[shorts, "Stop"]   = today.loc[shorts, "H3"]
    today.loc[shorts, "Target"] = today.loc[shorts, "Close"] - 1.25*(today.loc[shorts, "H3"]   - today.loc[shorts, "Close"])
    today["R_R"] = 1.25  # informational only

    # Attach likelihood grade
    today = today.merge(wstats, on="Pattern", how="left")
    # Fallback to global if pattern has too few samples or NaN
    use_fallback = (today["Samples10"].fillna(0) < MIN_SAMPLES) | today["WinRate10"].isna()
    if global_wr is not None:
        today.loc[use_fallback, "WinRate10"] = global_wr
        today.loc[use_fallback, "Samples10"] = hist["win"].notna().sum()

    today["Grade"] = today["WinRate10"].apply(grade_from_winrate)

    # Output
    cols = ["Date","Ticker","Group","Side","Pattern","Open","High","Low","Close",
            "RSI14","H3","L3","Stop","Target","R_R","WinRate10","Samples10","Grade"]
    out = today[cols].sort_values(["Grade","Pattern","Ticker"], ascending=[True,True,True]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"{len(out)} setups for {last_date.date()}  |  patterns used: {wstats.to_dict('records')}  |  global_wr={global_wr}")

if __name__ == "__main__":
    main()