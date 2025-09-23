# scripts/backtest_quality.py
from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path("data/combined.csv")
OUT_TRADES = Path("data/quality_trades.csv")
OUT_STATS  = Path("data/quality_stats.csv")

# ----- helpers
def rsi14(s: pd.Series) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1/14, adjust=False).mean()
    ad = dn.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-12)
    rs = au / ad
    return 100 - 100/(1+rs)

def rolling_prev_extrema(x: pd.Series, win: int, fn: str) -> pd.Series:
    # prior-window extremum (exclude today)
    s = x.shift(1).rolling(win, min_periods=1)
    return (s.max() if fn == "max" else s.min())

def engulfing_flags(df: pd.DataFrame):
    prev_open  = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)
    bull = (df["Close"] > df["Open"]) & (prev_close < prev_open) & \
           (df["Close"] >= prev_open) & (df["Open"] <= prev_close)
    bear = (df["Close"] < df["Open"]) & (prev_close > prev_open) & \
           (df["Close"] <= prev_open) & (df["Open"] >= prev_close)
    return bull, bear

def percent_retrace(close: pd.Series, lookback_high: pd.Series, lookback_low: pd.Series) -> pd.Series:
    # (recentHigh - Close) / (recentHigh - recentLow); clamp to [0,1]
    span = (lookback_high - lookback_low).replace(0, np.nan)
    pct = (lookback_high - close) / span
    return pct.clip(lower=0, upper=1).fillna(0)

# ----- load
df = pd.read_csv(SRC, parse_dates=["Date"]).sort_values(["Ticker","Date"])
if df.empty:
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(OUT_TRADES, index=False)
    pd.DataFrame().to_csv(OUT_STATS, index=False)
    raise SystemExit("empty combined.csv")

# ----- features
g = df.groupby("Ticker", group_keys=False)
df["RSI14"] = g["Close"].transform(rsi14)
df["MA20"]  = g["Close"].transform(lambda s: s.rolling(20).mean())
df["MA50"]  = g["Close"].transform(lambda s: s.rolling(50).mean())
df["MA200"] = g["Close"].transform(lambda s: s.rolling(200).mean())

# engulfing
bull_e, bear_e = engulfing_flags(df)
df["BullEngulf"] = bull_e
df["BearEngulf"] = bear_e

# 3-day stop anchors (prior window)
df["H3"] = g["High"].transform(lambda s: rolling_prev_extrema(s, 3, "max"))
df["L3"] = g["Low"] .transform(lambda s: rolling_prev_extrema(s, 3, "min"))

# future closes for returns
for d in (5,10,15):
    df[f"Close_fut{d}"] = g["Close"].shift(-d)

# future path (for target/stop hit checks)
# rolling max/min *forward* over next N days via reverse rolling
def fwd_roll_max(x, n): return x.shift(-1).rolling(n, min_periods=1).max()
def fwd_roll_min(x, n): return x.shift(-1).rolling(n, min_periods=1).min()
for n in (10,):
    df[f"FwdHigh_{n}"] = g["High"].apply(lambda s: fwd_roll_max(s, n))
    df[f"FwdLow_{n}"]  = g["Low"] .apply(lambda s: fwd_roll_min(s, n))

# swing context for continuation retrace
df["LH10"] = g["High"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).max())
df["LL10"] = g["Low"] .transform(lambda s: s.shift(1).rolling(10, min_periods=1).min())
df["RetracePct"] = percent_retrace(df["Close"], df["LH10"], df["LL10"])

# ----- quality gate (v1.0)
# Reversals
rev_long  = (df["Group"]=="oversold")   & (df["RSI14"]<=30) & df["BullEngulf"] & df["L3"].notna()
rev_short = (df["Group"]=="overbought") & (df["RSI14"]>=70) & df["BearEngulf"] & df["H3"].notna()

# Continuations (refined)
cont_long  = (df["Close"]>df["MA50"]) & (df["RSI14"].between(40,65, inclusive="both")) & \
             ((df["RetracePct"]<=0.38) | (df["Close"]>df["LH10"])) & (df["BullEngulf"] | (df["Close"]>df["LH10"])) & df["L3"].notna()
cont_short = (df["Close"]<df["MA50"]) & (df["RSI14"].between(35,60, inclusive="both")) & \
             ((df["RetracePct"]<=0.38) | (df["Close"]<df["LL10"])) & (df["BearEngulf"] | (df["Close"]<df["LL10"])) & df["H3"].notna()

df["Setup"] = np.select(
    [rev_long, rev_short, cont_long, cont_short],
    ["ReversalLong","ReversalShort","ContLong","ContShort"],
    default=""
)

# stops/targets and R/R (stock-level)
df["Side"] = df["Setup"].map({"ReversalLong":"long","ContLong":"long",
                              "ReversalShort":"short","ContShort":"short"})

# risk and target
df["Stop"]   = np.where(df["Side"]=="long",  df["L3"], df["H3"])
df["Risk"]   = np.where(df["Side"]=="long",  df["Close"]-df["Stop"], df["Stop"]-df["Close"])
df["Target"] = np.where(df["Side"]=="long",  df["Close"] + 1.25*df["Risk"], df["Close"] - 1.25*df["Risk"])
df["R_R"]    = np.where(df["Risk"]>0, 1.25, np.nan)

# quality filter
qual = df[(df["Setup"]!="") & (df["Risk"]>0) & (df["R_R"]>=1.25)].copy()
if qual.empty:
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    qual.to_csv(OUT_TRADES, index=False)
    pd.DataFrame(columns=["Setup","Side","horizon","n","avg","median","pos_rate","tgt_hit10","stp_hit10"]).to_csv(OUT_STATS, index=False)
    print("0 quality setups.")
    raise SystemExit()

# forward returns (signed; positive means correct direction)
for d in (5,10,15):
    r = (qual[f"Close_fut{d}"] - qual["Close"]) / qual["Close"]
    qual[f"ret{d}"] = np.where(qual["Side"]=="short", -r, r)
    qual[f"win{d}"] = qual[f"ret{d}"] > 0

# target/stop hit within 10 trading days using future High/Low
fh = qual["FwdHigh_10"]; fl = qual["FwdLow_10"]
tgt = qual["Target"]; stp = qual["Stop"]
# hits (ignoring path order due to daily data limitation)
tgt_hit = np.where(qual["Side"]=="long", fh>=tgt, fl<=tgt)
stp_hit = np.where(qual["Side"]=="long", fl<=stp, fh>=stp)
qual["tgt_hit10"] = tgt_hit
qual["stp_hit10"] = stp_hit

# outputs
keep = ["Date","Ticker","Group","Setup","Side","Open","High","Low","Close",
        "RSI14","MA50","MA200","H3","L3","Stop","Target","R_R",
        "ret5","ret10","ret15","win5","win10","win15","tgt_hit10","stp_hit10"]
qual_out = qual[keep].sort_values(["Date","Ticker"]).reset_index(drop=True)
OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
qual_out.to_csv(OUT_TRADES, index=False)

# aggregate stats
rows = []
for (setup, side) in qual_out.groupby(["Setup","Side"]).groups:
    sub = qual_out[(qual_out["Setup"]==setup) & (qual_out["Side"]==side)]
    for d in (5,10,15):
        vals = sub[f"ret{d}"].dropna()
        rows.append({
            "Setup": setup, "Side": side, "horizon": d,
            "n": int(vals.size),
            "avg": float(vals.mean()) if vals.size else np.nan,
            "median": float(vals.median()) if vals.size else np.nan,
            "pos_rate": float((vals>0).mean()) if vals.size else np.nan,
            "tgt_hit10": float(sub["tgt_hit10"].mean()) if len(sub) else np.nan,
            "stp_hit10": float(sub["stp_hit10"].mean()) if len(sub) else np.nan,
        })
pd.DataFrame(rows).to_csv(OUT_STATS, index=False)

print(f"Wrote {len(qual_out)} trades -> {OUT_TRADES}")
print(f"Stats -> {OUT_STATS}")