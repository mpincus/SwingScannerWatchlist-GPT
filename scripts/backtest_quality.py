from pathlib import Path
import pandas as pd
import numpy as np

# ===== Config (percent-based stops/targets) =====
STOP_PCT   = 0.05   # 5% adverse move = stop
TARGET_PCT = 0.08   # 8% favorable move = target
RR_MIN     = 1.25   # require >= 1.25 R/R

# ===== Paths =====
SRC = Path("data/combined.csv")
OUT_DIR = Path("data"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TRADES = OUT_DIR / "quality_trades.csv"
OUT_STATS  = OUT_DIR / "quality_stats.csv"

# ===== Helpers =====
def rsi14(s: pd.Series) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1/14, adjust=False).mean()
    ad = dn.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-12)
    rs = au / ad
    return 100 - 100/(1+rs)

def rolling_prev_extrema(x: pd.Series, win: int, fn: str) -> pd.Series:
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
    span = (lookback_high - lookback_low).replace(0, np.nan)
    pct = (lookback_high - close) / span
    return pct.clip(lower=0, upper=1).fillna(0)

# ===== Load =====
if not SRC.exists():
    pd.DataFrame().to_csv(OUT_TRADES, index=False)
    pd.DataFrame().to_csv(OUT_STATS, index=False)
    raise SystemExit("combined.csv missing")

df = pd.read_csv(SRC, parse_dates=["Date"]).sort_values(["Ticker","Date"])
req = {"Date","Ticker","Group","Open","High","Low","Close"}
missing = req - set(df.columns)
if missing or df.empty:
    pd.DataFrame().to_csv(OUT_TRADES, index=False)
    pd.DataFrame().to_csv(OUT_STATS, index=False)
    raise SystemExit(f"bad combined.csv; missing={sorted(missing)} empty={df.empty}")

# ===== Features =====
g = df.groupby("Ticker", group_keys=False)
df["RSI14"] = g["Close"].transform(rsi14)
df["MA50"]  = g["Close"].transform(lambda s: s.rolling(50).mean())
df["MA200"] = g["Close"].transform(lambda s: s.rolling(200).mean())

bull_e, bear_e = engulfing_flags(df)
df["BullEngulf"] = bull_e
df["BearEngulf"] = bear_e

df["H3"] = g["High"].transform(lambda s: rolling_prev_extrema(s, 3, "max"))
df["L3"] = g["Low"] .transform(lambda s: rolling_prev_extrema(s, 3, "min"))

for d in (5,10,15):
    df[f"Close_fut{d}"] = g["Close"].shift(-d)

def fwd_roll_max(x, n): return x.shift(-1).rolling(n, min_periods=1).max()
def fwd_roll_min(x, n): return x.shift(-1).rolling(n, min_periods=1).min()
df["FwdHigh_10"] = g["High"].apply(lambda s: fwd_roll_max(s, 10))
df["FwdLow_10"]  = g["Low"] .apply(lambda s: fwd_roll_min(s, 10))

df["LH10"] = g["High"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).max())
df["LL10"] = g["Low"] .transform(lambda s: s.shift(1).rolling(10, min_periods=1).min())
df["RetracePct"] = percent_retrace(df["Close"], df["LH10"], df["LL10"])

# ===== Quality Gate =====
# Reversal Long: oversold AND RSI rising (no engulfing requirement)
df["RSI_up"] = df.groupby("Ticker")["RSI14"].diff() > 0
rev_long  = (df["Group"]=="oversold")   & (df["RSI14"]<=30) & (df["RSI_up"])

# Reversal Short: strict (overbought + bearish engulfing)
rev_short = (df["Group"]=="overbought") & (df["RSI14"]>=70) & df["BearEngulf"]

# Continuation Long
cont_long  = (df["Close"]>df["MA50"]) & (df["RSI14"].between(40,65, inclusive="both")) & \
             ((df["RetracePct"]<=0.38) | (df["Close"]>df["LH10"])) & (df["BullEngulf"] | (df["Close"]>df["LH10"]))

# Continuation Short
cont_short = (df["Close"]<df["MA50"]) & (df["RSI14"].between(35,60, inclusive="both")) & \
             ((df["RetracePct"]<=0.38) | (df["Close"]<df["LL10"])) & (df["BearEngulf"] | (df["Close"]<df["LL10"]))

df["Setup"] = np.select(
    [rev_long, rev_short, cont_long, cont_short],
    ["ReversalLong","ReversalShort","ContLong","ContShort"],
    default=""
)
df = df[df["Setup"]!=""].copy()
df["Side"] = df["Setup"].map({"ReversalLong":"long","ContLong":"long",
                              "ReversalShort":"short","ContShort":"short"})

# ===== Percent-based stop/target and R/R =====
close = df["Close"]
long_stop  = close*(1.0-STOP_PCT)
long_tgt   = close*(1.0+TARGET_PCT)
short_stop = close*(1.0+STOP_PCT)
short_tgt  = close*(1.0-TARGET_PCT)

df["Stop"]   = np.where(df["Side"]=="long",  long_stop, short_stop)
df["Target"] = np.where(df["Side"]=="long",  long_tgt,  short_tgt)
df["R_R"]    = TARGET_PCT / STOP_PCT

qual = df[df["R_R"]>=RR_MIN].copy()

# ===== Forward returns and hits =====
for d in (5,10,15):
    r = (qual[f"Close_fut{d}"] - qual["Close"]) / qual["Close"]
    qual[f"ret{d}"] = np.where(qual["Side"]=="short", -r, r)
    qual[f"win{d}"] = qual[f"ret{d}"] > 0

fh, fl = qual["FwdHigh_10"], qual["FwdLow_10"]
tgt, stp = qual["Target"], qual["Stop"]
qual["tgt_hit10"] = np.where(qual["Side"]=="long", fh>=tgt, fl<=tgt)
qual["stp_hit10"] = np.where(qual["Side"]=="long", fl<=stp, fh>=stp)

# ===== Write =====
keep = ["Date","Ticker","Group","Setup","Side","Open","High","Low","Close",
        "RSI14","MA50","MA200","Stop","Target","R_R",
        "ret5","ret10","ret15","win5","win10","win15","tgt_hit10","stp_hit10"]
out = qual[keep].sort_values(["Date","Ticker"]).reset_index(drop=True)
out.to_csv(OUT_TRADES, index=False)

rows = []
for (setup, side), sub in out.groupby(["Setup","Side"]):
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

print(f"Wrote {len(out)} trades -> {OUT_TRADES}")
print(f"Stats -> {OUT_STATS}")