# scripts/fetch_data.py
# Requires: yfinance, pandas, pytz
import pandas as pd, yfinance as yf, pytz, time, os
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path("data")
WATCHLIST = Path(os.getenv("WATCHLIST_PATH", "combined_watchlist.csv"))
END = datetime.now(pytz.timezone("US/Eastern"))
START = END - timedelta(days=160)  # ~4–5 months calendar for ~90 trading days
INTERVAL = "1d"
MAX_RETRIES = 3
SLEEP = 2.0
CHUNK = 25

# Accept common variants
GROUP_MAP = {
    "oversold": "oversold",
    "overbought": "overbought",
    "breakouts": "breakouts",
    "breakout": "breakouts",
    "pullbacks": "breakouts",
    "pullback": "breakouts",
}

def read_watchlist():
    df = pd.read_csv(WATCHLIST)
    df.columns = [c.strip().capitalize() for c in df.columns]  # Ticker, List
    df["List"] = df["List"].astype(str).str.strip().str.lower()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()  # fixed: .str.upper()
    print("List value counts:", df["List"].value_counts().to_dict())
    return df

def buckets(df):
    out = {"oversold": set(), "overbought": set(), "breakouts": set()}
    unknown = set()
    for _, r in df.iterrows():
        key = GROUP_MAP.get(r["List"])
        if key:
            out[key].add(r["Ticker"])
        else:
            unknown.add(r["List"])
    if unknown:
        print("Unknown categories ignored:", sorted(unknown))
    for k, v in out.items():
        print(f"[bucket] {k}: {len(v)} tickers")
    return {k: sorted(v) for k, v in out.items()}

def dl(symbols):
    for a in range(1, MAX_RETRIES + 1):
        try:
            return yf.download(
                tickers=symbols,
                start=START.strftime("%Y-%m-%d"),
                end=(END + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=INTERVAL,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            if a == MAX_RETRIES:
                raise
            time.sleep(SLEEP)

def to_long(df, syms, group):
    cols = ["Date", "Ticker", "Group", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    rec = []
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    for s in syms:
        if s not in df.columns.get_level_values(0):
            continue
        sub = df[s].copy()
        if sub.empty:
            continue
        sub = sub.reset_index()
        sub["Ticker"] = s
        sub["Group"] = group
        sub = sub[cols]
        rec.append(sub)
    if not rec:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rec, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], utc=True).dt.tz_convert("US/Eastern").dt.date
    return out.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"])

def fetch_and_save(name, tickers):
    if not tickers:
        print(f"[{name}] empty. skip.")
        return pd.DataFrame()
    parts = []
    for i in range(0, len(tickers), CHUNK):
        raw = dl(tickers[i:i + CHUNK])
        parts.append(to_long(raw, tickers[i:i + CHUNK], name))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df = df[pd.notnull(df["Close"])]
    df = df[df["Volume"] >= 0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[{name}] rows: {len(df)} -> {path}")
    return df

def main():
    wl = read_watchlist()
    b = buckets(wl)
    frames = []
    for name, tickers in b.items():
        frames.append(fetch_and_save(name, tickers))
    frames = [f for f in frames if not f.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True).sort_values(["Group", "Ticker", "Date"])
        path = OUTPUT_DIR / "combined.csv"
        combined.to_csv(path, index=False)
        print(f"[combined] rows: {len(combined)} -> {path}")

if __name__ == "__main__":
    main()