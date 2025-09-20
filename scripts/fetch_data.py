# scripts/fetch_data.py
# Requires: yfinance, pandas, pytz
import pandas as pd, yfinance as yf, pytz, time
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path("data")
WATCHLIST = Path("combined_watchlist.csv")
END = datetime.now(pytz.timezone("US/Eastern"))
START = END - timedelta(days=160)  # ~4-5 months calendar to cover ~90 trading days
INTERVAL = "1d"
MAX_RETRIES = 3
SLEEP = 2.0
CHUNK = 25

GROUP_MAP = {
    "oversold": "oversold",
    "overbought": "overbought",
    "breakouts": "breakouts",
    "pullbacks": "breakouts",  # merge pullbacks into breakouts
}

def read_watchlist():
    df = pd.read_csv(WATCHLIST)
    df.columns = [c.strip().capitalize() for c in df.columns]  # Ticker, List
    df["List"] = df["List"].str.strip().str.lower()
    df["Ticker"] = df["Ticker"].str.strip().str.upper()
    return df

def buckets(df):
    out = {"oversold": set(), "overbought": set(), "breakouts": set()}
    for _, r in df.iterrows():
        kind = GROUP_MAP.get(r["List"])
        if not kind:
            continue
        out[kind].add(r["Ticker"])
    return {k: sorted(list(v)) for k, v in out.items()}

def dl(symbols):
    for a in range(1, MAX_RETRIES+1):
        try:
            return yf.download(
                tickers=symbols,
                start=START.strftime("%Y-%m-%d"),
                end=(END+timedelta(days=1)).strftime("%Y-%m-%d"),
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
    recs=[]
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date","Ticker","Group","Open","High","Low","Close","Adj Close","Volume"])
    for s in syms:
        if s not in df.columns.get_level_values(0): 
            continue
        sub=df[s].copy()
        if sub.empty: 
            continue
        sub=sub.reset_index()
        sub["Ticker"]=s
        sub["Group"]=group
        sub=sub[["Date","Ticker","Group","Open","High","Low","Close","Adj Close","Volume"]]
        recs.append(sub)
    if not recs:
        return pd.DataFrame(columns=["Date","Ticker","Group","Open","High","Low","Close","Adj Close","Volume"])
    out=pd.concat(recs, ignore_index=True)
    out["Date"]=pd.to_datetime(out["Date"], utc=True).dt.tz_convert("US/Eastern").dt.date
    out=out.sort_values(["Ticker","Date"]).drop_duplicates(["Ticker","Date"])
    return out

def fetch_and_save(name, tickers):
    if not tickers:
        print(f"[{name}] no tickers. skipped.")
        return pd.DataFrame()
    parts=[]
    for i in range(0, len(tickers), CHUNK):
        raw=dl(tickers[i:i+CHUNK])
        parts.append(to_long(raw, tickers[i:i+CHUNK], name))
    df=pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df=df[pd.notnull(df["Close"])]
    df=df[df["Volume"]>=0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(outpath, index=False)
    print(f"[{name}] saved {len(df)} rows to {outpath}")
    return df

def main():
    wl = read_watchlist()
    b = buckets(wl)
    all_frames=[]
    for name, tickers in b.items():
        df = fetch_and_save(name, tickers)
        if not df.empty:
            all_frames.append(df)
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.sort_values(["Group","Ticker","Date"])
        outpath = OUTPUT_DIR / "combined.csv"
        combined.to_csv(outpath, index=False)
        print(f"[combined] saved {len(combined)} rows to {outpath}")

if __name__=="__main__":
    main()
