import pandas as pd
from pathlib import Path

COMBINED = Path("data/combined.csv")
SIGNALS = Path("data/signals.csv")
OUT = Path("data/signals_quality.csv")

def main():
    df = pd.read_csv(COMBINED, parse_dates=["Date"])
    sig = pd.read_csv(SIGNALS, parse_dates=["Date"])

    # Merge OHLC + RSI into signals
    merged = pd.merge(
        sig,
        df[["Date","Ticker","Open","High","Low","Close","RSI14"]],
        on=["Date","Ticker"],
        how="left"
    )

    quality = []

    for _, row in merged.iterrows():
        rsi = row["RSI14"]
        side = row["Side"]
        trigger = row["Trigger"]
        o,h,l,c = row["Open"], row["High"], row["Low"], row["Close"]

        # RSI extreme gate
        if side == "long" and not (rsi <= 30):
            continue
        if side == "short" and not (rsi >= 70):
            continue

        # Reversal candle gate (simple engulfing check)
        prev = df[(df["Ticker"]==row["Ticker"]) & (df["Date"] < row["Date"])].sort_values("Date").tail(1)
        if prev.empty:
            continue
        prev_o, prev_c = prev["Open"].values[0], prev["Close"].values[0]

        engulf_ok = False
        if side == "long":
            if c > o and prev_c < prev_o and c > prev_o and o < prev_c:
                engulf_ok = True
        if side == "short":
            if c < o and prev_c > prev_o and c < prev_o and o > prev_c:
                engulf_ok = True
        if not engulf_ok:
            continue

        # Risk/Reward check
        if side == "long":
            stop = df[(df["Ticker"]==row["Ticker"]) & (df["Date"] < row["Date"])].tail(3)["Low"].min()
            risk = c - stop
            target = c + risk * 1.25
            rr = (target - c) / risk if risk > 0 else 0
        else:
            stop = df[(df["Ticker"]==row["Ticker"]) & (df["Date"] < row["Date"])].tail(3)["High"].max()
            risk = stop - c
            target = c - risk * 1.25
            rr = (c - target) / risk if risk > 0 else 0

        if rr < 1.25:
            continue

        row["R_R"] = rr
        quality.append(row)

    out = pd.DataFrame(quality)
    out.to_csv(OUT, index=False)
    print(f"{len(out)} quality signals written to {OUT}")

if __name__ == "__main__":
    main()