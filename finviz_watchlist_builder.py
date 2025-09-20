#!/usr/bin/env python3
"""
Builds a tighter Finviz-based watchlist using refined, liquid filters.

Outputs:
  - oversold.csv, overbought.csv, pullbacks.csv, breakouts.csv
  - combined_watchlist.csv  (columns: Ticker,List)
  - combined_watchlist.txt  (same list in plain text for copy/paste)

Notes:
- Respects Finviz by paging gently with small delays.
- De-dupes by first appearance order (oversold → overbought → pullbacks → breakouts).
- You can tweak filters in the SCREENS dict below.
"""

import time, csv, re, sys
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

# -------------
# Refined filters
# -------------
# Common filter fragments (Finviz screener URL params):
#  - cap_midover        : Market cap ≥ $2B (mid/large caps → better options liquidity)
#  - sh_opt_option      : Optionable
#  - sh_price_o10       : Price > $10 (avoid pennies/low tiers)
#  - sh_avgvol_o1000    : Avg volume > 1M shares
#  - geo_usa            : U.S.-listed (avoid ADR liquidity quirks). Remove if you want global
#  - ta_rsi_os30        : RSI ≤ 30 (oversold)
#  - ta_rsi_ob70        : RSI ≥ 70 (overbought)
#  - ta_highlow52w_a5   : Within 5% of 52-week high (breakout proximity)
#  - ta_sma50_pa        : Price above 50 SMA (trend confirmation)
#  - ta_sma50_pb        : Price near/pulling back to 50 SMA
#  - ta_perf_4w10o      : 4-week perf > +10% (momentum filter for pullbacks)

BASE = "v=111&f=cap_midover,sh_opt_option,sh_price_o10,sh_avgvol_o1000,geo_usa"

SCREENS: Dict[str, str] = {
    # True oversold candidates
    "oversold":  f"https://finviz.com/screener.ashx?{BASE},ta_rsi_os30",

    # True overbought candidates
    "overbought": f"https://finviz.com/screener.ashx?{BASE},ta_rsi_ob70",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
S = requests.Session(); S.headers.update(HEADERS)

# -------------
# Scraping helpers
# -------------

def extract_tickers(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "quote.ashx?t=" in href:
            t = a.text.strip().upper()
            if re.fullmatch(r"[A-Z.]{1,5}", t) and t not in seen:
                seen.add(t); out.append(t)
    return out

def paged(url: str, page_index: int) -> str:
    start = page_index * 20 + 1
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}r={start}"

def scrape_list(name: str, url: str, delay: float = 1.0, max_pages: int = 75) -> List[str]:
    tickers, i = [], 0
    while i < max_pages:
        u = paged(url, i)
        r = S.get(u, timeout=25)
        if r.status_code != 200:
            break
        part = extract_tickers(r.text)
        if i > 0 and not part:
            break
        for t in part:
            if t not in tickers:
                tickers.append(t)
        i += 1
        time.sleep(delay)
        if len(part) < 20:
            break
    return tickers

def write_csv(path: str, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if rows and isinstance(rows[0], dict):
            w.writerow(rows[0].keys())
            for r in rows:
                w.writerow([r[k] for k in rows[0].keys()])
        else:
            w.writerow(["Ticker"])
            for t in rows:
                w.writerow([t])

def write_txt(path: str, rows: List[dict]):
    with open(path, "w") as f:
        f.write("Ticker,List\n")
        for r in rows:
            f.write(f"{r['Ticker']},{r['List']}\n")

# -------------
# Main
# -------------

def main():
    all_rows = []
    order = ["oversold", "overbought", "pullbacks", "breakouts"]

    for name in order:
        url = SCREENS[name]
        try:
            print(f"Scraping {name}: {url}")
            tickers = scrape_list(name, url)
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")
            tickers = []
        write_csv(f"{name}.csv", tickers)
        for t in tickers:
            all_rows.append({"Ticker": t, "List": name})

    seen = set(); deduped = []
    for r in all_rows:
        if r["Ticker"] not in seen:
            seen.add(r["Ticker"])
            deduped.append(r)

    write_csv("combined_watchlist.csv", deduped)
    write_txt("combined_watchlist.txt", deduped)

    print(f"Done. Oversold:{len([r for r in deduped if r['List']=='oversold'])} | "
          f"Overbought:{len([r for r in deduped if r['List']=='overbought'])} | "
          f"Pullbacks:{len([r for r in deduped if r['List']=='pullbacks'])} | "
          f"Breakouts:{len([r for r in deduped if r['List']=='breakouts'])}")
    print(f"Combined (de-duped): {len(deduped)} tickers")

if __name__ == "__main__":
    main()
