#!/usr/bin/env python3
"""
Scrapes 4 Finviz screeners from your rules and writes:
- oversold.csv, overbought.csv, pullbacks.csv, breakouts.csv
- combined_watchlist.csv (Ticker,List), de-duped by first appearance.

Note: Be respectful of Finviz; this script uses small delays and light parsing.
"""

import time, math, csv, re
import requests
from bs4 import BeautifulSoup

SCREENS = {
    "oversold":  "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o1000,sh_price_o40,sh_opt_option,ta_rsi_os30",
    "overbought":"https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o1000,sh_price_o40,sh_opt_option,ta_rsi_ob70",
    "pullbacks": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o1000,sh_price_o40,sh_opt_option,ta_perf_4w20o,ta_sma50_pb",
    "breakouts": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o1000,sh_price_o40,sh_opt_option,ta_highlow52w_nh",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}
S = requests.Session(); S.headers.update(HEADERS)

def extract_tickers(html: str):
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "quote.ashx?t=" in href:
            t = a.text.strip().upper()
            if t.isalpha() and 1 <= len(t) <= 5 and t not in seen:
                seen.add(t); out.append(t)
    return out

def paged(url: str, i: int) -> str:
    start = i*20 + 1
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}r={start}"

def scrape_list(name: str, url: str, delay: float = 1.0, max_pages: int = 50):
    tickers, i = [], 0
    while i < max_pages:
        u = paged(url, i)
        r = S.get(u, timeout=25)
        if r.status_code != 200: break
        part = extract_tickers(r.text)
        if i > 0 and not part: break
        for t in part:
            if t not in tickers: tickers.append(t)
        i += 1
        time.sleep(delay)
        # Heuristic stop: less than 20 new items likely last page
        if len(part) < 20: break
    return tickers

def write_csv(path: str, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if rows and isinstance(rows[0], dict):
            w.writerow(rows[0].keys())
            for r in rows: w.writerow([r[k] for k in rows[0].keys()])
        else:
            w.writerow(["Ticker"])
            for t in rows: w.writerow([t])

def main():
    all_rows = []
    for name, url in SCREENS.items():
        try:
            tickers = scrape_list(name, url)
        except Exception as e:
            tickers = []
        write_csv(f"{name}.csv", tickers)
        for t in tickers:
            all_rows.append({"Ticker": t, "List": name})

    # De-dupe by first appearance
    seen = set(); deduped = []
    for r in all_rows:
        if r["Ticker"] not in seen:
            seen.add(r["Ticker"]); deduped.append(r)

    write_csv("combined_watchlist.csv", deduped)

if __name__ == "__main__":
    main()