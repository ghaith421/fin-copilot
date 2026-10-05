import requests
import pandas as pd
import time
from datetime import datetime
from config import TICKERS, DATA_RAW

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

GDELT_QUERIES = {
    "AAPL": '"Apple Inc" OR "Apple stock"',
    "MSFT": '"Microsoft" OR "Microsoft stock"',
    "NVDA": '"Nvidia" OR "Nvidia stock"',
    "TSLA": '"Tesla" OR "Tesla stock"',
    "AMZN": '"Amazon" OR "Amazon stock"',
}

def fetch_gdelt(ticker, timespan="3m", max_records=250, retries=3):
    query = GDELT_QUERIES.get(ticker, ticker)
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "timespan": timespan,
        "sort": "datedesc",
    }
    for attempt in range(retries):
        try:
            r = requests.get(GDELT_URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"[gdelt] {ticker}: 429, pause {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            articles = data.get("articles", [])
            rows = []
            for a in articles:
                try:
                    dt = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ")
                except Exception:
                    dt = datetime.utcnow()
                rows.append({
                    "date": dt,
                    "title": a.get("title", ""),
                    "summary": a.get("title", ""),
                    "source": a.get("domain", "gdelt"),
                    "tickers": ticker,
                })
            print(f"[gdelt] {ticker}: {len(rows)} articles")
            return rows
        except Exception as ex:
            print(f"[gdelt] Erreur {ticker}: {ex}")
            time.sleep(3)
    return []

def ingest_gdelt():
    all_rows = []
    for i, t in enumerate(TICKERS):
        all_rows.extend(fetch_gdelt(t))
        if i < len(TICKERS) - 1:
            print("[gdelt] Pause 6s pour eviter le rate limit...")
            time.sleep(6)

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("[gdelt] Aucun article recupere")
        return df

    df = df.drop_duplicates(subset=["title"]).dropna(subset=["title"])
    out = f"{DATA_RAW}/gdelt_news.parquet"
    df.to_parquet(out)
    print(f"[gdelt] TOTAL {len(df)} articles sauvegardes -> {out}")
    print(df.head(5))
    return df

if __name__ == "__main__":
    ingest_gdelt()