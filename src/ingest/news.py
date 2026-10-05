import feedparser
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from config import TICKERS, DATA_RAW, NEWSAPI_KEY

RSS_GENERAL = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.investing.com/rss/news_25.rss",
    "https://finance.yahoo.com/news/rssindex",
]

TICKER_KEYWORDS = {
    "AAPL": ["apple", "aapl", "iphone", "ipad", "macbook", "tim cook"],
    "MSFT": ["microsoft", "msft", "windows", "azure", "xbox", "satya nadella"],
    "NVDA": ["nvidia", "nvda", "geforce", "cuda", "jensen huang"],
    "TSLA": ["tesla", "tsla", "elon musk", "model 3", "model y", "cybertruck"],
    "AMZN": ["amazon", "amzn", "aws", "bezos", "prime"],
}

NEWSAPI_QUERIES = {
    "AAPL": "Apple OR AAPL OR iPhone",
    "MSFT": "Microsoft OR MSFT OR Azure",
    "NVDA": "Nvidia OR NVDA OR GeForce",
    "TSLA": "Tesla OR TSLA OR Elon Musk",
    "AMZN": "Amazon OR AMZN OR AWS",
}

def fetch_rss(url, forced_ticker=None):
    rows = []
    try:
        feed = feedparser.parse(url)
        for e in feed.entries:
            date = datetime(*e.published_parsed[:6]) if hasattr(e, "published_parsed") and e.published_parsed else datetime.utcnow()
            rows.append({
                "date": date,
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "source": url,
                "forced_ticker": forced_ticker,
            })
    except Exception as ex:
        print(f"[news] Erreur RSS {url}: {ex}")
    return rows

def fetch_newsapi(ticker, weeks_back=4, page_size=100):
    """Recupere les news par tranches d'une semaine sur N semaines."""
    if not NEWSAPI_KEY:
        print("[news] NEWSAPI_KEY manquante, NewsAPI ignore")
        return []

    rows = []
    query = NEWSAPI_QUERIES.get(ticker, ticker)
    today = datetime.utcnow()

    for w in range(weeks_back):
        to_date = today - timedelta(days=w * 7)
        from_date = to_date - timedelta(days=7)

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": page_size,
            "apiKey": NEWSAPI_KEY,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
            if data.get("status") != "ok":
                print(f"[news] NewsAPI erreur {ticker} sem {w}: {data.get('message')}")
                continue
            count = 0
            for a in data.get("articles", []):
                try:
                    dt = datetime.strptime(a["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    dt = datetime.utcnow()
                rows.append({
                    "date": dt,
                    "title": a.get("title") or "",
                    "summary": a.get("description") or "",
                    "source": a.get("source", {}).get("name", "newsapi"),
                    "forced_ticker": ticker,
                })
                count += 1
            print(f"[news]   {ticker} semaine -{w}: {count} articles")
        except Exception as ex:
            print(f"[news] Erreur NewsAPI {ticker} sem {w}: {ex}")
        time.sleep(1.5)

    return rows

def tag_tickers(df, tickers=TICKERS):
    def match(row):
        text = (row["title"] or "").lower() + " " + (row["summary"] or "").lower()
        if pd.notna(row.get("forced_ticker")) and row["forced_ticker"] in tickers:
            return [row["forced_ticker"]]
        found = []
        for t in tickers:
            keywords = TICKER_KEYWORDS.get(t, [t.lower()])
            if any(k in text for k in keywords):
                found.append(t)
        return found
    df["tickers"] = df.apply(match, axis=1)
    df = df.explode("tickers").dropna(subset=["tickers"])
    return df

def ingest_news():
    all_rows = []

    # 1. RSS generaux
    print("[news] Flux RSS generaux...")
    for url in RSS_GENERAL:
        all_rows.extend(fetch_rss(url))
    print(f"[news] {len(all_rows)} articles des flux generaux")

    # 2. RSS Yahoo par ticker
    print("[news] Flux RSS Yahoo par ticker...")
    for t in TICKERS:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
        rows = fetch_rss(url, forced_ticker=t)
        print(f"[news]   {t} RSS: {len(rows)} articles")
        all_rows.extend(rows)
        time.sleep(1)

    # 3. NewsAPI par ticker sur 4 semaines
    print("[news] NewsAPI par ticker (peut prendre 2-3 minutes)...")
    for t in TICKERS:
        rows = fetch_newsapi(t, weeks_back=4, page_size=100)
        all_rows.extend(rows)

    # Nettoyage
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("[news] Aucun article recupere")
        return df
    df = df.drop_duplicates(subset=["title"]).dropna(subset=["title"])
    df = tag_tickers(df)

    out = f"{DATA_RAW}/news.parquet"
    df.to_parquet(out)
    print(f"[news] TOTAL {len(df)} articles sauvegardes -> {out}")
    print(df["tickers"].value_counts())
    return df

if __name__ == "__main__":
    ingest_news()