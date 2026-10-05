import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED


def compute_event_impacts(horizons=(1, 5, 20)):
    """
    Pour chaque news, calcule le rendement de l'action a J+1, J+5, J+20
    apres la publication.
    """
    print("[event] Chargement des donnees...")
    news = pd.read_parquet(f"{DATA_RAW}/news_scored.parquet")
    prices = pd.read_parquet(f"{DATA_RAW}/prices.parquet")

    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None).dt.floor("D")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.floor("D")

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Pre-calculer les rendements futurs pour chaque (ticker, date)
    for h in horizons:
        prices[f"ret_{h}"] = prices.groupby("ticker")["close"].transform(
            lambda s: s.shift(-h) / s - 1
        )

    results = []
    total = len(news)
    print(f"[event] Calcul d'impact sur {total} news...")

    for i, row in news.iterrows():
        ticker = row["tickers"]
        ndate = row["date"]

        # Trouver le prochain jour de trading >= date de la news
        sub = prices[prices["ticker"] == ticker]
        if len(sub) == 0:
            continue
        future = sub[sub["date"] >= ndate]
        if len(future) == 0:
            continue

        entry = future.iloc[0]
        record = {
            "news_date": ndate,
            "trade_date": entry["date"],
            "ticker": ticker,
            "title": row["title"],
            "sentiment": row["sentiment"],
        }
        for h in horizons:
            record[f"impact_j{h}"] = entry.get(f"ret_{h}", np.nan)

        results.append(record)

    df = pd.DataFrame(results)
    df = df.dropna(subset=[f"impact_j{h}" for h in horizons], how="all")

    out = f"{DATA_PROCESSED}/event_impacts.parquet"
    df.to_parquet(out)
    print(f"[event] {len(df)} events calcules -> {out}")

    # Stats globales
    print("\n[event] === STATISTIQUES GLOBALES ===")
    for h in horizons:
        col = f"impact_j{h}"
        valid = df[col].dropna()
        if len(valid) > 0:
            print(f"J+{h} : moyenne {valid.mean()*100:+.2f}% | "
                  f"median {valid.median()*100:+.2f}% | "
                  f"hit rate {(valid > 0).mean()*100:.1f}% "
                  f"(n={len(valid)})")

    return df


def analyze_by_sentiment(df, horizons=(1, 5, 20)):
    """Analyse l'impact selon le sentiment (positif / neutre / negatif)."""
    print("\n[event] === IMPACT PAR SENTIMENT ===")
    df = df.copy()

    def cat(s):
        if s > 0.2:
            return "Positif"
        elif s < -0.2:
            return "Negatif"
        return "Neutre"

    df["sent_cat"] = df["sentiment"].apply(cat)

    for h in horizons:
        col = f"impact_j{h}"
        print(f"\n--- J+{h} ---")
        stats = df.groupby("sent_cat")[col].agg(["mean", "count"])
        for cat_name in ["Positif", "Neutre", "Negatif"]:
            if cat_name in stats.index:
                m = stats.loc[cat_name, "mean"]
                n = stats.loc[cat_name, "count"]
                print(f"  {cat_name:8s}: {m*100:+6.2f}%  (n={int(n)})")

    return df


if __name__ == "__main__":
    df = compute_event_impacts()
    df = analyze_by_sentiment(df)