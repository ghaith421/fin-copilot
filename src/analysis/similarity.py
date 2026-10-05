import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from config import DATA_PROCESSED

# Import avec fallback (sentence-transformers pas installe sur Streamlit Cloud)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _ST_OK = True
except Exception:
    _ST_OK = False

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_embeddings = None
_news_index = None
_event_impacts = None


def _load_all():
    global _model, _embeddings, _news_index, _event_impacts

    if _model is None and _ST_OK:
        print("[sim] Chargement du modele...")
        _model = SentenceTransformer(MODEL_NAME)
    if _embeddings is None:
        _embeddings = np.load(f"{DATA_PROCESSED}/news_embeddings.npy")
    if _news_index is None:
        idx = pd.read_parquet(f"{DATA_PROCESSED}/news_index.parquet")
        idx["date_key"] = pd.to_datetime(idx["date"]).dt.tz_localize(None).dt.floor("D")
        idx = idx.drop_duplicates(subset=["date_key", "title", "tickers"]).reset_index(drop=True)
        _news_index = idx
    if _event_impacts is None:
        ev = pd.read_parquet(f"{DATA_PROCESSED}/event_impacts.parquet")
        ev["date_key"] = pd.to_datetime(ev["news_date"]).dt.tz_localize(None).dt.floor("D")
        ev = ev.drop_duplicates(subset=["date_key", "ticker"], keep="first").reset_index(drop=True)
        _event_impacts = ev


def find_similar_news(query_text, ticker=None, k=5):
    _load_all()

    if not _ST_OK:
        # Fallback : retourner les news les plus recentes du ticker
        result = _news_index.copy()
        if ticker is not None:
            result = result[result["tickers"] == ticker]
        result = result.sort_values("date", ascending=False).head(k).reset_index(drop=True)
        result["similarity"] = 0.0
        impacts = _event_impacts[["date_key", "ticker", "impact_j1", "impact_j5", "trade_date"]].copy()
        impacts = impacts.rename(columns={"ticker": "tickers"})
        result = result.merge(impacts, on=["date_key", "tickers"], how="left")
        return result

    q_emb = _model.encode([query_text], convert_to_numpy=True)
    mask = np.arange(len(_news_index))
    if ticker is not None:
        mask = _news_index[_news_index["tickers"] == ticker].index.values
        if len(mask) < k:
            mask = np.arange(len(_news_index))

    emb_subset = _embeddings[mask]
    sims = cosine_similarity(q_emb, emb_subset)[0]

    top_k_local = np.argsort(sims)[::-1][:k * 3]
    top_k_global = mask[top_k_local]
    top_sims = sims[top_k_local]

    result = _news_index.iloc[top_k_global].copy().reset_index(drop=True)
    result["similarity"] = top_sims
    result = result.drop_duplicates(subset=["title"], keep="first").head(k).reset_index(drop=True)

    impacts = _event_impacts[["date_key", "ticker", "impact_j1", "impact_j5", "trade_date"]].copy()
    impacts = impacts.rename(columns={"ticker": "tickers"})
    result = result.merge(impacts, on=["date_key", "tickers"], how="left")
    return result


def analyze_news_impact(query_text, ticker=None, k=10):
    similar = find_similar_news(query_text, ticker=ticker, k=k)
    valid_j1 = similar["impact_j1"].dropna()
    valid_j5 = similar["impact_j5"].dropna()
    return {
        "query": query_text,
        "ticker": ticker,
        "n_similar": len(similar),
        "n_with_j1": len(valid_j1),
        "n_with_j5": len(valid_j5),
        "impact_j1_mean": valid_j1.mean() if len(valid_j1) > 0 else None,
        "impact_j5_mean": valid_j5.mean() if len(valid_j5) > 0 else None,
        "hit_rate_j1": (valid_j1 > 0).mean() if len(valid_j1) > 0 else None,
        "hit_rate_j5": (valid_j5 > 0).mean() if len(valid_j5) > 0 else None,
        "similar_news": similar,
    }


def format_report(report):
    lines = []
    lines.append("=== ANALYSE D'IMPACT ===")
    lines.append(f"News analysee : {report['query'][:200]}")
    if report["ticker"]:
        lines.append(f"Ticker concerne : {report['ticker']}")
    lines.append(f"News similaires trouvees : {report['n_similar']}")
    if report["impact_j1_mean"] is not None:
        lines.append(f"\nA J+1 (n={report['n_with_j1']}) : rendement moyen {report['impact_j1_mean']*100:+.2f}%, taux de hausse {report['hit_rate_j1']*100:.0f}%")
    if report["impact_j5_mean"] is not None:
        lines.append(f"A J+5 (n={report['n_with_j5']}) : rendement moyen {report['impact_j5_mean']*100:+.2f}%, taux de hausse {report['hit_rate_j5']*100:.0f}%")
    lines.append("\nTop 5 news similaires :")
    for _, r in report["similar_news"].head(5).iterrows():
        j1 = f"{r['impact_j1']*100:+.2f}%" if pd.notna(r["impact_j1"]) else "N/A"
        j5 = f"{r['impact_j5']*100:+.2f}%" if pd.notna(r["impact_j5"]) else "N/A"
        lines.append(f"  - {r['title'][:80]} | J+1: {j1} | J+5: {j5}")
    return "\n".join(lines)


if __name__ == "__main__":
    query = "Nvidia announces new AI chip"
    print(analyze_news_impact(query, ticker="NVDA", k=10))