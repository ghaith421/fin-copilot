import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from config import DATA_RAW

def score_news():
    path = f"{DATA_RAW}/news.parquet"
    df = pd.read_parquet(path)
    print(f"[sentiment] {len(df)} articles a analyser")

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for txt in tqdm(df["title"].fillna("") + ". " + df["summary"].fillna(""), desc="Scoring"):
        try:
            s = analyzer.polarity_scores(txt[:1000])
            scores.append(s["compound"])
        except Exception:
            scores.append(0.0)

    df["sentiment"] = scores
    out = f"{DATA_RAW}/news_scored.parquet"
    df.to_parquet(out)
    print(f"[sentiment] Sauvegarde -> {out}")
    print(df[["date", "title", "tickers", "sentiment"]].head(10))
    return df

if __name__ == "__main__":
    score_news()