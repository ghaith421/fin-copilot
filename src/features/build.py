import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED

def add_technical(df):
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker")["close"]

    df["ret_1"] = g.pct_change(1)
    df["ret_5"] = g.pct_change(5)
    df["ret_20"] = g.pct_change(20)
    df["vol_20"] = g.pct_change().groupby(df["ticker"]).transform(lambda s: s.rolling(20).std())

    df["ma_10"] = g.transform(lambda s: s.rolling(10).mean())
    df["ma_50"] = g.transform(lambda s: s.rolling(50).mean())
    df["ma_ratio"] = df["ma_10"] / df["ma_50"]

    delta = g.diff()
    gain = delta.clip(lower=0).groupby(df["ticker"]).transform(lambda s: s.rolling(14).mean())
    loss = (-delta.clip(upper=0)).groupby(df["ticker"]).transform(lambda s: s.rolling(14).mean())
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    ema_12 = g.transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_26 = g.transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    ma20 = g.transform(lambda s: s.rolling(20).mean())
    std20 = g.transform(lambda s: s.rolling(20).std())
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    df["vol_ma20"] = df.groupby("ticker")["volume"].transform(lambda s: s.rolling(20).mean())
    df["vol_ratio"] = df["volume"] / (df["vol_ma20"] + 1)

    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month

    return df

def add_market_context(df, spy):
    spy = spy.copy()
    spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None).dt.floor("D")
    spy = spy.sort_values("date")
    spy["spy_ret_5"] = spy["spy_close"].pct_change(5)
    spy["spy_ret_20"] = spy["spy_close"].pct_change(20)
    spy["spy_vol_20"] = spy["spy_close"].pct_change().rolling(20).std()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.floor("D")
    df = df.merge(spy[["date", "spy_ret_5", "spy_ret_20", "spy_vol_20"]], on="date", how="left")
    return df

def add_sentiment(prices, news, decay_days=5):
    news = news.copy()
    news["news_date"] = pd.to_datetime(news["date"]).dt.tz_localize(None).dt.floor("D")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.floor("D")

    # Rattacher chaque news au prochain jour de trading (gere les weekends)
    news_list = []
    for ticker in news["tickers"].unique():
        tnews = news[news["tickers"] == ticker].copy()
        tprices = sorted(prices[prices["ticker"] == ticker]["date"].unique())
        if len(tprices) == 0:
            continue
        tprices_arr = pd.to_datetime(tprices)
        tnews["trade_date"] = tnews["news_date"].apply(
            lambda d: next((td for td in tprices_arr if td >= d), None)
        )
        tnews = tnews.dropna(subset=["trade_date"])
        tnews["date"] = tnews["trade_date"]
        tnews["ticker"] = ticker
        news_list.append(tnews[["date", "ticker", "sentiment"]])

    if not news_list:
        for col in ["sent_mean", "sent_count", "sent_sum", "sent_ma3"]:
            prices[col] = 0.0
        return prices

    news_clean = pd.concat(news_list, ignore_index=True)

    agg = (news_clean.groupby(["ticker", "date"])["sentiment"]
                .agg(["mean", "count", "sum"])
                .reset_index()
                .rename(columns={"mean": "sent_mean",
                                 "count": "sent_count", "sum": "sent_sum"}))

    df = prices.merge(agg, on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for col in ["sent_mean", "sent_count", "sent_sum"]:
        df[col] = df.groupby("ticker")[col].ffill(limit=decay_days)
        df[col] = df[col].fillna(0)

    df["sent_ma3"] = df.groupby("ticker")["sent_mean"].transform(lambda s: s.rolling(3).mean())
    df["sent_ma3"] = df["sent_ma3"].fillna(0)
    return df

def add_labels(df, horizons=(1, 5)):
    df = df.sort_values(["ticker", "date"]).copy()
    for h in horizons:
        df[f"future_ret_{h}"] = df.groupby("ticker")["close"].shift(-h) / df["close"] - 1
        df[f"target_{h}"] = (df[f"future_ret_{h}"] > 0).astype(int)
    return df

def build():
    prices = pd.read_parquet(f"{DATA_RAW}/prices.parquet")
    news = pd.read_parquet(f"{DATA_RAW}/news_scored.parquet")
    spy = pd.read_parquet(f"{DATA_RAW}/spy.parquet")
    print(f"[features] {len(prices)} prix, {len(news)} news, {len(spy)} SPY")

    df = add_technical(prices)
    df = add_market_context(df, spy)
    df = add_sentiment(df, news)
    df = add_labels(df)

    df = df.dropna(subset=["ret_20", "ma_ratio", "rsi", "macd", "bb_position"])
    df = df.fillna(0)

    nz = (df["sent_mean"] != 0).sum()
    print(f"[features] lignes avec sentiment non nul : {nz} / {len(df)}")
    print(f"[features] pourcentage : {100 * nz / len(df):.2f}%")

    out = f"{DATA_PROCESSED}/features.parquet"
    df.to_parquet(out)
    print(f"[features] {df.shape} sauvegarde -> {out}")
    return df

if __name__ == "__main__":
    build()