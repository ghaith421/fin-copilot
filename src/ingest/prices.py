import yfinance as yf
import pandas as pd
from config import TICKERS, START_DATE, DATA_RAW

def download_prices(tickers=TICKERS, start=START_DATE):
    print(f"[prices] Telechargement pour {tickers} depuis {start}...")
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1).rename_axis(["date", "ticker"]).reset_index()
    else:
        df = df.reset_index()
        df["ticker"] = tickers[0]

    df.columns = [str(c).lower() for c in df.columns]
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    df = df.dropna()

    out = f"{DATA_RAW}/prices.parquet"
    df.to_parquet(out)
    print(f"[prices] {len(df)} lignes sauvegardees -> {out}")
    return df

def download_spy(start=START_DATE):
    print(f"[spy] Telechargement du S&P 500 (SPY) depuis {start}...")
    df = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df[["date", "close"]].rename(columns={"close": "spy_close"})
    df = df.dropna()

    out = f"{DATA_RAW}/spy.parquet"
    df.to_parquet(out)
    print(f"[spy] {len(df)} lignes sauvegardees -> {out}")
    return df

if __name__ == "__main__":
    download_prices()
    download_spy()