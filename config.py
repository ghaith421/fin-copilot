import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _get_secret(key, default=""):
    """Recupere un secret depuis l'environnement ou Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


TICKERS = _get_secret("TICKERS", "AAPL,MSFT,NVDA,TSLA,AMZN").split(",")
START_DATE = _get_secret("START_DATE", "2018-01-01")
GROQ_API_KEY = _get_secret("GROQ_API_KEY", "")
NEWSAPI_KEY = _get_secret("NEWSAPI_KEY", "")
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY", "")

DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
DATA_MODELS = "data/models"

for p in [DATA_RAW, DATA_PROCESSED, DATA_MODELS]:
    os.makedirs(p, exist_ok=True)