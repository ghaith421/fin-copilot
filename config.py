import os
from dotenv import load_dotenv

load_dotenv()

TICKERS = os.getenv("TICKERS", "AAPL,MSFT,NVDA,TSLA,AMZN").split(",")
START_DATE = os.getenv("START_DATE", "2018-01-01")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
DATA_MODELS = "data/models"

for p in [DATA_RAW, DATA_PROCESSED, DATA_MODELS]:
    os.makedirs(p, exist_ok=True)

print("[config] OK - projet initialise")