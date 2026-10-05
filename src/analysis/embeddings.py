import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import DATA_RAW, DATA_PROCESSED

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_embeddings():
    print(f"[embeddings] Chargement du modele {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("[embeddings] Chargement des news...")
    news = pd.read_parquet(f"{DATA_RAW}/news_scored.parquet")
    news = news.reset_index(drop=True)

    # Texte a encoder : titre + resume
    texts = (news["title"].fillna("") + ". " + news["summary"].fillna("")).tolist()
    texts = [t.strip()[:500] for t in texts]

    print(f"[embeddings] Calcul des embeddings pour {len(texts)} news...")
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    print(f"[embeddings] Shape : {embeddings.shape}")

    # Sauvegarder les embeddings separement (numpy)
    np.save(f"{DATA_PROCESSED}/news_embeddings.npy", embeddings)

    # Sauvegarder l'index des news pour retrouver le texte
    news_index = news[["date", "title", "summary", "source", "tickers", "sentiment"]].copy()
    news_index.to_parquet(f"{DATA_PROCESSED}/news_index.parquet")

    print(f"[embeddings] Embeddings sauvegardes -> {DATA_PROCESSED}/news_embeddings.npy")
    print(f"[embeddings] Index news sauvegarde -> {DATA_PROCESSED}/news_index.parquet")
    return embeddings, news_index


if __name__ == "__main__":
    build_embeddings()