import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from config import DATA_PROCESSED, DATA_MODELS, DATA_RAW

try:
    from src.rag.chat import ask
    CHATBOT_OK = True
except Exception as e:
    CHATBOT_OK = False
    CHAT_ERROR = str(e)

st.set_page_config(page_title="Fin Copilot", layout="wide")
st.title("💹 Fin Copilot — Aide a la decision")
st.caption("Outil pedagogique. Pas un conseil financier.")

FEATURES = [
    "ret_1", "ret_5", "ret_20", "vol_20",
    "ma_ratio", "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "vol_ratio", "range_pct",
    "day_of_week", "month",
    "spy_ret_5", "spy_ret_20", "spy_vol_20",
    "sent_mean", "sent_sum", "sent_count", "sent_ma3",
]

@st.cache_data
def load_features():
    return pd.read_parquet(f"{DATA_PROCESSED}/features.parquet")

@st.cache_data
def load_news():
    return pd.read_parquet(f"{DATA_RAW}/news_scored.parquet")

@st.cache_resource
def load_model():
    return joblib.load(f"{DATA_MODELS}/xgb_h5.joblib")

df = load_features()
news = load_news()
model = load_model()

tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chatbot"])

with tab1:
    st.sidebar.header("Filtres")
    ticker = st.sidebar.selectbox("Action", sorted(df["ticker"].unique()))
    sub = df[df["ticker"] == ticker].sort_values("date").copy()

    sub = sub.dropna(subset=FEATURES)
    if len(sub) > 0:
        sub["proba"] = model.predict_proba(sub[FEATURES].values)[:, 1]
    else:
        sub["proba"] = 0.0

    last = sub.iloc[-1] if len(sub) > 0 else None

    col1, col2, col3, col4 = st.columns(4)
    if last is not None:
        col1.metric("Dernier prix", f"{last['close']:.2f} $")
        col2.metric("Proba hausse (J+5)", f"{last['proba']:.1%}")
        col3.metric("Sentiment moyen", f"{last['sent_mean']:.2f}")
        col4.metric("RSI", f"{last['rsi']:.1f}")

    st.subheader(f"Prix et signaux — {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"], name="Prix", line=dict(color="#2E86DE")))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Probabilite de hausse (J+5)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sub["date"], y=sub["proba"], name="Proba", line=dict(color="#27AE60")))
    fig2.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig2.update_layout(height=300, yaxis_range=[0, 1], margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader(f"News recentes — {ticker}")
    tnews = news[news["tickers"] == ticker].sort_values("date", ascending=False).head(10)
    if len(tnews) > 0:
        for _, r in tnews.iterrows():
            c = "🟢" if r["sentiment"] > 0.1 else ("🔴" if r["sentiment"] < -0.1 else "⚪")
            st.markdown(f"{c} **{r['title']}** — sentiment: `{r['sentiment']:.2f}`")
    else:
        st.info("Aucune news pour ce ticker.")

    st.subheader("Dernieres lignes")
    st.dataframe(sub.tail(20)[["date", "close", "proba", "sent_mean", "rsi"]].iloc[::-1])

with tab2:
    st.header("💬 Assistant Fin Copilot")
    st.caption("Pose une question sur une action suivie (AAPL, MSFT, NVDA, TSLA, AMZN).")

    if not CHATBOT_OK:
        st.error(f"Chatbot indisponible : {CHAT_ERROR}")
        st.info("Verifie que ta cle GROQ_API_KEY est bien dans le fichier .env")
    else:
        ticker_chat = st.selectbox("Contexte action", ["Aucun (vue globale)"] + sorted(df["ticker"].unique()))
        ticker_val = None if ticker_chat.startswith("Aucun") else ticker_chat

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Pose ta question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Reflexion en cours..."):
                    try:
                        answer = ask(prompt, ticker=ticker_val)
                    except Exception as e:
                        answer = f"Erreur : {e}"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()

st.warning("⚠️ Outil pedagogique. Ne constitue pas un conseil en investissement.")