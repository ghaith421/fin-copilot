import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from config import DATA_PROCESSED, DATA_RAW, DATA_MODELS

load_dotenv()

MODEL = "openai/gpt-oss-20b"

FEATURES = [
    "ret_1", "ret_5", "ret_20", "vol_20",
    "ma_ratio", "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "vol_ratio", "range_pct",
    "day_of_week", "month",
    "spy_ret_5", "spy_ret_20", "spy_vol_20",
    "sent_mean", "sent_sum", "sent_count", "sent_ma3",
]


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_context(ticker=None):
    """Contexte general : derniers prix + news recentes."""
    df = pd.read_parquet(f"{DATA_PROCESSED}/features.parquet")
    news = pd.read_parquet(f"{DATA_RAW}/news_scored.parquet")

    context = []

    if ticker:
        sub = df[df["ticker"] == ticker].sort_values("date").tail(30)
        last = sub.iloc[-1] if len(sub) > 0 else None
        if last is not None:
            context.append(f"Derniere donnee pour {ticker} :")
            context.append(f"  - Prix : {last['close']:.2f} USD")
            context.append(f"  - RSI : {last['rsi']:.1f}")
            context.append(f"  - Rendement 5j : {last['ret_5']*100:.2f}%")
            context.append(f"  - Rendement 20j : {last['ret_20']*100:.2f}%")
            context.append(f"  - Volatilite 20j : {last['vol_20']*100:.2f}%")
            context.append(f"  - Ratio MA10/MA50 : {last['ma_ratio']:.3f}")
            context.append(f"  - MACD : {last['macd']:.3f}")

        tnews = news[news["tickers"] == ticker].sort_values("date", ascending=False).head(10)
        if len(tnews) > 0:
            context.append(f"\nDernieres news pour {ticker} :")
            for _, r in tnews.iterrows():
                context.append(f"  - [{r['sentiment']:+.2f}] {r['title']}")
    else:
        context.append("Synthese de toutes les actions suivies :")
        for t in df["ticker"].unique():
            sub = df[df["ticker"] == t].sort_values("date")
            last = sub.iloc[-1]
            context.append(
                f"  - {t} : prix {last['close']:.2f}, RSI {last['rsi']:.1f}, "
                f"ret20 {last['ret_20']*100:+.2f}%"
            )

    return "\n".join(context)


def build_impact_context(question, ticker=None, k=10):
    """Cherche les news passees similaires a la question et calcule leur impact."""
    try:
        from src.analysis.similarity import analyze_news_impact, format_report
        report = analyze_news_impact(question, ticker=ticker, k=k)
        return format_report(report)
    except Exception as e:
        return f"(Analyse d'impact indisponible : {e})"


def ask(question, ticker=None, history=None, use_impact=True):
    """Pose une question au chatbot avec le contexte du projet."""
    client = get_client()
    context = build_context(ticker)

    impact_context = ""
    if use_impact:
        impact_context = build_impact_context(question, ticker=ticker, k=10)

    system_prompt = """Tu es un assistant financier pedagogique pour un projet d'analyse d'actions.
Tu reponds en francais, de facon claire et concise.
Tu utilises UNIQUEMENT les donnees fournies dans le contexte ci-dessous.

Quand une ANALYSE D'IMPACT est fournie, tu DOIS :
- Citer le nombre de news similaires trouvees
- Donner le rendement moyen a J+1 et J+5
- Donner le taux de hausse (hit rate)
- Mentionner les 2-3 news similaires les plus pertinentes avec leur impact reel
- Rester factuel : si les donnees sont insuffisantes, le dire honnetement

Si une info n'est pas dans le contexte, dis-le honnetement.
Tu ne donnes JAMAIS de conseil d'investissement personnalise.
Tu termines toujours par un rappel que ceci est un outil pedagogique."""

    user_message = f"""Contexte general :
{context}

{impact_context}

Question : {question}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    if history:
        messages = [messages[0]] + history + [messages[-1]]

    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=700,
    )
    return r.choices[0].message.content


if __name__ == "__main__":
    print("=== Test 1 : question simple ===")
    print(ask("Bonjour, que peux-tu faire ?", ticker=None, use_impact=False))

    print("\n\n=== Test 2 : analyse d'impact ===")
    print(ask(
        "Nvidia annonce une nouvelle puce IA. Quel est l'impact attendu sur le cours ?",
        ticker="NVDA",
        use_impact=True,
    ))