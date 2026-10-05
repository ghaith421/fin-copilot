# 💹 Fin Copilot — Assistant IA pour l'analyse financière

> Outil pédagogique combinant Machine Learning, NLP et LLM (RAG) pour analyser 5 actions tech.

⚠️ **Projet pédagogique.** Ne constitue **pas** un conseil en investissement.

---

## 📖 Description

**Fin Copilot** est un projet de data science appliquée à la finance qui permet :

- 📊 **Suivre 5 actions tech** : AAPL, MSFT, NVDA, TSLA, AMZN
- 🧠 **Prédire la direction** du prix à J+5 avec XGBoost
- 📰 **Analyser le sentiment** des news financières (VADER)
- 📈 **Visualiser** les tendances via un dashboard interactif (Streamlit)
- 💬 **Poser des questions** à un assistant IA (RAG + Groq/Llama)
- 🔍 **Analyser l'impact** d'une news via des news similaires passées

---

## 🎯 Fonctionnalités

### 📊 Dashboard interactif
- Prix historiques (2018 → aujourd'hui)
- Probabilité de hausse à J+5
- Indicateurs techniques (RSI, MACD, Bollinger)
- News récentes avec score de sentiment

### 💬 Chatbot RAG
Un assistant qui répond en français :
- « Pourquoi AAPL a évolué récemment ? »
- « Compare NVDA et MSFT. »
- « Quel est l'impact d'une news sur NVDA ? »

### 🔬 Machine Learning
- **Modèle** : XGBoost classifier
- **Validation** : walk-forward
- **21 features** : techniques, sentiment, macro, calendaires

### 📈 Event Study
Pour chaque news, calcul de l'impact réel à J+1 et J+5.
Le chatbot cherche les news passées similaires via embeddings.

---

## 🏗️ Architecture
