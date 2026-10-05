import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from config import DATA_PROCESSED, DATA_MODELS

FEATURES = [
    "ret_1", "ret_5", "ret_20", "vol_20",
    "ma_ratio", "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "vol_ratio", "range_pct",
    "day_of_week", "month",
    "spy_ret_5", "spy_ret_20", "spy_vol_20",
    "sent_mean", "sent_sum", "sent_count", "sent_ma3",
]

def walk_forward_split(df, n_splits=3):
    df = df.sort_values("date")
    years = sorted(df["date"].dt.year.unique())
    folds = []
    for i in range(n_splits):
        test_year = years[-(n_splits - i)]
        train = df[df["date"].dt.year < test_year]
        test = df[df["date"].dt.year == test_year]
        if len(train) > 200 and len(test) > 30:
            folds.append((train, test))
    return folds

def train(horizon=5):
    df = pd.read_parquet(f"{DATA_PROCESSED}/features.parquet")
    df = df.dropna(subset=[f"target_{horizon}"])
    print(f"[train] {len(df)} lignes, {len(FEATURES)} features")

    folds = walk_forward_split(df)
    print(f"[train] {len(folds)} folds walk-forward")

    results = []
    last_model = None
    for train_df, test_df in folds:
        X_tr = train_df[FEATURES].values
        y_tr = train_df[f"target_{horizon}"].values
        X_te = test_df[FEATURES].values
        y_te = test_df[f"target_{horizon}"].values

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=2,
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        acc = accuracy_score(y_te, (proba > 0.5).astype(int))
        year = int(test_df["date"].dt.year.iloc[0])
        print(f"[train] {year}: AUC={auc:.3f} ACC={acc:.3f} (n={len(test_df)})")
        results.append({"year": year, "auc": auc, "acc": acc})
        last_model = model

    joblib.dump(last_model, f"{DATA_MODELS}/xgb_h{horizon}.joblib")

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": last_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("[train] Top 10 features :")
    print(imp.head(10).to_string(index=False))
    imp.to_csv(f"{DATA_MODELS}/feature_importance.csv", index=False)

    pd.DataFrame(results).to_csv(f"{DATA_MODELS}/metrics_h{horizon}.csv", index=False)
    print(f"[train] Modele sauvegarde -> {DATA_MODELS}/xgb_h{horizon}.joblib")
    return last_model

if __name__ == "__main__":
    train(horizon=5)