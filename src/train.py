"""Train two classifiers, compare, and save the best pipeline + metrics."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL, FEATURES, NUMERIC, TARGET, engineer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ecommerce_returns.csv"
MODEL_PATH = ROOT / "models" / "return_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Model configuration (edit e.g. n_estimators for step 8 and push)
RF_PARAMS = {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced", "random_state": 42}
LR_PARAMS = {"max_iter": 1000, "class_weight": "balanced", "random_state": 42}


def preprocessor(scale: bool) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("scale", StandardScaler()))
    cat = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer([("num", Pipeline(num), NUMERIC), ("cat", Pipeline(cat), CATEGORICAL)])


def main() -> dict:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print("Missing values before cleaning:\n", df.isna().sum()[df.isna().sum() > 0])
    df = engineer(df)
    X = df[FEATURES]
    y = (df[TARGET] == "Returned").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "LogisticRegression": Pipeline([("prep", preprocessor(True)), ("clf", LogisticRegression(**LR_PARAMS))]),
        "RandomForest": Pipeline([("prep", preprocessor(False)), ("clf", RandomForestClassifier(**RF_PARAMS))]),
    }
    results, fitted = {}, {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred, zero_division=0), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
        }
        fitted[name] = pipe
        print(name, results[name])

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print("Best model (by ROC-AUC):", best)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODEL_PATH)
    out = {"best_model": best, "selection_metric": "roc_auc", "results": results}
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
