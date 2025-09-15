
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

NUMERIC_FEATURES = ["valence","energy","danceability","tempo","loudness","speechiness","acousticness","instrumentalness","liveness"]

def _create_rule_based_mood(row: pd.Series) -> str:
    valence = row.get("valence", np.nan)
    energy = row.get("energy", np.nan)
    dance = row.get("danceability", np.nan)
    if pd.notna(valence) and pd.notna(energy):
        if valence >= 0.6 and energy >= 0.6:
            return "happy"
        if valence < 0.4 and energy < 0.4:
            return "sad"
        if (pd.notna(dance) and dance >= 0.6) or energy >= 0.7:
            return "energetic"
    return "calm"

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates().dropna(how="all")
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    if "mood" not in df.columns:
        df["mood"] = df.apply(_create_rule_based_mood, axis=1)
    df["mood"] = df["mood"].astype("category")
    return df

def train_from_csv(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    df = _prepare(df)
    X = df[[c for c in NUMERIC_FEATURES if c in df.columns]]
    y = df["mood"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        {"n_estimators":[100,200], "max_depth":[None,5,10]},
                        scoring="f1_macro", cv=3, n_jobs=-1)
    grid.fit(Xtr, ytr)
    preds = grid.best_estimator_.predict(Xte)
    print("Best params:", grid.best_params_)
    print(classification_report(yte, preds))
    print(confusion_matrix(yte, preds))
    joblib.dump(grid.best_estimator_, out_path)
    print(f"Saved model to {out_path}")

def evaluate_from_csv(csv_path: str, model_path: str) -> None:
    df = pd.read_csv(csv_path)
    df = _prepare(df)
    X = df[[c for c in NUMERIC_FEATURES if c in df.columns]]
    y = df["mood"]
    model = joblib.load(model_path)
    preds = model.predict(X)
    print(classification_report(y, preds))
    print(confusion_matrix(y, preds))
