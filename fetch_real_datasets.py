
"""Download & prepare real datasets (IMDB polarity, UCI YouTube Spam) into data/ folder.

Run:
    python scripts/fetch_real_datasets.py
"""
import os, io, zipfile, tarfile, csv, re, sys, shutil
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True, parents=True)

def download(url: str) -> bytes:
    print(f"Downloading: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def prepare_imdb_polarity():
    # Cornell polarity v1.0 (700 pos / 700 neg) — small but fully public
    # Source: https://www.cs.cornell.edu/people/pabo/movie-review-data/  (polarity dataset v1.0)
    url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
    raw = download(url)
    tar = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    pos_texts, neg_texts = [], []
    for member in tar.getmembers():
        if member.isfile() and member.name.startswith("txt_sentoken/pos/"):
            f = tar.extractfile(member)
            pos_texts.append(f.read().decode("latin-1", errors="ignore"))
        if member.isfile() and member.name.startswith("txt_sentoken/neg/"):
            f = tar.extractfile(member)
            neg_texts.append(f.read().decode("latin-1", errors="ignore"))
    df = pd.DataFrame({
        "text": pos_texts + neg_texts,
        "label": ["positive"]*len(pos_texts) + ["negative"]*len(neg_texts)
    })
    out = DATA / "youtube" / "imdb_reviews_as_sentiment.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {out} ({len(df)} rows)")

def prepare_youtube_spam():
    # UCI YouTube Spam Collection: 5 CSV files with columns CLASS, CONTENT
    # Dataset page: https://archive.ics.uci.edu/ml/datasets/youtube+spam+collection
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/00380/"
    files = ["Youtube01-Psy.csv","Youtube02-KatyPerry.csv","Youtube03-LMFAO.csv","Youtube04-Eminem.csv","Youtube05-Shakira.csv"]
    frames = []
    for f in files:
        raw = download(base + f)
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        # Map spam/ham to negative/neutral for demo (ham ~ neutral)
        df = df.rename(columns={"CONTENT":"text","CLASS":"spam"})
        df["label"] = df["spam"].map({1:"negative", 0:"neutral"})
        frames.append(df[["text","label"]])
    out = DATA / "youtube" / "youtube_spam_as_sentiment.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    try:
        prepare_imdb_polarity()
    except Exception as e:
        print("IMDB polarity download failed:", e)
    try:
        prepare_youtube_spam()
    except Exception as e:
        print("YouTube spam download failed:", e)
    print("Done.")
