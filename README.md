# ML Certificate Project — Professional Repo

Two certificate projects wrapped in a production-style repository:

1. **Music Mood Classifier** — predicts `happy/sad/energetic/calm` from audio features.
2. **YouTube Sentiment Analyzer** — classifies comments into `positive/negative/neutral`.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
### Music
```bash
ml-mood-train --csv data/music/songs.csv --out models/music/model.joblib
ml-mood-eval  --csv data/music/songs.csv --model models/music/model.joblib
```
### YouTube
```bash
ml-sent-train --csv data/youtube/comments.csv --out models/youtube/model.joblib
ml-sent-eval  --csv data/youtube/comments.csv --model models/youtube/model.joblib
```
MIT © 2025 Omar Eid


## Web App (Streamlit)

```bash
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```


---

## 🚀 Deployment

### Streamlit Cloud
1. Push this repo to GitHub.
2. Go to streamlit.io → Deploy app → point to `app/streamlit_app.py`.
3. Set Python version to 3.11+ and add file `requirements-app.txt` as packages file.

### Hugging Face Spaces (Streamlit)
1. Create a Space → type = Streamlit.
2. Upload the repo; make sure `app/streamlit_app.py` exists.
3. In Space settings, set `app file` to `app/streamlit_app.py` and add `requirements-app.txt`.
4. Optionally add secrets for APIs if you fetch real data.

### Fetch real datasets
```bash
pip install requests pandas
python scripts/fetch_real_datasets.py
# Outputs:
# data/youtube/imdb_reviews_as_sentiment.csv  (pos/neg)
# data/youtube/youtube_spam_as_sentiment.csv  (neutral/negative mapped)
```
Then point the app (or CLI) to those CSVs.


### New features
- ✅ KPI cards (Accuracy, Macro‑F1)
- ✅ Inference-only mode (upload a `.joblib` and get predictions without training)
- ✅ Downloadable evaluation report (.txt) and predictions (CSV)
- ✅ Gradio demo (`app/gradio_app.py`)

**Run Gradio:**
```bash
pip install gradio
python app/gradio_app.py
```


### Spotify integration
- Install: `pip install -r requirements-spotify.txt`
- Use script:
  ```bash
  export SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=...
  python scripts/fetch_spotify_features.py --playlist https://open.spotify.com/playlist/<id> --out data/music/songs.csv
  ```
- In the Streamlit app, open **Music → Import from Spotify** and paste your **client id/secret** and playlist URL.
  - For deployments, set secrets in **Streamlit Cloud** (Settings → Secrets).


### Bulk fetch Spotify playlists (one command)
```bash
pip install -r requirements-spotify.txt
export SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=...
python scripts/fetch_from_list.py --list scripts/playlists.txt --out data/music/songs_spotify_all.csv
```
Then train/evaluate with:
```bash
ml-mood-train --csv data/music/songs_spotify_all.csv --out models/music/spotify_model.joblib
ml-mood-eval  --csv data/music/songs_spotify_all.csv --model models/music/spotify_model.joblib
```
