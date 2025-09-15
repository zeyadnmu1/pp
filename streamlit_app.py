
import io, os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

st.set_page_config(page_title="ML Certificate Projects", page_icon="🎓", layout="wide")

# ---------- Header ----------
st.markdown(
    """
    <div style="padding: 18px; border-radius: 16px; border:1px solid #1f2937; background: linear-gradient(135deg,#0b0f14,#0c1420);">
      <h1 style="margin:0; font-size: 28px;">🎓 ML Certificate Projects</h1>
      <p style="margin:6px 0 0 0; color:#9ca3af;">Interactive demos for <b>Music Mood</b> & <b>YouTube Sentiment</b> — upload data, train models, evaluate, and download.</p>
    </div>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("Mode", ["Train & Evaluate", "Inference-only"], index=0, help="Run full training or just load a saved model to predict.")
    ts = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, help="Used only in training mode.")
    st.caption("Tip: Use 0.2 for a balanced evaluation.")

tab1, tab2 = st.tabs(["🎵 Music Mood Classifier", "💬 YouTube Sentiment Analyzer"])

# Helper to show KPI cards and downloadable report
def show_eval_and_download(y_true, y_pred, labels, section_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Macro‑F1", f"{f1:.3f}")
    st.text("Classification Report:")
    report_txt = classification_report(y_true, y_pred)
    st.code(report_txt, language="text")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    st.write(pd.DataFrame(cm, index=labels, columns=labels))

    # Downloadables
    rep_bytes = io.BytesIO(report_txt.encode("utf-8"))
    st.download_button(f"⬇️ Download {section_name} report (.txt)", data=rep_bytes.getvalue(),
                       file_name=f"{section_name.lower().replace(' ','_')}_report.txt")

# ----------------------- Music Mood -----------------------
with tab1:
    st.subheader("Data")
    col_up1, col_up2 = st.columns([2,1])
    with col_up1:
        up = st.file_uploader("Upload CSV (valence, energy, danceability, tempo, loudness, ...)", type=["csv"], key="music_up")
    with col_up2:
        use_sample = st.toggle("Use bundled sample", value=True)

    if use_sample:
        df_music = pd.read_csv("../data/music/songs.csv")
    else:
        df_music = pd.read_csv(up) if up is not None else None

    # Inference‑only: upload model
    if mode == "Inference-only":
        model_file = st.file_uploader("Upload trained model (.joblib)", type=["joblib"], key="music_model")
        if df_music is not None and model_file is not None:
            st.success("Model and data loaded.")
            # Ensure features exist
            features = ['valence','energy','danceability','tempo','loudness','speechiness','acousticness','instrumentalness','liveness']
            present = [c for c in features if c in df_music.columns]
            if not present:
                st.error("No numeric audio features found in your CSV.")
            else:
                X = df_music[present].copy()
                for c in present:
                    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
                model = joblib.load(io.BytesIO(model_file.read()))
                preds = model.predict(X)
                out = df_music.copy()
                out["pred_mood"] = preds
                st.dataframe(out.head(), use_container_width=True)
                # Download predictions
                buf = io.BytesIO()
                out.to_csv(buf, index=False)
                st.download_button("⬇️ Download predictions (CSV)", data=buf.getvalue(), file_name="music_predictions.csv")
        st.stop()

    if df_music is not None:
        st.dataframe(df_music.head(), use_container_width=True)
        if "mood" in df_music.columns:
            st.caption("Class distribution")
            fig1, ax1 = plt.subplots()
            df_music["mood"].value_counts().plot(kind="bar", ax=ax1)
            ax1.set_title("Mood Class Distribution"); ax1.set_xlabel("mood"); ax1.set_ylabel("count")
            st.pyplot(fig1)

        
with st.expander("🎧 Import from Spotify (optional)"):
    st.caption("Provide credentials via **st.secrets** or input boxes below. Requires internet access.")
    cid = st.text_input("SPOTIFY_CLIENT_ID", type="password", value=st.secrets.get("SPOTIFY_CLIENT_ID","") if hasattr(st, "secrets") else "")
    csec = st.text_input("SPOTIFY_CLIENT_SECRET", type="password", value=st.secrets.get("SPOTIFY_CLIENT_SECRET","") if hasattr(st, "secrets") else "")
    pl = st.text_input("Playlist URL (e.g., https://open.spotify.com/playlist/...)")
    if st.button("Fetch playlist features"):
        if not cid or not csec or not pl:
            st.error("Missing client id/secret or playlist URL.")
        else:
            try:
                import spotipy
                from spotipy.oauth2 import SpotifyClientCredentials
                auth = SpotifyClientCredentials(client_id=cid, client_secret=csec)
                sp = spotipy.Spotify(auth_manager=auth)
                # minimal fetch inline (first 100 tracks)
                pl_id = pl.rstrip("/").split("/")[-1].split("?")[0]
                results = sp.playlist_tracks(pl_id, fields="items(track(id)) , next", limit=100)
                ids = [it["track"]["id"] for it in results["items"] if it.get("track")]
                feats = sp.audio_features(ids)
                tracks = sp.tracks(ids)["tracks"]
                rows = []
                for f, m in zip(feats, tracks):
                    if f and m:
                        rows.append({
                            "id": m["id"],
                            "title": m["name"],
                            "artist": ", ".join(a["name"] for a in m["artists"]),
                            "tempo": f["tempo"],
                            "energy": f["energy"],
                            "valence": f["valence"],
                            "danceability": f["danceability"],
                            "loudness": f["loudness"],
                            "speechiness": f["speechiness"],
                            "acousticness": f["acousticness"],
                            "instrumentalness": f["instrumentalness"],
                            "liveness": f["liveness"],
                        })
                spdf = pd.DataFrame(rows)
                if not spdf.empty:
                    st.success(f"Fetched {len(spdf)} tracks.")
                    st.dataframe(spdf.head(), use_container_width=True)
                    # Merge with existing df
                    for col in spdf.columns:
                        if col not in df.columns:
                            df[col] = np.nan
                    df = pd.concat([df, spdf[df.columns]], ignore_index=True).drop_duplicates(subset=["id"] if "id" in df.columns else None)
                else:
                    st.warning("No tracks fetched.")
            except Exception as e:
                st.error(f"Spotify fetch failed: {e}")

        st.subheader("Train")
        features = ['valence','energy','danceability','tempo','loudness','speechiness','acousticness','instrumentalness','liveness']
        present = [c for c in features if c in df_music.columns]
        df = df_music.drop_duplicates().dropna(how="all").copy()
        for c in present:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

        if "mood" not in df.columns:
            def map_mood(row):
                v = row.get("valence", np.nan); e = row.get("energy", np.nan); d = row.get("danceability", np.nan)
                if pd.notna(v) and pd.notna(e):
                    if v >= 0.6 and e >= 0.6: return "happy"
                    if v < 0.4 and e < 0.4:   return "sad"
                    if (pd.notna(d) and d >= 0.6) or e >= 0.7: return "energetic"
                return "calm"
            df["mood"] = df.apply(map_mood, axis=1)

        X = df[present]
        y = df["mood"].astype("category")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, stratify=y, random_state=42)

        c1, c2 = st.columns([1,1])
        if c1.button("🚀 Train RandomForest (GridSearch)"):
            with st.spinner("Training..."):
                grid = GridSearchCV(RandomForestClassifier(random_state=42),
                                    {"n_estimators":[100,200], "max_depth":[None,5,10]},
                                    scoring="f1_macro", cv=3, n_jobs=-1)
                grid.fit(Xtr, ytr)
                pred = grid.best_estimator_.predict(Xte)
                c1.success("Done")
                st.code(f"Best params: {grid.best_params_}", language="text")

                show_eval_and_download(yte, pred, labels=y.cat.categories, section_name="Music Mood")

                # Download trained model
                bytes_io = io.BytesIO()
                joblib.dump(grid.best_estimator_, bytes_io)
                c2.download_button("⬇️ Download model (.joblib)", data=bytes_io.getvalue(), file_name="music_model.joblib")

        st.subheader("Try a Single Prediction")
        cols = st.columns(5)
        v = cols[0].number_input("valence (0–1)", 0.0, 1.0, 0.6, 0.01)
        e = cols[1].number_input("energy (0–1)", 0.0, 1.0, 0.6, 0.01)
        d = cols[2].number_input("danceability (0–1)", 0.0, 1.0, 0.6, 0.01)
        t = cols[3].number_input("tempo (bpm)", 40.0, 240.0, 120.0, 1.0)
        l = cols[4].number_input("loudness (dB)", -60.0, 0.0, -7.0, 0.1)
        c3 = st.columns(3)
        sp = c3[0].number_input("speechiness (0–1)", 0.0, 1.0, 0.08, 0.01)
        ac = c3[1].number_input("acousticness (0–1)", 0.0, 1.0, 0.2, 0.01)
        ins = c3[2].number_input("instrumentalness (0–1)", 0.0, 1.0, 0.1, 0.01)
        lv = st.number_input("liveness (0–1)", 0.0, 1.0, 0.2, 0.01, key="live")
        if st.button("Predict mood"):
            mood = "calm"
            if v >= 0.6 and e >= 0.6: mood = "happy"
            elif v < 0.4 and e < 0.4: mood = "sad"
            elif (d >= 0.6) or e >= 0.7: mood = "energetic"
            st.success(f"Predicted mood: {mood}")

# ----------------------- YouTube Sentiment -----------------------
with tab2:
    st.subheader("Data")
    col_up1, col_up2 = st.columns([2,1])
    with col_up1:
        up2 = st.file_uploader("Upload CSV (text,label)", type=["csv"], key="yt_up")
    with col_up2:
        use_sample2 = st.toggle("Use bundled sample", value=True, key="yt_sample")

    if use_sample2:
        df_yt = pd.read_csv("../data/youtube/comments.csv")
    else:
        df_yt = pd.read_csv(up2) if up2 is not None else None

    # Inference‑only: upload model
    if mode == "Inference-only":
        model_file = st.file_uploader("Upload trained sentiment model (.joblib)", type=["joblib"], key="yt_model")
        if df_yt is not None and model_file is not None:
            st.success("Model and data loaded.")
            X = df_yt["text"].astype(str)
            model = joblib.load(io.BytesIO(model_file.read()))
            preds = model.predict(X)
            out = df_yt.copy()
            out["pred_label"] = preds
            st.dataframe(out.head(), use_container_width=True)
            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            st.download_button("⬇️ Download predictions (CSV)", data=buf.getvalue(), file_name="youtube_predictions.csv")
        st.stop()

    if df_yt is not None:
        st.dataframe(df_yt.head(), use_container_width=True)
        st.caption("Class distribution")
        fig2, ax2 = plt.subplots()
        df_yt["label"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_title("Sentiment Class Distribution"); ax2.set_xlabel("label"); ax2.set_ylabel("count")
        st.pyplot(fig2)

        
with st.expander("🎧 Import from Spotify (optional)"):
    st.caption("Provide credentials via **st.secrets** or input boxes below. Requires internet access.")
    cid = st.text_input("SPOTIFY_CLIENT_ID", type="password", value=st.secrets.get("SPOTIFY_CLIENT_ID","") if hasattr(st, "secrets") else "")
    csec = st.text_input("SPOTIFY_CLIENT_SECRET", type="password", value=st.secrets.get("SPOTIFY_CLIENT_SECRET","") if hasattr(st, "secrets") else "")
    pl = st.text_input("Playlist URL (e.g., https://open.spotify.com/playlist/...)")
    if st.button("Fetch playlist features"):
        if not cid or not csec or not pl:
            st.error("Missing client id/secret or playlist URL.")
        else:
            try:
                import spotipy
                from spotipy.oauth2 import SpotifyClientCredentials
                auth = SpotifyClientCredentials(client_id=cid, client_secret=csec)
                sp = spotipy.Spotify(auth_manager=auth)
                # minimal fetch inline (first 100 tracks)
                pl_id = pl.rstrip("/").split("/")[-1].split("?")[0]
                results = sp.playlist_tracks(pl_id, fields="items(track(id)) , next", limit=100)
                ids = [it["track"]["id"] for it in results["items"] if it.get("track")]
                feats = sp.audio_features(ids)
                tracks = sp.tracks(ids)["tracks"]
                rows = []
                for f, m in zip(feats, tracks):
                    if f and m:
                        rows.append({
                            "id": m["id"],
                            "title": m["name"],
                            "artist": ", ".join(a["name"] for a in m["artists"]),
                            "tempo": f["tempo"],
                            "energy": f["energy"],
                            "valence": f["valence"],
                            "danceability": f["danceability"],
                            "loudness": f["loudness"],
                            "speechiness": f["speechiness"],
                            "acousticness": f["acousticness"],
                            "instrumentalness": f["instrumentalness"],
                            "liveness": f["liveness"],
                        })
                spdf = pd.DataFrame(rows)
                if not spdf.empty:
                    st.success(f"Fetched {len(spdf)} tracks.")
                    st.dataframe(spdf.head(), use_container_width=True)
                    # Merge with existing df
                    for col in spdf.columns:
                        if col not in df.columns:
                            df[col] = np.nan
                    df = pd.concat([df, spdf[df.columns]], ignore_index=True).drop_duplicates(subset=["id"] if "id" in df.columns else None)
                else:
                    st.warning("No tracks fetched.")
            except Exception as e:
                st.error(f"Spotify fetch failed: {e}")

        st.subheader("Train")
        X = df_yt["text"].astype(str)
        y = df_yt["label"].astype("category")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, stratify=y, random_state=42)

        c1, c2 = st.columns([1,1])
        if c1.button("🚀 Train LinearSVC (GridSearch)"):
            with st.spinner("Training..."):
                pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2))),
                                 ("clf", LinearSVC())])
                grid = GridSearchCV(pipe,
                                    {"tfidf__max_df":[0.9,1.0], "tfidf__min_df":[1,3], "clf__C":[0.5,1.0,2.0]},
                                    scoring="f1_macro", cv=3, n_jobs=-1)
                grid.fit(Xtr, ytr)
                pred = grid.best_estimator_.predict(Xte)
                c1.success("Done")
                st.code(f"Best params: {grid.best_params_}", language="text")

                show_eval_and_download(yte, pred, labels=y.cat.categories, section_name="YouTube Sentiment")

                bytes_io = io.BytesIO()
                joblib.dump(grid.best_estimator_, bytes_io)
                c2.download_button("⬇️ Download model (.joblib)", data=bytes_io.getvalue(), file_name="youtube_sentiment_model.joblib")

        st.subheader("Try a Single Prediction")
        user_txt = st.text_input("Enter a comment...")
        if st.button("Predict sentiment"):
            t = user_txt.lower()
            pos = sum(t.count(w) for w in ["amazing", "love", "great", "helpful", "thanks"])
            neg = sum(t.count(w) for w in ["terrible", "dislike", "waste", "confusing", "bad"])
            if pos>neg: st.success("Predicted: positive")
            elif neg>pos: st.error("Predicted: negative")
            else: st.info("Predicted: neutral")

st.caption("Built with Streamlit • Models saved as .joblib • KPIs, inference-only, and reports added ✨")
