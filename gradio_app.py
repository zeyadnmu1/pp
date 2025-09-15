
import gradio as gr
import pandas as pd
import joblib
from pathlib import Path

MUSIC_MODEL = Path("../models/music/model.joblib")
YT_MODEL = Path("../models/youtube/model.joblib")

music_model = joblib.load(MUSIC_MODEL) if MUSIC_MODEL.exists() else None
yt_model = joblib.load(YT_MODEL) if YT_MODEL.exists() else None

def predict_mood(valence, energy, danceability, tempo, loudness, speechiness=0.08, acousticness=0.2, instrumentalness=0.1, liveness=0.2):
    if music_model is None:
        # fallback simple rules
        if valence >= 0.6 and energy >= 0.6: return "happy"
        if valence < 0.4 and energy < 0.4: return "sad"
        if (danceability >= 0.6) or energy >= 0.7: return "energetic"
        return "calm"
    X = [[valence, energy, danceability, tempo, loudness, speechiness, acousticness, instrumentalness, liveness]]
    return music_model.predict(X)[0]

def predict_sentiment(text):
    if yt_model is None:
        t = text.lower()
        pos = sum(t.count(w) for w in ["amazing","love","great","helpful","thanks"])
        neg = sum(t.count(w) for w in ["terrible","dislike","waste","confusing","bad"])
        if pos>neg: return "positive"
        if neg>pos: return "negative"
        return "neutral"
    return yt_model.predict([text])[0]

music_inputs = [
    gr.Slider(0,1,0.6,label="valence"), gr.Slider(0,1,0.6,label="energy"),
    gr.Slider(0,1,0.6,label="danceability"), gr.Slider(40,240,120,label="tempo"),
    gr.Slider(-60,0,-7,label="loudness")
]

with gr.Blocks(title="ML Certificate Projects") as demo:
    gr.Markdown("## 🎵 Music Mood & 💬 YouTube Sentiment")
    with gr.Tab("Music Mood"):
        out1 = gr.Textbox(label="Predicted mood")
        gr.Interface(fn=predict_mood, inputs=music_inputs, outputs=out1, submit_btn="Predict").render()
    with gr.Tab("YouTube Sentiment"):
        txt = gr.Textbox(label="Enter a comment")
        out2 = gr.Textbox(label="Predicted sentiment")
        gr.Interface(fn=predict_sentiment, inputs=txt, outputs=out2, submit_btn="Predict").render()

if __name__ == "__main__":
    demo.launch()
