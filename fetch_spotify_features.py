
"""Fetch Spotify audio features to CSV.

Usage:
    export SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=...
    python scripts/fetch_spotify_features.py --playlist https://open.spotify.com/playlist/<id> --out data/music/songs.csv

You can also pass --tracks track_id1 track_id2 ... or --artist <artist_id>.

Notes:
- Requires 'spotipy' (`pip install spotipy`).
- Writes columns: id,title,artist,tempo,energy,valence,danceability,loudness,speechiness,acousticness,instrumentalness,liveness
"""
import os
import argparse
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

AUDIO_COLS = [
    "id","title","artist","tempo","energy","valence","danceability","loudness",
    "speechiness","acousticness","instrumentalness","liveness"
]

def _sp():
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    sec = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not sec:
        raise SystemExit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET env vars.")
    auth = SpotifyClientCredentials(client_id=cid, client_secret=sec)
    return spotipy.Spotify(auth_manager=auth)

def _audio_features(sp, track_ids):
    feats = sp.audio_features(tracks=track_ids)
    meta = sp.tracks(track_ids)["tracks"]
    rows = []
    for f, m in zip(feats, meta):
        if f is None or m is None: 
            continue
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
    return pd.DataFrame(rows, columns=AUDIO_COLS)

def from_playlist(sp, playlist_url):
    # Parse playlist ID
    pl_id = playlist_url.rstrip("/").split("/")[-1].split("?")[0]
    items = []
    results = sp.playlist_tracks(pl_id, fields="items(track(id)) , next", limit=100)
    items.extend(results["items"])
    while results.get("next"):
        results = sp.next(results)
        items.extend(results["items"])
    track_ids = [it["track"]["id"] for it in items if it.get("track") and it["track"].get("id")]
    # Chunk in 100s
    dfs = []
    for i in range(0, len(track_ids), 100):
        dfs.append(_audio_features(sp, track_ids[i:i+100]))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=AUDIO_COLS)

def from_artist(sp, artist_id, limit=50):
    results = sp.artist_top_tracks(artist_id)
    track_ids = [t["id"] for t in results["tracks"][:limit]]
    return _audio_features(sp, track_ids)

def from_tracks(sp, track_ids):
    return _audio_features(sp, track_ids)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--playlist", help="Spotify playlist URL or ID")
    g.add_argument("--artist", help="Spotify artist ID")
    g.add_argument("--tracks", nargs="+", help="List of track IDs")
    ap.add_argument("--out", required=True, help="Output CSV path (e.g., data/music/songs.csv)")
    args = ap.parse_args()

    sp = _sp()
    if args.playlist:
        df = from_playlist(sp, args.playlist)
    elif args.artist:
        df = from_artist(sp, args.artist)
    else:
        df = from_tracks(sp, args.tracks)

    if df.empty:
        raise SystemExit("No tracks found or features unavailable.")
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
