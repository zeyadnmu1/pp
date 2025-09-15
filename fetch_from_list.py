
"""Fetch Spotify audio features from a list of playlist URLs and combine into one CSV.

Usage:
    export SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=...
    python scripts/fetch_from_list.py --list scripts/playlists.txt --out data/music/songs_spotify_all.csv
"""
import os, argparse, pandas as pd
from pathlib import Path
from subprocess import run, CalledProcessError

ROOT = Path(__file__).resolve().parents[1]
FETCH = ROOT / "scripts" / "fetch_spotify_features.py"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="Path to a text file with one Spotify playlist URL per line")
    ap.add_argument("--out", required=True, help="Combined CSV output path")
    args = ap.parse_args()

    lst = Path(args.list).read_text(encoding="utf-8").strip().splitlines()
    tmp_dir = ROOT / "data" / "music"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, url in enumerate([l for l in lst if l.strip()], start=1):
        out_i = tmp_dir / f"pl_{i}.csv"
        print(f"[{i}/{len(lst)}] Fetching: {url}")
        cmd = ["python", str(FETCH), "--playlist", url.strip(), "--out", str(out_i)]
        try:
            run(cmd, check=True)
            paths.append(out_i)
        except CalledProcessError as e:
            print("Failed:", e)

    if not paths:
        raise SystemExit("No CSVs fetched. Check your credentials/URLs.")

    # Combine & dedupe by id if present
    dfs = [pd.read_csv(p) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset=["id"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"Saved {args.out} with shape {combined.shape}")

if __name__ == "__main__":
    main()
