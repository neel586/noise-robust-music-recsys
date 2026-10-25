import os
import numpy as np
import pandas as pd
import librosa
from features import extract_features

DATA_DIR = os.path.join("data", "raw", "genres_original")
OUT_FILE = os.path.join("data", "processed", "features_clean.csv")
PER_GENRE = 100   # test with 10 clips per genre; later change to 100 for everything

os.makedirs(os.path.join("data", "processed"), exist_ok=True)

names = ["centroid", "bandwidth", "rolloff", "zcr"] + [f"mfcc{i+1}" for i in range(13)]
rows = []

for genre in sorted(os.listdir(DATA_DIR)):
    genre_dir = os.path.join(DATA_DIR, genre)
    files = sorted(os.listdir(genre_dir))[:PER_GENRE]
    for f in files:
        path = os.path.join(genre_dir, f)
        try:
            y, sr = librosa.load(path, duration=30)
            feats = extract_features(y, sr)
            rows.append([f, genre] + list(feats))
        except Exception as e:
            print("skipped", f, "-", e)
    print("done:", genre)

df = pd.DataFrame(rows, columns=["filename", "genre"] + names)
df.to_csv(OUT_FILE, index=False)
print("saved", len(df), "songs to", OUT_FILE)