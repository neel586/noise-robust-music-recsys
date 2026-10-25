import os
import numpy as np
import pandas as pd
import librosa
from features import extract_features

np.random.seed(42)

DATA_DIR = os.path.join("data", "raw", "genres_original")
PER_GENRE = 100          # must match build_features.py
SNR_LEVELS = [20, 10, 0]

names = ["centroid", "bandwidth", "rolloff", "zcr"] + [f"mfcc{i+1}" for i in range(13)]
rows = {snr: [] for snr in SNR_LEVELS}

for genre in sorted(os.listdir(DATA_DIR)):
    genre_dir = os.path.join(DATA_DIR, genre)
    files = sorted(os.listdir(genre_dir))[:PER_GENRE]
    for f in files:
        path = os.path.join(genre_dir, f)
        try:
            y, sr = librosa.load(path, duration=30)
        except Exception as e:
            print("skipped", f, "-", e)
            continue
        signal_power = np.mean(y ** 2)
        for snr in SNR_LEVELS:
            noise_power = signal_power / (10 ** (snr / 10))
            y_noisy = y + np.random.randn(len(y)) * np.sqrt(noise_power)
            feats = extract_features(y_noisy, sr)
            rows[snr].append([f, genre] + list(feats))
    print("done:", genre)

for snr in SNR_LEVELS:
    df = pd.DataFrame(rows[snr], columns=["filename", "genre"] + names)
    out = os.path.join("data", "processed", f"features_snr{snr}.csv")
    df.to_csv(out, index=False)
    print("saved", len(df), "songs to", out)