import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from features import extract_features

np.random.seed(42)
torch.manual_seed(42)

print("=" * 62)
print("PART 1: what noise does to one song")
print("=" * 62)
path = os.path.join("data", "raw", "genres_original", "blues", "blues.00000.wav")
y, sr = librosa.load(path, duration=30)
power = np.mean(y ** 2)
clean_f = extract_features(y, sr)
print("Song: blues.00000.wav")
print(f"{'':14s}{'brightness (Hz)':>16s}{'zero-crossing rate':>20s}")
print(f"{'clean':14s}{clean_f[0]:16.0f}{clean_f[3]:20.3f}")
for snr in [20, 10, 0]:
    noise = np.random.randn(len(y)) * np.sqrt(power / (10 ** (snr / 10)))
    f = extract_features(y + noise, sr)
    print(f"{str(snr) + ' dB noise':14s}{f[0]:16.0f}{f[3]:20.3f}")

print()
print("=" * 62)
print("PART 2: recommending NEW songs (no listening history) to one user")
print("=" * 62)
inter = pd.read_csv("data/processed/interactions.csv")
fav = pd.read_csv("data/processed/user_favorites.csv")
clean_df = pd.read_csv("data/processed/features_clean.csv")
names = clean_df["filename"].values
genres = clean_df["genre"].values
n_users, n_songs = len(fav), len(clean_df)

R = np.zeros((n_users, n_songs))
R[inter["user_id"].values, inter["song_index"].values] = 1

pos = clean_df.groupby("genre").cumcount().values
cold = (pos % 5 == 0)          # 200 songs we pretend are brand new
warm = ~cold

def load(p):
    df = pd.read_csv(p)
    assert (df["filename"].values == clean_df["filename"].values).all()
    return df.iloc[:, 2:].values

clean_raw = clean_df.iloc[:, 2:].values
white = {s: load(f"data/processed/features_snr{s}.csv") for s in [20, 10, 0]}

scaler = StandardScaler().fit(clean_raw[warm])
Xc = scaler.transform(clean_raw)
Xn = {s: scaler.transform(white[s]) for s in white}

# the noise fix: a small neural network trained on songs that have history
Xtr = torch.tensor(np.vstack([Xn[s][warm] for s in Xn]), dtype=torch.float32)
Ytr = torch.tensor(np.vstack([Xc[warm]] * len(Xn)), dtype=torch.float32)
net = nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 17))
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
for _ in range(500):
    opt.zero_grad()
    nn.functional.mse_loss(net(Xtr), Ytr).backward()
    opt.step()

def fix(X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy()

R_warm, R_cold = R[:, warm], R[:, cold]
profile = (R_warm @ Xc[warm]) / np.maximum(R_warm.sum(axis=1, keepdims=True), 1)
cold_idx = np.where(cold)[0]
users_eval = np.where(R_cold.sum(axis=1) > 0)[0]

versions = {
    "clean audio (ideal)": Xc[cold],
    "noisy audio, 0 dB, no fix": Xn[0][cold],
    "noisy audio, 0 dB, with fix": fix(Xn[0][cold]),
}

u = int(sys.argv[1]) if len(sys.argv) > 1 else int(users_eval[0])
favs = [fav.loc[u, "fav1"], fav.loc[u, "fav2"]]
print(f"User {u} likes: {favs[0]} and {favs[1]}")
print(f"They have listened to {int(R_warm[u].sum())} other songs.")
print(f"The {len(cold_idx)} songs below are new to the system.")
for label, X in versions.items():
    sim = cosine_similarity(profile[u:u + 1], X)[0]
    top = np.argsort(sim)[::-1][:5]
    hits = sum(genres[cold_idx[i]] in favs for i in top)
    print(f"\n{label}: top 5 ({hits} of 5 in the user's favourite genres)")
    for i in top:
        g = genres[cold_idx[i]]
        mark = "yes" if g in favs else "no"
        print(f"   {names[cold_idx[i]]:20s} {g:10s} favourite genre: {mark}")

print()
print("=" * 62)
print("RESULT over all users: Recall@10 on hidden listens of new songs")
print("=" * 62)
for label, X in versions.items():
    sim = cosine_similarity(profile, X)
    out = []
    for v in users_eval:
        top = np.argsort(sim[v])[::-1][:10]
        out.append(R_cold[v, top].sum() / R_cold[v].sum())
    print(f"{label:30s} {np.mean(out):.3f}")
print(f"{'random guessing':30s} 0.050")