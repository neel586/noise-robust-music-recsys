import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

clean = pd.read_csv("data/processed/features_clean.csv")
scaler = StandardScaler().fit(clean.iloc[:, 2:].values)
Xc = scaler.transform(clean.iloc[:, 2:].values)
genres = clean["genre"].values
n = len(clean)

def evaluate(Xq):
    sim = cosine_similarity(Xq, Xc)
    sim_ex = sim.copy()
    np.fill_diagonal(sim_ex, -1)
    genre_scores = []
    for i in range(n):
        top5 = np.argsort(sim_ex[i])[::-1][:5]
        genre_scores.append(np.mean(genres[top5] == genres[i]))
    self_rate = np.mean(sim.argmax(axis=1) == np.arange(n))
    return np.mean(genre_scores), self_rate

def load(path):
    df = pd.read_csv(path)
    assert (df["filename"].values == clean["filename"].values).all()
    return scaler.transform(df.iloc[:, 2:].values)

print(f"{'condition':10s} {'noisy genre':>12s} {'denoised genre':>15s} {'noisy self':>11s} {'denoised self':>14s}")
for snr in [20, 10, 0]:
    gn, sn = evaluate(load(f"data/processed/features_snr{snr}.csv"))
    gd, sd = evaluate(load(f"data/processed/features_denoised_snr{snr}.csv"))
    print(f"{str(snr) + ' dB':10s} {gn:12.3f} {gd:15.3f} {sn:11.3f} {sd:14.3f}")