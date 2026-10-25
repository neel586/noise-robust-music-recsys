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

    # Score 1: same-genre rate in top 5 (the song's own clean copy excluded)
    sim_ex = sim.copy()
    np.fill_diagonal(sim_ex, -1)
    genre_scores = []
    for i in range(n):
        top5 = np.argsort(sim_ex[i])[::-1][:5]
        genre_scores.append(np.mean(genres[top5] == genres[i]))

    # Score 2: is the song's own clean version the #1 match?
    self_rate = np.mean(sim.argmax(axis=1) == np.arange(n))
    return np.mean(genre_scores), self_rate

print(f"{'condition':12s} {'same-genre@5':>14s} {'self-match':>12s}")
g, s = evaluate(Xc)
print(f"{'clean':12s} {g:14.3f} {s:12.3f}")

for snr in [20, 10, 0]:
    noisy = pd.read_csv(f"data/processed/features_snr{snr}.csv")
    assert (noisy["filename"].values == clean["filename"].values).all()
    Xq = scaler.transform(noisy.iloc[:, 2:].values)
    g, s = evaluate(Xq)
    print(f"{str(snr) + ' dB':12s} {g:14.3f} {s:12.3f}")