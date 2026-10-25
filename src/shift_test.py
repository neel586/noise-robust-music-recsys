import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def genre_at5(df):
    X = StandardScaler().fit_transform(df.iloc[:, 2:].values)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -1)
    genres = df["genre"].values
    scores = []
    for i in range(len(df)):
        top5 = np.argsort(sim[i])[::-1][:5]
        scores.append(np.mean(genres[top5] == genres[i]))
    return np.mean(scores)

print(f"{'condition':10s} {'same-genre@5':>13s}")
print(f"{'clean':10s} {genre_at5(pd.read_csv('data/processed/features_clean.csv')):13.3f}")
for snr in [20, 10, 0]:
    df = pd.read_csv(f"data/processed/features_snr{snr}.csv")
    print(f"{str(snr) + ' dB':10s} {genre_at5(df):13.3f}")