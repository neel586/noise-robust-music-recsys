import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/features_clean.csv")
print("table size:", df.shape)

X = StandardScaler().fit_transform(df.iloc[:, 2:].values)
sim = cosine_similarity(X)
np.fill_diagonal(sim, -1)   # a song should not match itself

# Demo: pick the first jazz song and show its 5 closest matches
q = df.index[df["genre"] == "jazz"][0]
top = np.argsort(sim[q])[::-1][:5]
print("\nQuery song:", df.loc[q, "filename"], "(" + df.loc[q, "genre"] + ")")
print("Top 5 similar songs:")
for i in top:
    print("  ", df.loc[i, "filename"], "(" + df.loc[i, "genre"] + ")", round(sim[q, i], 3))

# Overall score: how often are the 5 neighbours the same genre as the query?
scores = []
for i in range(len(df)):
    top5 = np.argsort(sim[i])[::-1][:5]
    scores.append(np.mean(df["genre"].values[top5] == df["genre"].values[i]))
print("\nAverage same-genre rate in top 5:", round(np.mean(scores), 3))