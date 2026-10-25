import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

np.random.seed(42)

inter = pd.read_csv("data/processed/interactions.csv")
n_users, n_songs = 1000, 999

R = np.zeros((n_users, n_songs))
test_mask = np.zeros((n_users, n_songs), dtype=bool)

for u, grp in inter.groupby("user_id"):
    idx = grp["song_index"].values.copy()
    np.random.shuffle(idx)
    k = max(1, int(round(0.2 * len(idx))))
    test_mask[u, idx[:k]] = True
    R[u, idx] = 1

R_train = R.copy()
R_train[test_mask] = 0

np.save("data/processed/R_train.npy", R_train)
np.save("data/processed/test_mask.npy", test_mask)

def recall_at_k(scores, k=10):
    s = scores.copy()
    s[R_train == 1] = -np.inf          # never recommend songs already heard
    out = []
    for u in range(n_users):
        top = np.argsort(s[u])[::-1][:k]
        out.append(test_mask[u, top].sum() / test_mask[u].sum())
    return np.mean(out)

svd = TruncatedSVD(n_components=10, random_state=42)
U = svd.fit_transform(R_train)
cf_scores = U @ svd.components_

pop = R_train.sum(axis=0)
pop_scores = np.tile(pop, (n_users, 1)).astype(float)
rand_scores = np.random.rand(n_users, n_songs)

print("train listens:", int(R_train.sum()), " hidden listens:", int(test_mask.sum()))
print(f"\n{'method':12s} {'Recall@10':>10s}")
print(f"{'random':12s} {recall_at_k(rand_scores):10.3f}")
print(f"{'popularity':12s} {recall_at_k(pop_scores):10.3f}")
print(f"{'CF (SVD)':12s} {recall_at_k(cf_scores):10.3f}")