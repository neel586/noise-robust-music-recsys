import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

R_train = np.load("data/processed/R_train.npy")
test_mask = np.load("data/processed/test_mask.npy")
n_users, n_songs = R_train.shape

clean = pd.read_csv("data/processed/features_clean.csv")
scaler = StandardScaler().fit(clean.iloc[:, 2:].values)

def get_features(snr):
    if snr is None:
        df = clean
    else:
        df = pd.read_csv(f"data/processed/features_snr{snr}.csv")
        assert (df["filename"].values == clean["filename"].values).all()
    return scaler.transform(df.iloc[:, 2:].values)

def recall_at_k(scores, k=10):
    s = scores.copy()
    s[R_train == 1] = -np.inf
    out = []
    for u in range(n_users):
        top = np.argsort(s[u])[::-1][:k]
        out.append(test_mask[u, top].sum() / test_mask[u].sum())
    return np.mean(out)

def zrow(s):
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)

# CF scores (same model as before)
svd = TruncatedSVD(n_components=10, random_state=42)
U = svd.fit_transform(R_train)
cf = zrow(U @ svd.components_)
cf_recall = recall_at_k(cf)

def audio_scores(X):
    profile = (R_train @ X) / R_train.sum(axis=1, keepdims=True)
    return zrow(cosine_similarity(profile, X))

ALPHA = 0.5   # weight on CF; the rest goes to audio

print(f"{'condition':10s} {'audio-only':>11s} {'CF-only':>9s} {'hybrid':>8s}")
for label, snr in [("clean", None), ("20 dB", 20), ("10 dB", 10), ("0 dB", 0)]:
    au = audio_scores(get_features(snr))
    hy = ALPHA * cf + (1 - ALPHA) * au
    print(f"{label:10s} {recall_at_k(au):11.3f} {cf_recall:9.3f} {recall_at_k(hy):8.3f}")