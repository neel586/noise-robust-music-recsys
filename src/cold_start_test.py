import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

inter = pd.read_csv("data/processed/interactions.csv")
clean = pd.read_csv("data/processed/features_clean.csv")
n_users, n_songs = 200, len(clean)

R = np.zeros((n_users, n_songs))
R[inter["user_id"].values, inter["song_index"].values] = 1

pos = clean.groupby("genre").cumcount().values
cold = pos >= 8          # 2 songs per genre have no listening history
warm = ~cold

scaler = StandardScaler().fit(clean.iloc[:, 2:].values[warm])
Xc = scaler.transform(clean.iloc[:, 2:].values)

noisy = {}
for snr in [20, 10, 0]:
    df = pd.read_csv(f"data/processed/features_snr{snr}.csv")
    assert (df["filename"].values == clean["filename"].values).all()
    noisy[snr] = scaler.transform(df.iloc[:, 2:].values)

# the "fix" from Step 23, trained on warm songs only
fix = Ridge(alpha=1.0).fit(
    np.vstack([noisy[s][warm] for s in noisy]),
    np.vstack([Xc[warm] for s in noisy]))

R_warm = R[:, warm]
R_cold = R[:, cold]
profile = (R_warm @ Xc[warm]) / np.maximum(R_warm.sum(axis=1, keepdims=True), 1)
users = np.where(R_cold.sum(axis=1) > 0)[0]

def recall5(Xq):
    sim = cosine_similarity(profile, Xq)
    out = []
    for u in users:
        top = np.argsort(sim[u])[::-1][:5]
        out.append(R_cold[u, top].sum() / R_cold[u].sum())
    return np.mean(out)

print("cold songs:", cold.sum(), " users evaluated:", len(users))
print("random guessing: 0.250")
print(f"\n{'condition':10s} {'raw':>7s} {'fixed':>7s}")
c = recall5(Xc[cold])
print(f"{'clean':10s} {c:7.3f} {c:7.3f}")
for snr in [20, 10, 0]:
    Xn = noisy[snr][cold]
    print(f"{str(snr) + ' dB':10s} {recall5(Xn):7.3f} {recall5(fix.predict(Xn)):7.3f}")