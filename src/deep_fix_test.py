import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

torch.manual_seed(42)
np.random.seed(42)

inter = pd.read_csv("data/processed/interactions.csv")
clean = pd.read_csv("data/processed/features_clean.csv")
n_users, n_songs = 200, len(clean)

R = np.zeros((n_users, n_songs))
R[inter["user_id"].values, inter["song_index"].values] = 1

pos = clean.groupby("genre").cumcount().values
cold = pos >= 8
warm = ~cold

scaler = StandardScaler().fit(clean.iloc[:, 2:].values[warm])
Xc = scaler.transform(clean.iloc[:, 2:].values)

noisy = {}
for snr in [20, 10, 0]:
    df = pd.read_csv(f"data/processed/features_snr{snr}.csv")
    assert (df["filename"].values == clean["filename"].values).all()
    noisy[snr] = scaler.transform(df.iloc[:, 2:].values)

Xtr = np.vstack([noisy[s][warm] for s in noisy])
Ytr = np.vstack([Xc[warm] for s in noisy])

# Linear fix (from before)
ridge = Ridge(alpha=1.0).fit(Xtr, Ytr)

# Neural network fix
Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
net = nn.Sequential(
    nn.Linear(17, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 17),
)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(500):
    opt.zero_grad()
    loss = loss_fn(net(Xtr_t), Ytr_t)
    loss.backward()
    opt.step()
    if (epoch + 1) % 100 == 0:
        print("epoch", epoch + 1, "train loss", round(loss.item(), 4))

def mlp_predict(X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy()

# Cold-start evaluation (same as Step 27)
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

print("\nrandom guessing: 0.250")
print(f"{'condition':10s} {'raw':>7s} {'linear':>8s} {'neural':>8s}")
c = recall5(Xc[cold])
print(f"{'clean':10s} {c:7.3f} {c:8.3f} {c:8.3f}")
for snr in [20, 10, 0]:
    Xn = noisy[snr][cold]
    print(f"{str(snr) + ' dB':10s} {recall5(Xn):7.3f} {recall5(ridge.predict(Xn)):8.3f} {recall5(mlp_predict(Xn)):8.3f}")