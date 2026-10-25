import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

inter = pd.read_csv("data/processed/interactions.csv")
clean = pd.read_csv("data/processed/features_clean.csv")
n_users, n_songs = 1000, len(clean)

R = np.zeros((n_users, n_songs))
R[inter["user_id"].values, inter["song_index"].values] = 1

pos = clean.groupby("genre").cumcount().values
clean_raw = clean.iloc[:, 2:].values

noisy_raw = {}
for snr in [20, 10, 0]:
    df = pd.read_csv(f"data/processed/features_snr{snr}.csv")
    assert (df["filename"].values == clean["filename"].values).all()
    noisy_raw[snr] = df.iloc[:, 2:].values

def train_mlp(Xtr, Ytr, seed=42):
    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(17, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, 17))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(500):
        opt.zero_grad()
        loss_fn(net(Xt), Yt).backward()
        opt.step()
    return net

results = []
for split in range(5):
    cold = (pos % 5 == split)
    warm = ~cold

    scaler = StandardScaler().fit(clean_raw[warm])
    Xc = scaler.transform(clean_raw)
    noisy = {s: scaler.transform(noisy_raw[s]) for s in noisy_raw}

    Xtr = np.vstack([noisy[s][warm] for s in noisy])
    Ytr = np.vstack([Xc[warm] for s in noisy])
    ridge = Ridge(alpha=1.0).fit(Xtr, Ytr)
    net = train_mlp(Xtr, Ytr)

    def mlp_predict(X):
        with torch.no_grad():
            return net(torch.tensor(X, dtype=torch.float32)).numpy()

    R_warm, R_cold = R[:, warm], R[:, cold]
    profile = (R_warm @ Xc[warm]) / np.maximum(R_warm.sum(axis=1, keepdims=True), 1)
    users = np.where(R_cold.sum(axis=1) > 0)[0]

    def recall10(Xq):
        sim = cosine_similarity(profile, Xq)
        out = []
        for u in users:
            top = np.argsort(sim[u])[::-1][:10]
            out.append(R_cold[u, top].sum() / R_cold[u].sum())
        return np.mean(out)

    if split == 0:
        print("cold songs per split:", int(cold.sum()), " users evaluated:", len(users))

    c = recall10(Xc[cold])
    results.append((split, "clean", c, c, c))
    for snr in [20, 10, 0]:
        Xn = noisy[snr][cold]
        results.append((split, f"{snr} dB", recall10(Xn),
                        recall10(ridge.predict(Xn)), recall10(mlp_predict(Xn))))
    print("finished split", split)

df = pd.DataFrame(results, columns=["split", "condition", "raw", "linear", "neural"])
order = ["clean", "20 dB", "10 dB", "0 dB"]

print("\nrandom guessing is about 0.05; genre-only ceiling is roughly 0.2")
print("\nAVERAGE over 5 splits:")
print(df.groupby("condition")[["raw", "linear", "neural"]].mean().loc[order].round(3))
print("\nSTANDARD DEVIATION across splits:")
print(df.groupby("condition")[["raw", "linear", "neural"]].std().loc[order].round(3))

noisy_rows = df[df["condition"] != "clean"]
print("\nneural beat linear in", int((noisy_rows["neural"] > noisy_rows["linear"]).sum()),
      "of", len(noisy_rows), "cases")
print("neural beat raw in", int((noisy_rows["neural"] > noisy_rows["raw"]).sum()),
      "of", len(noisy_rows), "cases")
print("linear beat raw in", int((noisy_rows["linear"] > noisy_rows["raw"]).sum()),
      "of", len(noisy_rows), "cases")