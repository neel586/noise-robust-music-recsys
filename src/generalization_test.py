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

def load(path):
    df = pd.read_csv(path)
    assert (df["filename"].values == clean["filename"].values).all()
    return df.iloc[:, 2:].values

white_raw = {s: load(f"data/processed/features_snr{s}.csv") for s in [20, 10, 0]}
pink_raw = {s: load(f"data/processed/features_pink_snr{s}.csv") for s in [10, 0]}

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

def predict(net, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy()

results = []
for split in range(5):
    cold = (pos % 5 == split)
    warm = ~cold

    scaler = StandardScaler().fit(clean_raw[warm])
    Xc = scaler.transform(clean_raw)
    white = {s: scaler.transform(white_raw[s]) for s in white_raw}
    pink = {s: scaler.transform(pink_raw[s]) for s in pink_raw}

    # trained on white noise only
    Xw = np.vstack([white[s][warm] for s in white])
    Yw = np.vstack([Xc[warm] for s in white])
    ridge = Ridge(alpha=1.0).fit(Xw, Yw)
    net_w = train_mlp(Xw, Yw)

    # trained on white + pink noise
    Xm = np.vstack([Xw] + [pink[s][warm] for s in pink])
    Ym = np.vstack([Yw] + [Xc[warm] for s in pink])
    net_m = train_mlp(Xm, Ym)

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

    c = recall10(Xc[cold])
    results.append((split, "clean", c, c, c, c))

    tests = [("white 10 dB", white[10]), ("white 0 dB", white[0]),
             ("pink 10 dB", pink[10]), ("pink 0 dB", pink[0])]
    for name, X in tests:
        Xn = X[cold]
        results.append((split, name, recall10(Xn),
                        recall10(ridge.predict(Xn)),
                        recall10(predict(net_w, Xn)),
                        recall10(predict(net_m, Xn))))
    print("finished split", split)

df = pd.DataFrame(results, columns=["split", "condition", "raw", "linear", "neural", "neural_mixed"])
order = ["clean", "white 10 dB", "white 0 dB", "pink 10 dB", "pink 0 dB"]
cols = ["raw", "linear", "neural", "neural_mixed"]

print("\nrandom guessing is about 0.05")
print("\nAVERAGE over 5 splits:")
print(df.groupby("condition")[cols].mean().loc[order].round(3))
print("\nSTANDARD DEVIATION across splits:")
print(df.groupby("condition")[cols].std().loc[order].round(3))

p = df[df["condition"].str.startswith("pink")]
print("\nOn pink noise (", len(p), "cases ):")
print("  neural (white-trained) beat raw in", int((p["neural"] > p["raw"]).sum()))
print("  linear (white-trained) beat raw in", int((p["linear"] > p["raw"]).sum()))
print("  neural_mixed beat neural in", int((p["neural_mixed"] > p["neural"]).sum()))