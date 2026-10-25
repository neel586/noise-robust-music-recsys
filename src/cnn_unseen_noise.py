import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

N_SPLITS = 5      # 1 = quick test; set to 5 for the final run
EPOCHS = 15
LAM = 1.0         # weight of the "embedding stays the same under noise" loss
TRAIN_CONDS = ["white20", "white10", "white0"]   # pink noise is never seen in training
np.random.seed(42)
torch.manual_seed(42)
t0 = time.time()

inter = pd.read_csv("data/processed/interactions.csv")
clean_df = pd.read_csv("data/processed/features_clean.csv")
n_users, n_songs = 1000, len(clean_df)
R = np.zeros((n_users, n_songs))
R[inter["user_id"].values, inter["song_index"].values] = 1
pos = clean_df.groupby("genre").cumcount().values

conds_noisy = ["white20", "white10", "white0", "pink10", "pink0"]
tests = ["clean"] + conds_noisy
feat_files = {"white20": "features_snr20", "white10": "features_snr10", "white0": "features_snr0",
              "pink10": "features_pink_snr10", "pink0": "features_pink_snr0"}

def load_feat(name):
    df = pd.read_csv(f"data/processed/{name}.csv")
    assert (df["filename"].values == clean_df["filename"].values).all()
    return df.iloc[:, 2:].values

feat_clean = clean_df.iloc[:, 2:].values
feat_noisy = {c: load_feat(feat_files[c]) for c in conds_noisy}
mel = {c: np.load(f"data/processed/mel_{c}.npy") for c in tests}

def train_mlp(Xtr, Ytr, seed=42):
    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 17))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(500):
        opt.zero_grad()
        nn.functional.mse_loss(net(Xt), Yt).backward()
        opt.step()
    return net

def mlp_predict(net, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy()

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        def down(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.enc = nn.Sequential(down(1, 16), down(16, 32), down(32, 64),
                                 nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec = nn.Sequential(up(64, 32), up(32, 16), up(16, 8), nn.Conv2d(8, 1, 3, 1, 1))
    def forward(self, x):
        return self.dec(self.enc(x))
    def embed(self, x):
        return self.enc(x).mean(dim=(2, 3))

def train_ae(Xin, Yc, seed=42):
    torch.manual_seed(seed)
    model = AE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_in, n_c = len(Xin), len(Yc)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_in)
        total = 0.0
        for i in range(0, n_in, 64):
            idx = perm[i:i + 64]
            x = Xin[idx].float().unsqueeze(1)
            y = Yc[idx % n_c].float().unsqueeze(1)
            opt.zero_grad()
            h = model.enc(torch.cat([x, y]))
            rec = model.dec(h[:len(x)])
            z = h.mean(dim=(2, 3))
            loss = nn.functional.mse_loss(rec, y) + LAM * nn.functional.mse_loss(z[:len(x)], z[len(x):].detach())
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (ep + 1) % 5 == 0:
            print("  epoch", ep + 1, "loss", round(total / n_in, 4), " time", round(time.time() - t0), "s")
    return model

def embed_songs(model, M):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(M), 50):
            b = M[i:i + 50].float().reshape(-1, 1, 64, 128)
            out.append(model.embed(b).reshape(-1, 3, 64).mean(dim=1).numpy())
    return np.vstack(out)

results = []
for split in range(N_SPLITS):
    cold = (pos % 5 == split)
    warm = ~cold
    wi, ci = np.where(warm)[0], np.where(cold)[0]
    R_warm, R_cold = R[:, warm], R[:, cold]
    users = np.where(R_cold.sum(axis=1) > 0)[0]
    counts = np.maximum(R_warm.sum(axis=1, keepdims=True), 1)

    def recall10(profile, Xq):
        sim = cosine_similarity(profile, Xq)
        out = []
        for u in users:
            top = np.argsort(sim[u])[::-1][:10]
            out.append(R_cold[u, top].sum() / R_cold[u].sum())
        return np.mean(out)

    # handcrafted features, raw and with the neural correction
    sc_h = StandardScaler().fit(feat_clean[warm])
    Hc = sc_h.transform(feat_clean)
    Hn = {c: sc_h.transform(feat_noisy[c]) for c in conds_noisy}
    mlp = train_mlp(np.vstack([Hn[c][warm] for c in TRAIN_CONDS]),
                    np.vstack([Hc[warm]] * len(TRAIN_CONDS)))
    prof_h = (R_warm @ Hc[warm]) / counts

    # CNN denoising autoencoder
    mu = float(mel["clean"][wi].astype(np.float32).mean())
    sd = float(mel["clean"][wi].astype(np.float32).std())
    T = {c: torch.tensor((mel[c].astype(np.float32) - mu) / sd).half() for c in tests}
    Yc = T["clean"][wi].reshape(-1, 64, 128)
    Xin = torch.cat([T[c][wi].reshape(-1, 64, 128) for c in TRAIN_CONDS])
    print("split", split, ": training CNN on", len(Xin), "spectrograms")
    cnn = train_ae(Xin, Yc)
    E_clean = embed_songs(cnn, T["clean"])
    sc_e = StandardScaler().fit(E_clean[warm])
    Ec = sc_e.transform(E_clean)
    prof_e = (R_warm @ Ec[warm]) / counts

    for c in tests:
        if c == "clean":
            raw = fixed = recall10(prof_h, Hc[cold])
            cn = recall10(prof_e, Ec[cold])
        else:
            raw = recall10(prof_h, Hn[c][cold])
            fixed = recall10(prof_h, mlp_predict(mlp, Hn[c][cold]))
            cn = recall10(prof_e, sc_e.transform(embed_songs(cnn, T[c][ci])))
        results.append((split, c, raw, fixed, cn))
    print("finished split", split, " total time", round(time.time() - t0), "s")
    print(pd.DataFrame(results, columns=["split", "condition", "raw", "fixed", "cnn"]).tail(6).round(3).to_string(index=False))

cols = ["handcrafted_raw", "handcrafted_fixed", "cnn"]
df = pd.DataFrame(results, columns=["split", "condition"] + cols)
print("\nrandom guessing is about 0.05")
print("\nAVERAGE over", N_SPLITS, "split(s):")
print(df.groupby("condition")[cols].mean().loc[tests].round(3))
if N_SPLITS > 1:
    print("\nSTANDARD DEVIATION across splits:")
    print(df.groupby("condition")[cols].std().loc[tests].round(3))
nz = df[df["condition"] != "clean"]
print("\nCNN beat handcrafted_fixed in", int((nz["cnn"] > nz["handcrafted_fixed"]).sum()), "of", len(nz), "noisy cases")
print("CNN beat handcrafted_raw in", int((nz["cnn"] > nz["handcrafted_raw"]).sum()), "of", len(nz), "noisy cases")