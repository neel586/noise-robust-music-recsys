import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.width", 200)
np.random.seed(42)
ALPHA = 0.5   # CF weight in the blend for older songs
K = 5         # slots for older songs, and again for new songs

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

def zmask(S, mask):
    S2 = np.where(mask, S, np.nan)
    mu = np.nanmean(S2, axis=1, keepdims=True)
    sd = np.nanstd(S2, axis=1, keepdims=True) + 1e-9
    return (S - mu) / sd

conditions = ["clean", "white 10 dB", "white 0 dB", "pink 0 dB"]
comb_rows, cold_rows, warm_rows, rand_rows = [], [], [], []

for split in range(5):
    rng = np.random.RandomState(split)
    cold = (pos % 5 == split)     # new songs: no listening history
    warm = ~cold

    hide_warm = (R == 1) & warm[None, :] & (rng.rand(n_users, n_songs) < 0.2)
    test_mask = ((R == 1) & cold[None, :]) | hide_warm
    R_train = R.copy()
    R_train[test_mask] = 0

    tot_w = (test_mask & warm[None, :]).sum(axis=1)
    tot_c = (test_mask & cold[None, :]).sum(axis=1)
    tot = tot_w + tot_c
    if split == 0:
        print("share of hidden listens on new songs:", round(tot_c.sum() / tot.sum(), 3))

    cand_w = warm[None, :] & (R_train == 0)
    cand_c = np.broadcast_to(cold[None, :], (n_users, n_songs))

    def top_hits(S, cand):
        s = np.where(cand, S, -np.inf)
        top = np.argsort(-s, axis=1)[:, :K]
        return np.take_along_axis(test_mask, top, axis=1).sum(axis=1)

    def mean_ratio(h, t):
        ok = t > 0
        return np.mean(h[ok] / t[ok])

    scaler = StandardScaler().fit(clean_raw[warm])
    Xc = scaler.transform(clean_raw)
    white = {s: scaler.transform(white_raw[s]) for s in white_raw}
    pink = {s: scaler.transform(pink_raw[s]) for s in pink_raw}

    Xtr = np.vstack([white[s][warm] for s in white] + [pink[s][warm] for s in pink])
    Ytr = np.vstack([Xc[warm]] * (len(white) + len(pink)))
    net = train_mlp(Xtr, Ytr)

    svd = TruncatedSVD(n_components=10, random_state=42)
    U = svd.fit_transform(R_train)
    cf = U @ svd.components_

    profile = (R_train @ Xc) / np.maximum(R_train.sum(axis=1, keepdims=True), 1)
    au_warm = cosine_similarity(profile, Xc)
    blend = ALPHA * zmask(cf, cand_w) + (1 - ALPHA) * zmask(au_warm, cand_w)

    hw = {"cf": top_hits(cf, cand_w), "blend": top_hits(blend, cand_w)}
    for k, v in hw.items():
        warm_rows.append((split, k, mean_ratio(v, tot_w)))

    hw_r = top_hits(rng.rand(n_users, n_songs), cand_w)
    hc_r = top_hits(rng.rand(n_users, n_songs), cand_c)
    rand_rows.append(mean_ratio(hw_r + hc_r, tot))

    tests = {"clean": Xc[cold],
             "white 10 dB": white[10][cold],
             "white 0 dB": white[0][cold],
             "pink 0 dB": pink[0][cold]}
    for name in conditions:
        Xq = tests[name]
        Xf = Xq if name == "clean" else predict(net, Xq)
        hc = {}
        for lab, X in [("raw", Xq), ("fixed", Xf)]:
            S = np.full((n_users, n_songs), -np.inf)
            S[:, cold] = cosine_similarity(profile, X)
            hc[lab] = top_hits(S, cand_c)
            cold_rows.append((split, name, lab, mean_ratio(hc[lab], tot_c)))
        for wl in hw:
            for cl in hc:
                comb_rows.append((split, name, f"{wl}+{cl}", mean_ratio(hw[wl] + hc[cl], tot)))
    print("finished split", split)

print("\nrandom guessing (5 + 5 slots):", round(np.mean(rand_rows), 3))

comb = pd.DataFrame(comb_rows, columns=["split", "condition", "system", "recall"])
print("\nCOMBINED (older-song slots + new-song slots), average over 5 splits:")
print(comb.pivot_table(index="condition", columns="system", values="recall", aggfunc="mean").loc[conditions].round(3))
print("\nstandard deviation across splits:")
print(comb.pivot_table(index="condition", columns="system", values="recall", aggfunc="std").loc[conditions].round(3))

cold_df = pd.DataFrame(cold_rows, columns=["split", "condition", "audio", "recall"])
print("\nNEW SONGS only (5 slots), average:")
print(cold_df.pivot_table(index="condition", columns="audio", values="recall", aggfunc="mean").loc[conditions].round(3))

warm_df = pd.DataFrame(warm_rows, columns=["split", "method", "recall"])
print("\nOLDER SONGS only (5 slots), average and standard deviation:")
print(warm_df.groupby("method")["recall"].agg(["mean", "std"]).round(3))