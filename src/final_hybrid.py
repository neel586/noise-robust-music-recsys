import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.width", 200)
np.random.seed(42)
ALPHA = 0.5   # weight on CF for songs that have listening history

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
    # z-score each user's scores using only the songs where mask is True
    S2 = np.where(mask, S, np.nan)
    mu = np.nanmean(S2, axis=1, keepdims=True)
    sd = np.nanstd(S2, axis=1, keepdims=True) + 1e-9
    return (S - mu) / sd

conditions = ["clean", "white 10 dB", "white 0 dB", "pink 0 dB"]
results = []
rand_scores = []

for split in range(5):
    rng = np.random.RandomState(split)
    cold = (pos % 5 == split)     # the new songs
    warm = ~cold

    # hide every listen of a new song, plus 20% of listens on the other songs
    hide_warm = (R == 1) & warm[None, :] & (rng.rand(n_users, n_songs) < 0.2)
    test_mask = ((R == 1) & cold[None, :]) | hide_warm
    R_train = R.copy()
    R_train[test_mask] = 0
    if split == 0:
        print("share of hidden listens that are on new songs:",
              round(test_mask[:, cold].sum() / test_mask.sum(), 3))

    def recall10(final):
        s = np.where(R_train == 1, -np.inf, final)
        top = np.argsort(-s, axis=1)[:, :10]
        hits = np.take_along_axis(test_mask, top, axis=1).sum(axis=1)
        tot = test_mask.sum(axis=1)
        ok = tot > 0
        return np.mean(hits[ok] / tot[ok])

    scaler = StandardScaler().fit(clean_raw[warm])
    Xc = scaler.transform(clean_raw)
    white = {s: scaler.transform(white_raw[s]) for s in white_raw}
    pink = {s: scaler.transform(pink_raw[s]) for s in pink_raw}

    # the noise fix: neural network trained on white + pink noise, older songs only
    Xtr = np.vstack([white[s][warm] for s in white] + [pink[s][warm] for s in pink])
    Ytr = np.vstack([Xc[warm]] * (len(white) + len(pink)))
    net = train_mlp(Xtr, Ytr)

    # collaborative filtering
    svd = TruncatedSVD(n_components=10, random_state=42)
    U = svd.fit_transform(R_train)
    cf = U @ svd.components_

    warm_un = warm[None, :] & (R_train == 0)
    cold_m = np.broadcast_to(cold[None, :], (n_users, n_songs))
    cf_z = zmask(cf, warm_un)

    # each user's taste profile, built from the clean features of songs they heard
    profile = (R_train @ Xc) / np.maximum(R_train.sum(axis=1, keepdims=True), 1)

    def audio_final(Xq_cold, use_cf):
        Xcand = Xc.copy()
        Xcand[cold] = Xq_cold          # new songs arrive in their noisy/fixed form
        au = cosine_similarity(profile, Xcand)
        au_w = zmask(au, warm_un)
        au_c = zmask(au, cold_m)
        if use_cf:
            warm_part = zmask(ALPHA * cf_z + (1 - ALPHA) * au_w, warm_un)
        else:
            warm_part = au_w
        return np.where(cold[None, :], au_c, warm_part)

    cf_only = recall10(np.where(cold[None, :], -np.inf, cf))
    rand_scores.append(recall10(np.random.rand(n_users, n_songs)))

    tests = {"clean": Xc[cold],
             "white 10 dB": white[10][cold],
             "white 0 dB": white[0][cold],
             "pink 0 dB": pink[0][cold]}
    for name in conditions:
        Xq = tests[name]
        Xf = Xq if name == "clean" else predict(net, Xq)
        results.append((split, name, cf_only,
                        recall10(audio_final(Xq, False)),
                        recall10(audio_final(Xf, False)),
                        recall10(audio_final(Xq, True)),
                        recall10(audio_final(Xf, True))))
    print("finished split", split)

cols = ["cf_only", "audio_raw", "audio_fixed", "hybrid_raw", "hybrid_fixed"]
df = pd.DataFrame(results, columns=["split", "condition"] + cols)
print("\nrandom guessing:", round(np.mean(rand_scores), 3))
print("\nAVERAGE over 5 splits:")
print(df.groupby("condition")[cols].mean().loc[conditions].round(3))
print("\nSTANDARD DEVIATION across splits:")
print(df.groupby("condition")[cols].std().loc[conditions].round(3))