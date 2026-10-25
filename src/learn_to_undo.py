import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

clean = pd.read_csv("data/processed/features_clean.csv")
pos = clean.groupby("genre").cumcount().values
train = pos < 7          # first 7 songs of each genre
test = ~train            # last 3 songs of each genre

scaler = StandardScaler().fit(clean.iloc[:, 2:].values[train])
Xc = scaler.transform(clean.iloc[:, 2:].values)
genres = clean["genre"].values

noisy = {}
for snr in [20, 10, 0]:
    df = pd.read_csv(f"data/processed/features_snr{snr}.csv")
    assert (df["filename"].values == clean["filename"].values).all()
    noisy[snr] = scaler.transform(df.iloc[:, 2:].values)

# Train: noisy features (all 3 levels mixed) -> clean features
X_train = np.vstack([noisy[s][train] for s in noisy])
Y_train = np.vstack([Xc[train] for s in noisy])
model = Ridge(alpha=1.0).fit(X_train, Y_train)

Xc_test = Xc[test]
g_test = genres[test]
m = len(Xc_test)
print("train songs:", train.sum(), " test songs:", m)

def evaluate(Xq):
    sim = cosine_similarity(Xq, Xc_test)
    self_rate = np.mean(sim.argmax(axis=1) == np.arange(m))
    sim_ex = sim.copy()
    np.fill_diagonal(sim_ex, -1)
    scores = []
    for i in range(m):
        top5 = np.argsort(sim_ex[i])[::-1][:5]
        scores.append(np.mean(g_test[top5] == g_test[i]))
    return np.mean(scores), self_rate

g, s = evaluate(Xc_test)
print(f"\n{'condition':10s} {'raw genre':>10s} {'fixed genre':>12s} {'raw self':>9s} {'fixed self':>11s}")
print(f"{'clean':10s} {g:10.3f} {g:12.3f} {s:9.3f} {s:11.3f}")
for snr in [20, 10, 0]:
    Xn = noisy[snr][test]
    gr, sr_ = evaluate(Xn)
    gf, sf = evaluate(model.predict(Xn))
    print(f"{str(snr) + ' dB':10s} {gr:10.3f} {gf:12.3f} {sr_:9.3f} {sf:11.3f}")