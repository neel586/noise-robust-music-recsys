import numpy as np
import pandas as pd

np.random.seed(42)

songs = pd.read_csv("data/processed/features_clean.csv")[["filename", "genre"]]
genre_list = sorted(songs["genre"].unique())
N_USERS = 1000

inter_rows = []
fav_rows = []
in_fav = 0

for u in range(N_USERS):
    favs = list(np.random.choice(genre_list, size=2, replace=False))
    fav_rows.append([u, favs[0], favs[1]])
    for idx, g in enumerate(songs["genre"]):
        p = 0.2 if g in favs else 0.01
        if np.random.rand() < p:
            inter_rows.append([u, idx, songs["filename"][idx]])
            if g in favs:
                in_fav += 1

inter = pd.DataFrame(inter_rows, columns=["user_id", "song_index", "filename"])
fav = pd.DataFrame(fav_rows, columns=["user_id", "fav1", "fav2"])
inter.to_csv("data/processed/interactions.csv", index=False)
fav.to_csv("data/processed/user_favorites.csv", index=False)

per_user = inter.groupby("user_id").size()
print("total listens:", len(inter))
print("songs per user: mean", round(per_user.mean(), 1), " min", per_user.min(), " max", per_user.max())
print("listens inside favorite genres:", round(in_fav / len(inter), 3))
print("matrix density:", round(len(inter) / (N_USERS * len(songs)), 3))