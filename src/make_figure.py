import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

conditions = ["white 10 dB", "white 0 dB", "pink 0 dB"]
raw = [0.053, 0.047, 0.056]
fixed = [0.063, 0.058, 0.063]
clean_level = 0.063
random_level = 0.025

x = np.arange(len(conditions))
w = 0.35
plt.figure(figsize=(7, 4.5))
b1 = plt.bar(x - w / 2, raw, w, label="noisy audio, no fix")
b2 = plt.bar(x + w / 2, fixed, w, label="noisy audio + neural fix")
plt.axhline(clean_level, linestyle="--", color="green", label="clean audio (0.063)")
plt.axhline(random_level, linestyle=":", color="gray", label="random guessing (0.025)")

for bars in (b1, b2):
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                 f"{b.get_height():.3f}", ha="center", fontsize=9)

plt.xticks(x, conditions)
plt.ylabel("Recall@5 on new songs")
plt.title("Recommending new songs from noisy audio")
plt.ylim(0, 0.09)
plt.legend(loc="upper center", ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("results/new_song_recall.png", dpi=200)
print("saved results/new_song_recall.png")