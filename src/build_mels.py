import os
import numpy as np
import pandas as pd
import librosa

np.random.seed(11)

DATA_DIR = os.path.join("data", "raw", "genres_original")
OUT_DIR = os.path.join("data", "processed")
STARTS = [0, 220500, 441000]     # seconds 0, 10, 20 at 22,050 samples per second
CHUNK = 66150                    # 3 seconds
N_MELS, N_FRAMES = 64, 128

def pink_noise(n):
    spectrum = np.fft.rfft(np.random.randn(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    spectrum = spectrum / np.sqrt(freqs)
    noise = np.fft.irfft(spectrum, n)
    return noise / np.std(noise)

def mel(chunk, sr):
    S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=1024,
                                       hop_length=512, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=1.0, top_db=None)[:, :N_FRAMES]

conditions = {"clean": None, "white20": ("white", 20), "white10": ("white", 10),
              "white0": ("white", 0), "pink10": ("pink", 10), "pink0": ("pink", 0)}
store = {c: [] for c in conditions}
names = []

for genre in sorted(os.listdir(DATA_DIR)):
    genre_dir = os.path.join(DATA_DIR, genre)
    for f in sorted(os.listdir(genre_dir))[:100]:
        try:
            y, sr = librosa.load(os.path.join(genre_dir, f), duration=30)
        except Exception as e:
            print("skipped", f, "-", e)
            continue
        if len(y) < STARTS[-1] + CHUNK:
            y = np.pad(y, (0, STARTS[-1] + CHUNK - len(y)))
        names.append(f)
        power = np.mean(y ** 2)
        for cond, spec in conditions.items():
            if spec is None:
                yc = y
            else:
                kind, snr = spec
                noise = np.random.randn(len(y)) if kind == "white" else pink_noise(len(y))
                yc = y + noise * np.sqrt(power / (10 ** (snr / 10)))
            store[cond].append(np.stack([mel(yc[s:s + CHUNK], sr) for s in STARTS]))
    print("done:", genre)

for cond in conditions:
    arr = np.array(store[cond], dtype=np.float16)
    np.save(os.path.join(OUT_DIR, f"mel_{cond}.npy"), arr)
    print("saved", cond, arr.shape)

ref = pd.read_csv(os.path.join(OUT_DIR, "features_clean.csv"))["filename"].values
print("rows match features_clean.csv:", list(ref) == names)