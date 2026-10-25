import numpy as np
import librosa
from features import extract_features

np.random.seed(42)

names = ["centroid", "bandwidth", "rolloff", "zcr"] + [f"mfcc{i+1}" for i in range(13)]

y, sr = librosa.load(librosa.example('trumpet'))
clean = extract_features(y, sr)

snr_db = 10
signal_power = np.mean(y ** 2)
noise_power = signal_power / (10 ** (snr_db / 10))
y_noisy = y + np.random.randn(len(y)) * np.sqrt(noise_power)
noisy = extract_features(y_noisy, sr)

print(f"{'feature':10s} {'clean':>10s} {'noisy':>10s} {'change %':>10s}")
for n, c, nz in zip(names, clean, noisy):
    print(f"{n:10s} {c:10.2f} {nz:10.2f} {(nz - c) / abs(c) * 100:9.1f}%")