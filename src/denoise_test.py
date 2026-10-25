import numpy as np
import librosa
import noisereduce as nr
from features import extract_features

np.random.seed(42)

names = ["centroid", "bandwidth", "rolloff", "zcr"]

y, sr = librosa.load(librosa.example('trumpet'))
clean = extract_features(y, sr)

snr_db = 10
signal_power = np.mean(y ** 2)
noise_power = signal_power / (10 ** (snr_db / 10))
y_noisy = y + np.random.randn(len(y)) * np.sqrt(noise_power)

y_denoised = nr.reduce_noise(y=y_noisy, sr=sr, stationary=True)

noisy = extract_features(y_noisy, sr)
denoised = extract_features(y_denoised, sr)

print(f"{'feature':10s} {'clean':>10s} {'noisy':>10s} {'denoised':>10s}")
for i, n in enumerate(names):
    print(f"{n:10s} {clean[i]:10.2f} {noisy[i]:10.2f} {denoised[i]:10.2f}")