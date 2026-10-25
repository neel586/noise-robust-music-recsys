import numpy as np
import librosa

np.random.seed(42)  # makes the random noise repeatable

y, sr = librosa.load(librosa.example('trumpet'))
signal_power = np.mean(y ** 2)

clean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
print("clean:", clean)

for snr_db in [30, 20, 10, 0]:
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
    y_noisy = y + noise
    c = np.mean(librosa.feature.spectral_centroid(y=y_noisy, sr=sr))
    print(snr_db, "dB:", c)