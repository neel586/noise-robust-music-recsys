import numpy as np
import librosa

def extract_features(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    return np.concatenate([[centroid, bandwidth, rolloff, zcr], mfcc])

if __name__ == "__main__":
    y, sr = librosa.load(librosa.example('trumpet'))
    f = extract_features(y, sr)
    print("number of features:", len(f))
    print(np.round(f, 2))