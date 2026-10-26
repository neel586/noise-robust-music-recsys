import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

os.makedirs("results", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# -------------------------------------------------------------
# FIGURE 1: Benchmark Recall@10 across Conditions
# -------------------------------------------------------------
conditions = [
    "Clean Audio", 
    "White 20 dB", 
    "White 10 dB", 
    "White 0 dB (Severe)", 
    "Pink 0 dB (Seen)", 
    "Pink 0 dB (Unseen)"
]
raw = [0.120, 0.118, 0.106, 0.091, 0.109, 0.109]
ridge = [0.120, 0.119, 0.112, 0.109, 0.118, 0.111]
mlp = [0.120, 0.120, 0.115, 0.114, 0.124, 0.117]
cnn = [0.127, 0.125, 0.121, 0.115, 0.120, 0.102]

x = np.arange(len(conditions))
w = 0.20

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
b1 = ax.bar(x - 1.5*w, raw, w, label="Raw Audio (No Fix)", color="#94A3B8", edgecolor="black", lw=0.5)
b2 = ax.bar(x - 0.5*w, ridge, w, label="Linear Ridge Fix", color="#60A5FA", edgecolor="black", lw=0.5)
b3 = ax.bar(x + 0.5*w, mlp, w, label="Neural MLP Fix (Ours)", color="#2563EB", edgecolor="black", lw=0.5)
b4 = ax.bar(x + 1.5*w, cnn, w, label="CNN Denoising Autoencoder", color="#7C3AED", edgecolor="black", lw=0.5)

ax.axhline(0.050, color='#DC2626', linestyle='--', linewidth=1.5, label='Random Guessing Baseline (~0.050)')
ax.set_ylabel("Recall@10 on New Songs", fontsize=11, fontweight='bold')
ax.set_title("Recommendation Accuracy (Recall@10) Under Additive Noise Environments\n(5-Fold Cross-Validation, 200 Cold Songs per Split)", fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=15, ha='right', fontsize=9.5, fontweight='medium')
ax.set_ylim(0, 0.148)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right', fontsize=9)

# Value annotations on the 0 dB severe bars
ax.annotate(f"{raw[3]:.3f}\n(-24%)", xy=(x[3]-1.5*w, raw[3]), xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8, color="#DC2626", fontweight='bold')
ax.annotate(f"{mlp[3]:.3f}\n(+25%)", xy=(x[3]+0.5*w, mlp[3]), xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8, color="#15803D", fontweight='bold')
ax.annotate(f"{cnn[5]:.3f}\n(Overfit)", xy=(x[5]+1.5*w, cnn[5]), xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8, color="#B91C1C", fontweight='bold')

plt.tight_layout()
fig1_path = os.path.join("results", "benchmark_recall_comparison.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print("Saved:", fig1_path)

# -------------------------------------------------------------
# FIGURE 2: SNR Degradation & Recovery Curves
# -------------------------------------------------------------
snr_levels = [np.nan, 20, 10, 0] # Clean mapped to high SNR
snr_ticks = ["Clean", "20 dB", "10 dB", "0 dB"]
x_snr = np.arange(len(snr_ticks))

raw_curve = [0.120, 0.118, 0.106, 0.091]
mlp_curve = [0.120, 0.120, 0.115, 0.114]
cnn_curve = [0.127, 0.125, 0.121, 0.115]

fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
ax.plot(x_snr, raw_curve, 'o--', color='#DC2626', lw=2.2, markersize=8, label='Raw Audio (Degrading toward Random)')
ax.plot(x_snr, mlp_curve, 's-', color='#2563EB', lw=2.5, markersize=8, label='Handcrafted + Neural MLP Fix (Robust)')
ax.plot(x_snr, cnn_curve, '^-', color='#7C3AED', lw=2.2, markersize=8, label='CNN Autoencoder (Consistent on Seen Noise)')
ax.axhline(0.050, color='gray', linestyle=':', label='Random Guessing (0.050)')

for i in range(len(x_snr)):
    ax.text(x_snr[i], raw_curve[i] - 0.0035, f"{raw_curve[i]:.3f}", ha='center', fontsize=9, color='#DC2626')
    ax.text(x_snr[i], mlp_curve[i] + 0.0025, f"{mlp_curve[i]:.3f}", ha='center', fontsize=9, color='#2563EB')

ax.set_xticks(x_snr)
ax.set_xticklabels(snr_ticks, fontsize=10, fontweight='bold')
ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=11, fontweight='bold')
ax.set_ylabel("Recall@10 on New Songs", fontsize=11, fontweight='bold')
ax.set_title("Performance Degradation vs. Deep Learning Noise Resilience", fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0.04, 0.138)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower left')

plt.tight_layout()
fig2_path = os.path.join("results", "snr_degradation_curve.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print("Saved:", fig2_path)

# -------------------------------------------------------------
# FIGURE 3: 17-Dimensional Feature Vector Shift & Neural Reconstruction
# -------------------------------------------------------------
clean_df = pd.read_csv("data/processed/features_clean.csv")
noisy0_df = pd.read_csv("data/processed/features_snr0.csv")

clean_raw = clean_df.iloc[:, 2:].values
noisy0_raw = noisy0_df.iloc[:, 2:].values

scaler = StandardScaler().fit(clean_raw)
Xc = scaler.transform(clean_raw)
Xn = scaler.transform(noisy0_raw)

# Fast 100-step MLP for demonstration of feature recovery
torch.manual_seed(42)
net = nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 17))
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
Xt = torch.tensor(Xn, dtype=torch.float32)
Yt = torch.tensor(Xc, dtype=torch.float32)
for _ in range(300):
    opt.zero_grad()
    nn.functional.mse_loss(net(Xt), Yt).backward()
    opt.step()

with torch.no_grad():
    X_restored = net(Xt).numpy()

# Select blues track: index 0
song_idx = 0
feat_clean = Xc[song_idx]
feat_noisy = Xn[song_idx]
feat_rest = X_restored[song_idx]

feat_labels = ["Centroid", "Bandwidth", "Rolloff", "ZCR"] + [f"MFCC-{i}" for i in range(1, 14)]
idx_feats = np.arange(17)
w_feat = 0.28

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=300)
ax.bar(idx_feats - w_feat, feat_clean, w_feat, label="Clean Ground Truth", color='#2563EB', alpha=0.9)
ax.bar(idx_feats, feat_noisy, w_feat, label="0 dB Noise Corrupted", color='#EF4444', alpha=0.9)
ax.bar(idx_feats + w_feat, feat_rest, w_feat, label="Neural MLP Restored", color='#10B981', alpha=0.9)

ax.set_xticks(idx_feats)
ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Standardized Value (z-score)", fontsize=10, fontweight='bold')
ax.set_title("Acoustic Distortion and Neural Inverse Mapping Across 17 Audio Descriptors\n(Example: blues.00000.wav under 0 dB White Noise)", fontsize=11, fontweight='bold', pad=10)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')

plt.tight_layout()
fig3_path = os.path.join("results", "feature_distortion_shift.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print("Saved:", fig3_path)

# -------------------------------------------------------------
# FIGURE 4: Spectrogram Comparison (Clean vs Noisy)
# -------------------------------------------------------------
audio_file = os.path.join("data", "raw", "genres_original", "blues", "blues.00000.wav")
if os.path.exists(audio_file):
    y, sr = librosa.load(audio_file, duration=5, sr=22050)
    sig_power = np.mean(y**2)
    noise = np.random.randn(len(y)) * np.sqrt(sig_power) # 0 dB noise
    y_noisy = y + noise

    S_clean = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64), ref=np.max)
    S_noisy = librosa.power_to_db(librosa.feature.melspectrogram(y=y_noisy, sr=sr, n_mels=64), ref=np.max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=300)
    im1 = ax1.imshow(S_clean, aspect="auto", origin="lower", cmap="magma", extent=[0, 5, 0, sr/2])
    ax1.set_title("Clean Audio Mel-Spectrogram (Harmonic Detail)", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Time (seconds)", fontsize=9)
    ax1.set_ylabel("Frequency (Hz)", fontsize=9)
    fig.colorbar(im1, ax=ax1, format="%+2.0f dB")

    im2 = ax2.imshow(S_noisy, aspect="auto", origin="lower", cmap="magma", extent=[0, 5, 0, sr/2])
    ax2.set_title("0 dB Noisy Mel-Spectrogram (Frequency Masking & Noise Floor)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Time (seconds)", fontsize=9)
    ax2.set_ylabel("Frequency (Hz)", fontsize=9)
    fig.colorbar(im2, ax=ax2, format="%+2.0f dB")

    plt.tight_layout()
    fig4_path = os.path.join("results", "spectrogram_clean_vs_noisy.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print("Saved:", fig4_path)
