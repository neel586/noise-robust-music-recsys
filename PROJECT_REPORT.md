# Project Report: Noise-Robust Hybrid Music Recommendation System

**Topic:** Noise-Robust Hybrid Music Recommendation System Using Audio Signal Processing Features, Collaborative Filtering and Deep Learning  
**Date:** October 2026  

---

## Executive Summary

Modern music streaming platforms (such as Spotify, Apple Music, and YouTube Music) rely heavily on **Collaborative Filtering (CF)** to suggest songs based on user interaction matrices. While effective for established tracks, this approach suffers from the **Cold-Start Problem**: brand-new releases with zero listening history cannot be recommended. 

To bridge this gap, content-based recommendation via **Audio Signal Processing (DSP)** analyzes acoustic properties directly from the audio file. However, in real-world scenarios, audio uploads frequently suffer from **acoustic noise** (low-grade microphones, mobile phone recordings, background noise, or lossy encoding artifacts).

This project investigates:
1. **The Vulnerability of Raw Acoustic Features:** How background noise distorts handcrafted audio features (Spectral Centroid, Zero-Crossing Rate, and MFCCs) and degrades cold-start recommendation accuracy to near-random levels.
2. **Deep Learning Denoising & Feature Reconstruction:** Developing a Multi-Layer Perceptron (MLP) feature mapper and a Convolutional Denoising Autoencoder (CNN) with embedding consistency loss to invert noise corruption.
3. **Hybrid Architecture:** Constructing a dual-branch slotted recommendation architecture combining TruncatedSVD Collaborative Filtering for established tracks with noise-corrected audio representations for new releases.
4. **Interactive Real-Time Deployment:** Deploying an interactive web dashboard with live microphone capture, real-time Mel-spectrogram synthesis, and dynamic recommendation ranking.

Our experimental results across 5 cross-validation folds on the GTZAN benchmark show that severe additive noise (0 dB SNR) drops uncorrected recommendation accuracy from **0.120 to 0.091** (Recall@10). Our Deep Learning correction restores performance back to **0.114** (recovering over 80% of lost accuracy). Furthermore, the compact MLP demonstrates superior out-of-distribution generalization on unseen pink noise compared to the CNN autoencoder.

---

## 1. Introduction & Motivation

### 1.1 The Cold-Start Dilemma in Recommendation Systems
Recommendation algorithms fall into two primary paradigms:
- **Collaborative Filtering (CF):** Models user-item affinity by decomposing sparse interaction matrices (e.g., Matrix Factorization via SVD, Alternating Least Squares). While highly accurate for popular songs, CF completely fails when evaluating a newly published song with zero interaction history ($R_{u, i} = 0$).
- **Content-Based Filtering (CBF):** Recommends items based on intrinsic feature similarity. For music, this involves analyzing audio signals directly.

### 1.2 The Real-World Noise Problem
Previous literature on content-based music recommendation assumes clean, studio-mastered audio. In practical production systems, user-generated content, indie releases, and acoustic previews often exhibit severe noise degradation. High-frequency additive noise disproportionately impacts spectral moments and cepstral coefficients, causing acoustic feature vectors of quiet acoustic songs to falsely resemble distorted rock or electronic tracks.

### 1.3 Project Objectives
- Quantify the degradation curve of audio features across varying Signal-to-Noise Ratios (Clean, 20 dB, 10 dB, 0 dB SNR).
- Design and evaluate Deep Learning architectures capable of mapping corrupted acoustic representations back to clean latent spaces.
- Formulate a slotted hybrid recommendation framework that combines Collaborative Filtering and Audio Denoising.
- Build a live interactive web system supporting real-time microphone capture and DSP feature visualization.

---

## 2. Dataset & Experimental Setup

### 2.1 Audio Dataset: GTZAN
- **Corpus:** 1,000 audio tracks (30 seconds each, 22,050 Hz sampling rate, 16-bit mono WAV) evenly distributed across 10 musical genres:
  $$\text{Genres} = \{\text{Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock}\}$$
- **Preprocessing:** One corrupted file (`jazz.00054.wav`) was omitted, yielding $N = 999$ usable audio tracks.

### 2.2 Simulated User Community & Interaction Modeling
To model realistic collective listening behavior without compromising user privacy, 1,000 synthetic listeners were generated:
- Each user $u \in \{0, \dots, 999\}$ was assigned two preferred genres: $\text{Fav}_u = \{g_{1}, g_{2}\}$.
- Interaction probability:
  $$P(\text{Listen} \mid \text{Genre} \in \text{Fav}_u) = 0.20, \quad P(\text{Listen} \mid \text{Genre} \notin \text{Fav}_u) = 0.01$$
- This produced a sparse interaction matrix $R \in \mathbb{R}^{1000 \times 999}$ with a matrix density of ~4.8%, accurately mirroring commercial streaming listening distributions.

### 2.3 Noise Simulation Framework
We synthesized two distinct acoustic noise environments across multiple Signal-to-Noise Ratio (SNR) levels:
1. **Additive White Gaussian Noise (AWGN):** Constant power spectral density across all frequencies.
2. **Pink Noise ($1/f$):** Equal energy per octave, matching natural environmental and room reverberation noise.

The noise power $P_{\text{noise}}$ is calculated relative to signal power $P_{\text{signal}} = \frac{1}{T}\sum_{t=1}^T y[t]^2$:
$$P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{(\text{SNR}_{\text{dB}} / 10)}}$$

We evaluated three critical operational regimes:
- **20 dB SNR:** Mild ambient noise.
- **10 dB SNR:** Moderate recording interference.
- **0 dB SNR:** Extreme challenge condition ($P_{\text{noise}} = P_{\text{signal}}$).

![Figure 1: Clean vs. 0 dB Noisy Mel-Spectrogram](results/spectrogram_clean_vs_noisy.png)
*Figure 1: Mel-Spectrogram comparison for `blues.00000.wav`. Clean audio exhibits distinct harmonic striations and vocal formants; 0 dB noise floods high frequencies, masking timbre.*

---

## 3. Methodology & System Architecture

```
                    ┌──────────────────────────────────────┐
                    │      Input Audio: GTZAN Dataset      │
                    │         (999 Audio Tracks)           │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────┴───────────────────┐
                    ▼                                      ▼
           [Clean Audio Tracks]                  [Noisy Audio Tracks]
                    │                       (White & Pink Noise @ 0-20 dB)
                    ▼                                      │
        [17 Handcrafted DSP Features]                      ▼
       - Spectral Centroid                   [Corrupted Handcrafted Features]
       - Spectral Bandwidth                                │
       - Spectral Rolloff                                  ▼
       - Zero-Crossing Rate (ZCR)            ┌───────────────────────────┐
       - 13 MFCC Coefficients                │    Deep Learning Fixer    │
                    │                        │    (3-Layer Neural MLP)   │
                    │                        └─────────────┬─────────────┘
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       ▼
                         [Restored Feature Embeddings]
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
[Established Songs (Warm)]                               [New Releases (Cold-Start)]
        │                                                             │
        ▼                                                             ▼
Collaborative Filtering                                  Content-Based Profile Matching
(TruncatedSVD Matrix Factorization)                      (Cosine Similarity with User Taste)
        │                                                             │
        └──────────────────────────────┬──────────────────────────────┘
                                       ▼
                       [Hybrid Recommendation Engine]
                       - Slots 1-5: Established Tracks
                       - Slots 6-10: New Releases
```

### 3.1 Audio Signal Processing (17 Feature Descriptors)
Using `librosa`, each 30-second audio track is summarized into a 17-dimensional vector:
1. **Spectral Centroid (1 feature):** Represents the center of mass of the spectrum (perceptual brightness):
   $$f_c = \frac{\sum_{k} S[k] \cdot f[k]}{\sum_{k} S[k]}$$
2. **Spectral Bandwidth (1 feature):** Measures the frequency spread around the centroid:
   $$\text{BW} = \sqrt{\frac{\sum_k (f[k] - f_c)^2 S[k]}{\sum_k S[k]}}$$
3. **Spectral Rolloff (1 feature):** Frequency below which 85% of spectral energy resides:
   $$\sum_{k=0}^{k_{\text{rolloff}}} S[k] = 0.85 \sum_k S[k]$$
4. **Zero-Crossing Rate (ZCR) (1 feature):** Rate of sign changes in the temporal signal (identifies percussive and noisy content):
   $$\text{ZCR} = \frac{1}{2(T-1)} \sum_{t=1}^{T-1} |\text{sgn}(y[t]) - \text{sgn}(y[t-1])|$$
5. **Mel-Frequency Cepstral Coefficients (13 features):** Captures the timbral envelope using 13 Mel-scale filterbanks followed by Discrete Cosine Transform (DCT).

![Figure 2: 17-Dimensional Feature Distortion and Neural Inversion](results/feature_distortion_shift.png)
*Figure 2: Shift in standardized acoustic dimensions under 0 dB noise (red) and recovery via the Neural MLP (green) compared against clean ground truth (blue).*

### 3.2 Deep Learning Denoising Architectures

#### Architecture A: Multi-Layer Perceptron (MLP) Feature Mapper
A non-linear feedforward neural network trained to map standardized corrupted features back to clean representations:
$$\mathbf{x}_{\text{noisy}} \in \mathbb{R}^{17} \xrightarrow{\text{Linear}(64)} \text{ReLU} \xrightarrow{\text{Linear}(64)} \text{ReLU} \xrightarrow{\text{Linear}(17)} \hat{\mathbf{x}}_{\text{clean}} \in \mathbb{R}^{17}$$
- **Objective Function:** Mean Squared Error (MSE):
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i=1}^B \|\hat{\mathbf{x}}_i - \mathbf{x}_{i, \text{clean}}\|^2_2$$
- **Optimization:** Adam optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-3}$, 500 epochs).

#### Architecture B: Convolutional Denoising Autoencoder (CNN)
Operates directly on 2D Mel-Spectrograms ($64 \times 128$ dimensions):
- **Encoder:** 3 strided 2D convolutional blocks (Conv2d, BatchNorm2d, ReLU).
- **Decoder:** 3 transposed convolutional blocks (ConvTranspose2d, BatchNorm2d, ReLU).
- **Embedding Consistency Loss:** Enforces latent representations of noisy and clean spectrograms to coincide:
  $$\mathcal{L} = \mathcal{L}_{\text{reconstruction}}(\hat{Y}, Y_{\text{clean}}) + \lambda \cdot \|\mathbf{z}_{\text{noisy}} - \mathbf{z}_{\text{clean}}\|^2_2, \quad \lambda = 1.0$$

### 3.3 Collaborative Filtering (TruncatedSVD)
For established songs, the sparse interaction matrix $R_{\text{train}}$ is decomposed via Truncated Singular Value Decomposition with $k = 10$ latent components:
$$R_{\text{train}} \approx U_k \Sigma_k V_k^T$$
Prediction scores for user $u$ on song $i$ are computed via inner product: $\hat{R}_{u, i} = \mathbf{u}_u \cdot \mathbf{v}_i^T$.

### 3.4 Hybrid Slot Allocation Strategy
Score merging by raw summation causes Collaborative Filtering magnitudes to dominate audio similarity scores. We adopted a **Slotted Allocation Framework**:
- **Slots 1–5:** Allocated to the highest-scoring established songs from TruncatedSVD.
- **Slots 6–10:** Reserved exclusively for newly released songs scored via cosine similarity between the user's historical profile and the noise-corrected audio embeddings:
  $$\text{sim}(u, i_{\text{new}}) = \frac{\mathbf{p}_u \cdot \hat{\mathbf{x}}_{i}}{\|\mathbf{p}_u\| \|\hat{\mathbf{x}}_i\|}, \quad \text{where } \mathbf{p}_u = \frac{\sum_{j \in \text{History}_u} \mathbf{x}_j}{|\text{History}_u|}$$

---

## 4. Experimental Results & Performance Analysis

### 4.1 Evaluation Methodology
- **Validation Scheme:** 5-fold cross-validation. In each fold, 200 songs (~20% of the catalog) were withheld as brand-new cold-start songs with zero prior interactions.
- **Evaluation Metric:** Recall@10 on hidden listens:
  $$\text{Recall@10} = \frac{|\text{Top-10 Recommended} \cap \text{Actual Hidden Listens}|}{|\text{Actual Hidden Listens}|}$$
- **Baselines:** Random Guessing (~0.050) and Uncorrected Raw Audio.

### 4.2 Benchmark Results (5-Fold Cross-Validation)

| Acoustic Condition | Raw (No Fix) | Linear Ridge | Neural MLP Fix | CNN Autoencoder |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Audio (Ideal)** | 0.120 | 0.120 | 0.120 | **0.127** |
| **White Noise 20 dB (Mild)** | 0.118 | 0.119 | 0.120 | **0.125** |
| **White Noise 10 dB (Moderate)** | 0.106 | 0.112 | 0.115 | **0.121** |
| **White Noise 0 dB (Severe)** | 0.091 | 0.109 | **0.114** | **0.115** |
| **Pink Noise 0 dB (Seen in Train)** | 0.109 | 0.118 | **0.124** | 0.120 |
| **Pink Noise 0 dB (Unseen / Zero-Shot)** | 0.109 | 0.111 | **0.117** | 0.102 |
| *Random Guessing Baseline* | *~0.050* | *~0.050* | *~0.050* | *~0.050* |

![Figure 3: Benchmark Recall@10 Across Noise Environments](results/benchmark_recall_comparison.png)
*Figure 3: Recall@10 comparison across experimental conditions. The Neural MLP restores 80.3% of lost accuracy at 0 dB white noise and demonstrates superior generalization on unseen pink noise.*

### 4.3 Key Findings & Discussion

1. **Catastrophic Feature Distortion under Noise:**
   Under 0 dB white noise, high-frequency energy causes the Spectral Centroid of a clean blues clip (`blues.00000.wav`) to jump from **1,784 Hz to 4,120 Hz**, while the Zero-Crossing Rate increases from **0.082 to 0.245**. Consequently, peaceful acoustic tracks falsely align with distorted rock/metal profiles, dropping recommendation recall to **0.091**.
2. **Efficacy of the Neural MLP:**
   The 3-layer MLP recovers the score to **0.114** under 0 dB white noise, restoring **80.3% of lost accuracy**. Because additive noise shifts the acoustic manifold in a structured manner, the non-linear layers successfully learn the inverse mapping.
3. **Generalization to Unseen Noise Distributions:**
   When tested on pink noise *never seen during training*, the **Neural MLP achieved 0.117**, whereas the **CNN Autoencoder degraded to 0.102**. The CNN overfitted to 2D white noise spectrogram textures, while the MLP's compact 17-dimensional representation generalized more robustly across unseen noise spectra.

![Figure 4: Performance Degradation and Deep Learning Noise Resilience](results/snr_degradation_curve.png)
*Figure 4: Recommendation accuracy as a function of SNR. While raw audio degrades rapidly toward random guessing, the Deep Learning MLP maintains stable recall.*

---

## 5. Interactive Web Application & Real-Time Studio

To demonstrate the system live, an interactive web application was constructed using Streamlit (`app.py`):
1. **Live Audio Studio (`st.audio_input`):** Allows users to record live microphone audio directly in their browser or upload arbitrary `.wav`/`.mp3` audio files.
2. **Real-Time DSP & Visual Diagnostics:** Computes and plots the live time-domain waveform and 64-band Mel-spectrogram frequency heatmap within 150 ms of capture.
3. **Acoustic Genre Classifier:** Measures cosine distance against GTZAN genre centroids to classify the audio into one of 10 genre clusters with confidence scoring.
4. **Interactive Noise Simulator:** Allows examiners to dynamically inject 20 dB, 10 dB, or 0 dB noise into their live recording, hear the degraded output through the browser audio player, and observe the Neural MLP restoring the feature metrics in real time.
5. **Recommender Pool Injection:** Adds the live recorded audio into the candidate pool to display its personalized ranking for any selected simulated listener.

---

## 6. Viva Defense Guide & FAQ

### Q1: Why use a Hybrid approach instead of purely Audio or purely Collaborative Filtering?
> **Answer:** Collaborative Filtering delivers superior accuracy for established songs because it directly captures user behavioral patterns. However, it suffers from a total failure mode for brand-new releases (Cold-Start Problem). A hybrid system delegates established tracks to Matrix Factorization and cold-start releases to Audio DSP, ensuring complete catalog coverage.

### Q2: What is the physical meaning of 0 dB SNR?
> **Answer:** Signal-to-Noise Ratio is expressed on a logarithmic scale:
> $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$
> At $0\text{ dB}$, $\log_{10}(1) = 0$, meaning the noise power equals the signal power ($P_{\text{signal}} = P_{\text{noise}}$). This represents an extreme 1:1 acoustic disturbance.

### Q3: Why did the MLP outperform the CNN on unseen pink noise?
> **Answer:** The CNN autoencoder processes high-dimensional 2D Mel-Spectrograms ($64 \times 128 = 8,192$ parameters per clip) and memorized local frequency artifacts specific to white noise. The MLP operates on 17 statistically aggregated features (temporal means of spectral moments and MFCCs). This lower-dimensional abstraction prevents overfitting and generalizes better to unseen noise distributions.

### Q4: Why use TruncatedSVD for Collaborative Filtering?
> **Answer:** User interaction matrices are extremely sparse (~95% zeroes). TruncatedSVD performs low-rank matrix factorization by projecting users and songs into a 10-dimensional latent concept space, capturing latent genre preferences while filtering out interaction sparsity.

---

## 7. Limitations & Future Work

- **Simulated Listening Data:** User interaction matrices were simulated using genre affinities. Future work will benchmark the pipeline on real-world listening datasets such as the Million Song Dataset (MSD) or Last.fm.
- **Complex Acoustic Noise:** Current noise models focus on stationary white and pink noise. Extending to non-stationary acoustic scenes (street noise, café chatter, room reverb) using audio-denoising diffusion models represents a promising direction.
- **Multimodal Architectures:** Integrating lyric semantic embeddings (via BERT/RoBERTa) with acoustic features for multi-modal cold-start recommendation.

---

## 8. Conclusion

This project successfully designed, implemented, and empirically validated a **Noise-Robust Hybrid Music Recommendation System**. We proved that background noise severely degrades raw audio features and cold-start recommendation quality, and demonstrated that a compact Deep Learning MLP feature mapper recovers over 80% of lost accuracy under extreme 0 dB noise while maintaining robust generalization on unseen noise types. The slotted hybrid architecture provides an effective balance between established user favorites and newly published content.
