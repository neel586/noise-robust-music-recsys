# Noise-Robust Hybrid Music Recommendation System

A music recommender that combines collaborative filtering (who listened to what), audio signal-processing features (how songs sound) and deep learning, and measures how background noise hurts it. The main question is how well new songs, which have no listening history, can be recommended from noisy audio.

## Main findings

- Plain audio features are fragile: even mild noise cuts accuracy sharply, and at 0 dB it is at chance level.
- Most of the information survives and is shifted, so a model can learn to undo the damage.
- A small neural network that maps noisy features to clean ones recovers most of the lost accuracy on new songs.
- A CNN denoising autoencoder trained with a consistency loss matches that on noise it has seen, but does worse on noise it has not seen.
- In the hybrid, collaborative filtering ranks songs with listening history and corrected audio ranks new songs.

Headline results (Recall@10 on 200 new songs per split, average of 5 splits, 999 songs; random guessing is about 0.05):

| Condition | Handcrafted raw | Handcrafted + fix | CNN |
|---|---|---|---|
| Clean | 0.120 | 0.120 | 0.127 |
| White noise 0 dB | 0.091 | 0.114 | 0.115 |
| Pink noise 0 dB, seen in training | 0.109 | 0.124 | 0.120 |
| Pink noise 0 dB, never seen in training | 0.109 | 0.117 | 0.102 |

Listening data is simulated (see Limitations), so these results show that the method works, not real-world accuracy.

## Data

- Audio: GTZAN (1,000 clips, 10 genres). Not included in this repository. Download it from Kaggle ("GTZAN Dataset - Music Genre Classification") and place the genre folders in `data/raw/genres_original/`. One file, `jazz.00054.wav`, cannot be read, so 999 songs are used.
- Listening data: simulated by `src/simulate_users.py` (1,000 users with genre-based tastes).

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Interactive Web Dashboard

Launch the interactive Streamlit dashboard for real-time demonstration, viva, and presentations:

```bash
streamlit run app.py
```
Or double-click `run_app.bat`.

### Dashboard Features:
- **🔀 Live Recommendations**: Real-time side-by-side comparison (Clean vs Corrupted Noisy Audio vs Neural MLP Restored) with match badges and TruncatedSVD Collaborative Filtering slots.
- **🎙️ Live Audio Studio & Real-Time Analysis**:
  - Direct in-browser microphone audio recording (`st.audio_input`).
  - Custom audio file upload (.wav, .mp3, .ogg, .flac).
  - Real-time 17 Audio Signal Processing (DSP) feature extraction (Spectral Centroid, Bandwidth, Rolloff, ZCR, 13 MFCCs).
  - Dual live visualizations: Time-domain waveform & Mel-spectrogram frequency energy heatmap.
  - Live musical genre classification & acoustic similarity matching with GTZAN catalog.
  - Live personalization compatibility scoring with simulated user profiles.
  - One-click injection of user-added tracks into the recommendation candidate pool.
- **🔬 Feature Distortion Inspector**: Deep-dive into acoustic feature shifts under 20 dB, 10 dB, and 0 dB noise.
- **📊 Benchmarks & Metric Results**: Comprehensive 5-fold cross-validation tables and performance bar charts.
- **🎓 Viva & Presentation Guide**: Built-in interactive defense cheat sheet answering common technical viva questions.

## Run Order (Pipeline Reproduction)

```bash
python src\build_features.py
python src\build_noisy_features.py
python src\build_pink_features.py
python src\simulate_users.py
python src\collab_filtering.py
python src\cold_start_full.py
python src\generalization_test.py
python src\final_slots.py
python src\build_mels.py
python src\cnn_test.py
python src\cnn_unseen_noise.py
python src\make_figure.py
```

`build_mels.py`, `cnn_test.py` and `cnn_unseen_noise.py` take a long time on a CPU (roughly 15 minutes, 1.5 hours and 1 hour).

## Other scripts

- `demo.py`: Command-line interactive demonstration script.
- `noise_experiment.py` and `shift_test.py`: noise damage on plain features. First run on a 100-song subset, then rerun on all 999 songs.
- `compare_denoising.py` and `build_denoised_features.py`: a standard denoiser, tested on the 100-song subset only.
- `learn_to_undo.py`, `deep_fix_test.py`, `multi_split_test.py`, `cold_start_test.py`, `hybrid_test.py`: earlier small-scale versions, superseded by the scripts above.
- `final_hybrid.py`: a discarded experiment. Its merged ranking was dominated by the scoring rule, so its results are not used.
- `noise_test.py`, `robustness_test.py`, `denoise_test.py`, `similar_songs.py`: learning demos.

## Limitations

- Simulated listeners, only 999 clips and synthetic noise (white and pink) only.
- Small handcrafted features, a small MLP and an untuned CNN.
- The CNN consistency loss was chosen after looking at one split's results.
- Splits share training songs, so the reported spreads understate the real uncertainty.
