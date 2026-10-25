\# Noise-Robust Hybrid Music Recommendation System





A music recommender that combines collaborative filtering (who listened to what), audio signal-processing features (how songs sound) and deep learning, and measures how background noise hurts it. The main question is how well new songs, which have no listening history, can be recommended from noisy audio.



\## Main findings



\- Plain audio features are fragile: even mild noise cuts accuracy sharply, and at 0 dB it is at chance level.

\- Most of the information survives and is shifted, so a model can learn to undo the damage.

\- A small neural network that maps noisy features to clean ones recovers most of the lost accuracy on new songs.

\- A CNN denoising autoencoder trained with a consistency loss matches that on noise it has seen, but does worse on noise it has not seen.

\- In the hybrid, collaborative filtering ranks songs with listening history and corrected audio ranks new songs.



Headline results (Recall@10 on 200 new songs per split, average of 5 splits, 999 songs; random guessing is about 0.05):



| Condition | Handcrafted raw | Handcrafted + fix | CNN |

|---|---|---|---|

| Clean | 0.120 | 0.120 | 0.127 |

| White noise 0 dB | 0.091 | 0.114 | 0.115 |

| Pink noise 0 dB, seen in training | 0.109 | 0.124 | 0.120 |

| Pink noise 0 dB, never seen in training | 0.109 | 0.117 | 0.102 |



Listening data is simulated (see Limitations), so these results show that the method works, not real-world accuracy.



\## Data



\- Audio: GTZAN (1,000 clips, 10 genres). Not included in this repository. Download it from Kaggle ("GTZAN Dataset - Music Genre Classification") and place the genre folders in `data/raw/genres\_original/`. One file, `jazz.00054.wav`, cannot be read, so 999 songs are used.

\- Listening data: simulated by `src/simulate\_users.py` (1,000 users with genre-based tastes).



\## Setup



```

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



\## Run order



```

python src\\build\_features.py

python src\\build\_noisy\_features.py

python src\\build\_pink\_features.py

python src\\simulate\_users.py

python src\\collab\_filtering.py

python src\\cold\_start\_full.py

python src\\generalization\_test.py

python src\\final\_slots.py

python src\\build\_mels.py

python src\\cnn\_test.py

python src\\cnn\_unseen\_noise.py

python src\\make\_figure.py

```



`build\_mels.py`, `cnn\_test.py` and `cnn\_unseen\_noise.py` take a long time on a CPU (roughly 15 minutes, 1.5 hours and 1 hour).



\## Other scripts



\- `noise\_experiment.py` and `shift\_test.py`: noise damage on plain features. First run on a 100-song subset, then rerun on all 999 songs.

\- `compare\_denoising.py` and `build\_denoised\_features.py`: a standard denoiser, tested on the 100-song subset only.

\- `learn\_to\_undo.py`, `deep\_fix\_test.py`, `multi\_split\_test.py`, `cold\_start\_test.py`, `hybrid\_test.py`: earlier small-scale versions, superseded by the scripts above.

\- `final\_hybrid.py`: a discarded experiment. Its merged ranking was dominated by the scoring rule, so its results are not used.

\- `noise\_test.py`, `robustness\_test.py`, `denoise\_test.py`, `similar\_songs.py`: learning demos.



\## Limitations



\- Simulated listeners, only 999 clips and synthetic noise (white and pink) only.

\- Small handcrafted features, a small MLP and an untuned CNN.

\- The CNN consistency loss was chosen after looking at one split's results.

\- Splits share training songs, so the reported spreads understate the real uncertainty.



\## Report



The full report, with figures and an explanation of every script, is provided separately.

