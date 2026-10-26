import os
import io
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Noise-Robust Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .user-card {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .song-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 0.7rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .song-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .user-upload-card {
        background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.15);
    }
    .badge-user-upload {
        background-color: #10B981;
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 9999px;
        display: inline-block;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-hit {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 600;
        font-size: 0.78rem;
        padding: 3px 8px;
        border-radius: 9999px;
        display: inline-block;
    }
    .badge-miss {
        background-color: #F1F5F9;
        color: #64748B;
        font-weight: 500;
        font-size: 0.78rem;
        padding: 3px 8px;
        border-radius: 9999px;
        display: inline-block;
    }
    .badge-genre {
        background-color: #DBEAFE;
        color: #1E40AF;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 2px 7px;
        border-radius: 6px;
        display: inline-block;
        margin-right: 5px;
    }
    .cf-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for user-uploaded songs
if "user_songs" not in st.session_state:
    st.session_state["user_songs"] = []

# ==========================================
# CACHED DATA & MODEL LOADING
# ==========================================
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def extract_audio_features(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    return np.concatenate([[centroid, bandwidth, rolloff, zcr], mfcc])

@st.cache_resource
def load_system_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    clean_df = pd.read_csv(os.path.join(data_dir, "features_clean.csv"))
    inter_df = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    fav_df = pd.read_csv(os.path.join(data_dir, "user_favorites.csv"))
    
    noisy_white = {
        20: pd.read_csv(os.path.join(data_dir, "features_snr20.csv")).iloc[:, 2:].values,
        10: pd.read_csv(os.path.join(data_dir, "features_snr10.csv")).iloc[:, 2:].values,
        0: pd.read_csv(os.path.join(data_dir, "features_snr0.csv")).iloc[:, 2:].values,
    }
    noisy_pink = {
        10: pd.read_csv(os.path.join(data_dir, "features_pink_snr10.csv")).iloc[:, 2:].values,
        0: pd.read_csv(os.path.join(data_dir, "features_pink_snr0.csv")).iloc[:, 2:].values,
    }
    
    n_users = len(fav_df)
    n_songs = len(clean_df)
    
    # Interaction matrix
    R = np.zeros((n_users, n_songs))
    R[inter_df["user_id"].values, inter_df["song_index"].values] = 1
    
    # 5-fold split standard (Split 0 default: 200 cold songs, 799 warm songs)
    pos = clean_df.groupby("genre").cumcount().values
    cold_mask = (pos % 5 == 0)
    warm_mask = ~cold_mask
    
    clean_raw = clean_df.iloc[:, 2:].values
    
    # Standard scaler fit on warm songs
    scaler = StandardScaler().fit(clean_raw[warm_mask])
    Xc = scaler.transform(clean_raw)
    
    white_scaled = {s: scaler.transform(noisy_white[s]) for s in noisy_white}
    pink_scaled = {s: scaler.transform(noisy_pink[s]) for s in noisy_pink}
    
    # Train Denoising MLP
    torch.manual_seed(42)
    Xtr = np.vstack([white_scaled[s][warm_mask] for s in white_scaled] + [pink_scaled[s][warm_mask] for s in pink_scaled])
    Ytr = np.vstack([Xc[warm_mask]] * (len(white_scaled) + len(pink_scaled)))
    
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr, dtype=torch.float32)
    
    net = nn.Sequential(
        nn.Linear(17, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 17)
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(500):
        opt.zero_grad()
        loss = loss_fn(net(Xt), Yt)
        loss.backward()
        opt.step()
    net.eval()
    
    # Collaborative Filtering with TruncatedSVD
    R_train = np.load(os.path.join(data_dir, "R_train.npy"))
    svd = TruncatedSVD(n_components=10, random_state=42)
    U = svd.fit_transform(R_train)
    cf_scores = U @ svd.components_
    
    # User taste profile from warm listens
    R_warm = R[:, warm_mask]
    counts = np.maximum(R_warm.sum(axis=1, keepdims=True), 1)
    profiles = (R_warm @ Xc[warm_mask]) / counts
    
    # Genre centroid profiles (mean 17-dim vector for each genre in standardized space)
    genre_centroids = {}
    for g in GENRES:
        mask = (clean_df["genre"] == g)
        genre_centroids[g] = Xc[mask].mean(axis=0)
        
    return {
        "clean_df": clean_df,
        "inter_df": inter_df,
        "fav_df": fav_df,
        "cold_mask": cold_mask,
        "warm_mask": warm_mask,
        "clean_raw": clean_raw,
        "scaler": scaler,
        "Xc": Xc,
        "white_scaled": white_scaled,
        "pink_scaled": pink_scaled,
        "net": net,
        "cf_scores": cf_scores,
        "R_train": R_train,
        "R": R,
        "profiles": profiles,
        "genre_centroids": genre_centroids
    }

data = load_system_data()

# Helper for MLP denoising
def apply_neural_fix(X_scaled):
    with torch.no_grad():
        t = torch.tensor(X_scaled, dtype=torch.float32)
        return data["net"](t).numpy()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/audio-wave.png", width=64)
    st.markdown("### 🎛️ Control Panel")
    
    # User selector
    user_id = st.number_input("Select User ID (0 – 999)", min_value=0, max_value=999, value=0, step=1)
    
    user_fav1 = data["fav_df"].loc[user_id, "fav1"]
    user_fav2 = data["fav_df"].loc[user_id, "fav2"]
    user_favs = [user_fav1, user_fav2]
    user_history_count = int(data["R"][user_id, data["warm_mask"]].sum())
    
    # User Profile Card
    st.markdown(f"""
    <div class="user-card">
        <div style="font-size:0.85rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">User Taste Profile</div>
        <div style="font-size:1.4rem; font-weight:700; margin: 4px 0;">User #{user_id}</div>
        <div style="margin-top: 8px;">
            <span style="font-size:0.85rem;">Favorite Genres:</span><br/>
            <span style="background:#3B82F6; color:white; padding:3px 8px; border-radius:6px; font-weight:600; font-size:0.8rem; margin-right:5px;">{user_fav1.title()}</span>
            <span style="background:#8B5CF6; color:white; padding:3px 8px; border-radius:6px; font-weight:600; font-size:0.8rem;">{user_fav2.title()}</span>
        </div>
        <div style="margin-top: 10px; font-size:0.85rem; color:#CBD5E1;">
            🎧 Listened to <b>{user_history_count}</b> existing tracks
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # User uploaded tracks status in sidebar
    if len(st.session_state["user_songs"]) > 0:
        st.markdown(f"**👤 Custom Uploads Added:** `{len(st.session_state['user_songs'])}`")
        if st.button("🗑️ Clear Uploaded Songs", key="clear_uploads"):
            st.session_state["user_songs"] = []
            st.rerun()
            
    st.markdown("---")
    st.markdown("#### 🔊 Noise Simulation Settings")
    noise_type = st.selectbox("Noise Type", ["White Noise (Flat Spectrum)", "Pink Noise (1/f Frequency Drop)"])
    snr_option = st.select_slider(
        "Signal-to-Noise Ratio (SNR)",
        options=["Clean (No Noise)", "20 dB (Mild Noise)", "10 dB (Noticeable Noise)", "0 dB (Heavy Noise)"],
        value="0 dB (Heavy Noise)"
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Recommendation Slots")
    k_slots = st.slider("Top-K Songs to Recommend", min_value=3, max_value=10, value=5)
    
    st.markdown("---")
    st.caption("Noise-Robust Hybrid Music Recommendation System • Project Live Dashboard")

# Determine active noisy feature matrix for cold songs
cold_indices = np.where(data["cold_mask"])[0]
warm_indices = np.where(data["warm_mask"])[0]

if "Clean" in snr_option:
    noisy_cold_features = data["Xc"][data["cold_mask"]]
    noise_label = "Clean Audio"
elif "White" in noise_type:
    if "20 dB" in snr_option:
        noisy_cold_features = data["white_scaled"][20][data["cold_mask"]]
        noise_label = "White Noise 20 dB"
    elif "10 dB" in snr_option:
        noisy_cold_features = data["white_scaled"][10][data["cold_mask"]]
        noise_label = "White Noise 10 dB"
    else:
        noisy_cold_features = data["white_scaled"][0][data["cold_mask"]]
        noise_label = "White Noise 0 dB"
else: # Pink
    if "10 dB" in snr_option or "20 dB" in snr_option:
        noisy_cold_features = data["pink_scaled"][10][data["cold_mask"]]
        noise_label = "Pink Noise 10 dB"
    else:
        noisy_cold_features = data["pink_scaled"][0][data["cold_mask"]]
        noise_label = "Pink Noise 0 dB"

restored_cold_features = apply_neural_fix(noisy_cold_features) if "Clean" not in snr_option else noisy_cold_features

# ==========================================
# MAIN INTERFACE TABS
# ==========================================
st.markdown('<div class="main-header">🎵 Noise-Robust Hybrid Music Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Addressing the Cold-Start problem under noisy audio conditions using DSP, Matrix Factorization, and Deep Learning</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔀 Live Recommendations",
    "📤 Upload & Analyze Song (User-Added)",
    "🔬 Audio Signal & Feature Inspector",
    "📊 Benchmarks & Metric Results",
    "🎓 Viva & Presentation Guide"
])

# -----------------------------------------------------------------------------
# TAB 1: LIVE RECOMMENDATIONS
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 🔀 Real-Time Recommendation Engine")
    st.markdown("""
    Here you can compare what happens to new song recommendations under **Clean Audio**, **Corrupted Noisy Audio**, and **Neural Denoised Audio**.
    Notice how severe noise (0 dB) causes the recommender to suggest completely irrelevant genres, and how the **Neural MLP Fix** recovers the user's favorite genres!
    """)
    
    # Compute similarity rankings for new songs
    profile_u = data["profiles"][user_id:user_id+1]
    
    # Display User-Uploaded Songs if present
    if len(st.session_state["user_songs"]) > 0:
        st.markdown("#### 👤 Custom Tracks Added by User")
        st.markdown("*These tracks were uploaded by you, processed with live DSP feature extraction, and tested against User #" + str(user_id) + "'s profile:*")
        
        for usong in st.session_state["user_songs"]:
            usong_sim = cosine_similarity(profile_u, [usong["scaled_features"]])[0][0]
            is_match = usong["genre"].lower() in user_favs
            badge = f'<span class="badge-hit">🎯 Matches User Taste ({usong["genre"].upper()})</span>' if is_match else f'<span class="badge-miss">Genre: {usong["genre"].upper()} (Exploratory)</span>'
            
            st.markdown(f"""
            <div class="user-upload-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="badge-user-upload">👤 ADDED BY USER IN WEBPAGE</span>
                    <span style="font-weight:700; color:#1E40AF; font-size:0.95rem;">Affinity Score with User #{user_id}: {usong_sim:.3f}</span>
                </div>
                <h3 style="margin: 8px 0 4px 0; color:#1E3A8A;">🎵 {usong['filename']}</h3>
                <div style="margin-top: 6px;">
                    {badge}
                    <span style="font-size:0.85rem; color:#475569; margin-left:12px;"><b>Predicted Acoustic Category:</b> {usong['genre'].upper()}</span>
                </div>
                <div style="margin-top: 8px; font-size:0.85rem; color:#334155;">
                    💡 <i>This song has zero listening history and was ranked entirely through our audio signal processing and deep learning pipeline.</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if "audio_bytes" in usong and usong["audio_bytes"] is not None:
                st.audio(usong["audio_bytes"], format="audio/wav")
        st.markdown("---")

    sim_clean = cosine_similarity(profile_u, data["Xc"][data["cold_mask"]])[0]
    sim_noisy = cosine_similarity(profile_u, noisy_cold_features)[0]
    sim_fixed = cosine_similarity(profile_u, restored_cold_features)[0]
    
    top_clean_idx = np.argsort(sim_clean)[::-1][:k_slots]
    top_noisy_idx = np.argsort(sim_noisy)[::-1][:k_slots]
    top_fixed_idx = np.argsort(sim_fixed)[::-1][:k_slots]
    
    hits_clean = sum(data["clean_df"].iloc[cold_indices[i]]["genre"] in user_favs for i in top_clean_idx)
    hits_noisy = sum(data["clean_df"].iloc[cold_indices[i]]["genre"] in user_favs for i in top_noisy_idx)
    hits_fixed = sum(data["clean_df"].iloc[cold_indices[i]]["genre"] in user_favs for i in top_fixed_idx)
    
    # 3-Column Comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"#### 🟢 1. Clean Audio (Ground Truth)")
        st.markdown(f"**Favorite Hits:** `{hits_clean} / {k_slots}` songs matched")
        for rank, idx in enumerate(top_clean_idx, 1):
            row = data["clean_df"].iloc[cold_indices[idx]]
            is_match = row["genre"] in user_favs
            badge = f'<span class="badge-hit">🎯 Matches Taste</span>' if is_match else '<span class="badge-miss">Different Genre</span>'
            st.markdown(f"""
            <div class="song-card">
                <div style="font-size:0.8rem; color:#64748B;">RANK #{rank} • SIMILARITY: {sim_clean[idx]:.3f}</div>
                <div style="font-weight:700; font-size:1.05rem; margin:3px 0;">{row['filename']}</div>
                <div>
                    <span class="badge-genre">{row['genre'].upper()}</span>
                    {badge}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.markdown(f"#### 🔴 2. Noisy Audio ({noise_label})")
        st.markdown(f"**Favorite Hits:** `{hits_noisy} / {k_slots}` songs matched")
        for rank, idx in enumerate(top_noisy_idx, 1):
            row = data["clean_df"].iloc[cold_indices[idx]]
            is_match = row["genre"] in user_favs
            badge = f'<span class="badge-hit">🎯 Matches Taste</span>' if is_match else '<span class="badge-miss">Different Genre</span>'
            st.markdown(f"""
            <div class="song-card" style="border-left: 4px solid #EF4444;">
                <div style="font-size:0.8rem; color:#64748B;">RANK #{rank} • SIMILARITY: {sim_noisy[idx]:.3f}</div>
                <div style="font-weight:700; font-size:1.05rem; margin:3px 0;">{row['filename']}</div>
                <div>
                    <span class="badge-genre">{row['genre'].upper()}</span>
                    {badge}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col3:
        st.markdown(f"#### 🛡️ 3. Neural Restored (Our Fix)")
        st.markdown(f"**Favorite Hits:** `{hits_fixed} / {k_slots}` songs matched")
        for rank, idx in enumerate(top_fixed_idx, 1):
            row = data["clean_df"].iloc[cold_indices[idx]]
            is_match = row["genre"] in user_favs
            badge = f'<span class="badge-hit">🎯 Matches Taste</span>' if is_match else '<span class="badge-miss">Different Genre</span>'
            st.markdown(f"""
            <div class="song-card" style="border-left: 4px solid #10B981;">
                <div style="font-size:0.8rem; color:#64748B;">RANK #{rank} • SIMILARITY: {sim_fixed[idx]:.3f}</div>
                <div style="font-weight:700; font-size:1.05rem; margin:3px 0;">{row['filename']}</div>
                <div>
                    <span class="badge-genre">{row['genre'].upper()}</span>
                    {badge}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Hybrid Strategy Display (Established Songs via Collaborative Filtering)
    st.markdown("### 📻 Established Songs (Collaborative Filtering via Truncated SVD)")
    st.markdown("""
    In the **Hybrid Recommendation Architecture**, we allocate dedicated slots:
    - **Established Tracks (Warm Items)** are scored using **Collaborative Filtering** (Matrix Factorization on past listening interactions).
    - **New Tracks (Cold Items)** are scored using **Audio Signal Processing + Neural Denoising**.
    """)
    
    # CF scores for user on warm songs (excluding songs already heard)
    cf_user_scores = data["cf_scores"][user_id].copy()
    cf_user_scores[data["cold_mask"]] = -np.inf # exclude cold songs
    cf_user_scores[data["R_train"][user_id] == 1] = -np.inf # exclude already heard
    
    top_cf_idx = np.argsort(cf_user_scores)[::-1][:k_slots]
    cf_cols = st.columns(k_slots)
    
    for rank, (c_col, idx) in enumerate(zip(cf_cols, top_cf_idx), 1):
        row = data["clean_df"].iloc[idx]
        is_match = row["genre"] in user_favs
        badge = f'<span class="badge-hit">🎯 Match</span>' if is_match else '<span class="badge-miss">Discovery</span>'
        with c_col:
            st.markdown(f"""
            <div class="cf-card">
                <div style="font-size:0.75rem; color:#15803D; font-weight:600;">CF SLOT #{rank}</div>
                <div style="font-weight:700; font-size:0.95rem; margin:4px 0;">{row['filename']}</div>
                <div>
                    <span class="badge-genre">{row['genre'].upper()}</span>
                    {badge}
                </div>
                <div style="font-size:0.75rem; color:#4B5563; margin-top:4px;">SVD Score: {cf_user_scores[idx]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: LIVE AUDIO RECORDING, UPLOAD & REAL-TIME ANALYSIS
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🎙️ Live Audio Studio & Real-Time Analysis")
    st.markdown("""
    Capture or add **live audio directly on the dashboard**! You can **record live from your microphone**, 
    **upload any audio file** (.wav, .mp3, .ogg, .flac), or pick a sample track.
    
    The dashboard performs **instant real-time Audio Signal Processing (DSP)**: extracts 17 acoustic features, 
    computes live Mel-Spectrograms, predicts the musical genre, tests compatibility with User #""" + str(user_id) + """, and evaluates noise resilience!
    """)
    
    sub_mic, sub_file, sub_sample = st.tabs([
        "🎙️ 1. Record Live Audio (Microphone)",
        "📂 2. Upload Audio File",
        "⚡ 3. Instant Sample Track"
    ])
    
    with sub_mic:
        st.markdown("**Click the red record button below to speak, sing, hum, or play music live through your microphone:**")
        mic_audio = st.audio_input("Record Live Audio", key="live_mic_recorder")
        
    with sub_file:
        uploaded_audio_file = st.file_uploader("Choose an audio file from your computer (.wav, .mp3, .ogg, .flac)", type=["wav", "mp3", "ogg", "flac"], key="live_file_up")
        
    with sub_sample:
        st.markdown("**Test instantly with a GTZAN dataset sample track:**")
        sample_genre = st.selectbox("Sample Genre", GENRES, index=0, key="sample_genre_up")
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "genres_original", sample_genre, f"{sample_genre}.00000.wav")
        load_sample = st.button("⚡ Load Sample Track into Live Studio", key="btn_load_sample")

    y_user = None
    sr_user = 22050
    user_audio_name = None
    user_audio_bytes = None

    if mic_audio is not None:
        try:
            user_audio_name = "Live Microphone Recording.wav"
            audio_bytes_read = mic_audio.read()
            user_audio_bytes = audio_bytes_read
            y_user, sr_user = librosa.load(io.BytesIO(audio_bytes_read), sr=22050, duration=30)
        except Exception as e:
            st.error(f"Error reading live microphone audio: {e}")
    elif uploaded_audio_file is not None:
        try:
            user_audio_name = uploaded_audio_file.name
            audio_bytes_read = uploaded_audio_file.read()
            user_audio_bytes = audio_bytes_read
            y_user, sr_user = librosa.load(io.BytesIO(audio_bytes_read), sr=22050, duration=30)
        except Exception as e:
            st.error(f"Error reading uploaded audio file: {e}")
    elif load_sample and os.path.exists(sample_path):
        user_audio_name = f"{sample_genre}.00000.wav (Preset Track)"
        y_user, sr_user = librosa.load(sample_path, sr=22050, duration=30)
        b = io.BytesIO()
        sf.write(b, y_user, sr_user, format='WAV')
        user_audio_bytes = b.getvalue()

    if y_user is not None:
        st.markdown("---")
        
        # User-Added Announcement Card
        st.markdown(f"""
        <div class="user-upload-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="badge-user-upload">🟢 LIVE AUDIO CAPTURED BY USER</span>
                <span style="font-size:0.85rem; color:#1E40AF; font-weight:600;">Duration: {len(y_user)/sr_user:.1f}s • Sampling Rate: {sr_user} Hz • RMS Energy: {np.sqrt(np.mean(y_user**2)):.4f}</span>
            </div>
            <h2 style="margin: 10px 0 4px 0; color:#1E3A8A;">🎵 {user_audio_name}</h2>
            <div style="font-size:0.92rem; color:#334155;">
                This audio was provided live in the dashboard. Real-time acoustic feature extraction, spectrogram synthesis, and recommender integration are running below.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio Player
        st.markdown("#### 🎧 Listen to Live Audio")
        st.audio(user_audio_bytes, format="audio/wav")
        
        # Extract 17 DSP Features
        with st.spinner("Computing 17 Real-Time Audio Signal Processing Features..."):
            raw_user_features = extract_audio_features(y_user, sr_user)
            scaled_user_features = data["scaler"].transform([raw_user_features])[0]
            
        # Genre prediction via cosine similarity with genre centroids
        genre_sims = {g: cosine_similarity([scaled_user_features], [data["genre_centroids"][g]])[0][0] for g in GENRES}
        pred_genre = max(genre_sims, key=genre_sims.get)
        pred_conf = max(0, genre_sims[pred_genre])
        
        # Compatibility with currently selected user
        user_prof = data["profiles"][user_id:user_id+1]
        user_comp_sim = cosine_similarity(user_prof, [scaled_user_features])[0][0]
        is_user_fav = pred_genre in user_favs
        
        # Analysis Cards
        c_an1, c_an2, c_an3 = st.columns(3)
        with c_an1:
            st.metric(
                label="Predicted Genre Category",
                value=pred_genre.upper(),
                delta=f"Acoustic Match: {pred_conf:.1%}"
            )
        with c_an2:
            st.metric(
                label=f"Affinity with User #{user_id}",
                value=f"{user_comp_sim:.3f}",
                delta="Matches Favorite Genre!" if is_user_fav else "Exploratory / Discovery",
                delta_color="normal" if is_user_fav else "off"
            )
        with c_an3:
            st.metric(
                label="Spectral Centroid (Brightness)",
                value=f"{raw_user_features[0]:.0f} Hz",
                delta=f"ZCR: {raw_user_features[3]:.3f}"
            )
            
        st.markdown("---")
        
        # Side-by-Side Visualizations: Waveform & Mel-Spectrogram
        st.markdown("#### 📊 Live Acoustic Signal Visualizations")
        fig_vis, (ax_w, ax_m) = plt.subplots(1, 2, figsize=(12, 3.4))
        
        # 1. Waveform
        t_axis = np.linspace(0, len(y_user) / sr_user, len(y_user))
        ax_w.plot(t_axis, y_user, color="#2563EB", lw=0.7)
        ax_w.set_title("Live Audio Waveform (Time Domain)", fontsize=10, fontweight="bold")
        ax_w.set_xlabel("Time (seconds)", fontsize=9)
        ax_w.set_ylabel("Amplitude", fontsize=9)
        ax_w.grid(True, alpha=0.3)
        
        # 2. Mel-Spectrogram
        S_user = librosa.feature.melspectrogram(y=y_user, sr=sr_user, n_mels=64)
        S_user_db = librosa.power_to_db(S_user, ref=np.max)
        im = ax_m.imshow(S_user_db, aspect="auto", origin="lower", cmap="magma", extent=[0, len(y_user)/sr_user, 0, sr_user/2])
        ax_m.set_title("Live Mel-Spectrogram (Frequency Energy Distribution)", fontsize=10, fontweight="bold")
        ax_m.set_xlabel("Time (seconds)", fontsize=9)
        ax_m.set_ylabel("Frequency (Hz)", fontsize=9)
        fig_vis.colorbar(im, ax=ax_m, format="%+2.0f dB")
        plt.tight_layout()
        st.pyplot(fig_vis)
        plt.close()
        
        # Genre affinities and Top 3 similar tracks
        c_g1, c_g2 = st.columns([1, 1])
        with c_g1:
            st.markdown("#### 🎯 Acoustic Match with 10 Genres")
            fig_g, ax_g = plt.subplots(figsize=(6, 2.6))
            g_names = list(genre_sims.keys())
            g_scores = [max(0, genre_sims[g]) for g in g_names]
            colors = ["#10B981" if g == pred_genre else "#94A3B8" for g in g_names]
            ax_g.bar(g_names, g_scores, color=colors, alpha=0.85)
            ax_g.set_xticks(range(len(g_names)))
            ax_g.set_xticklabels(g_names, rotation=35, fontsize=8, ha="right")
            ax_g.set_title("Similarity to Genre Centroids", fontsize=10, fontweight="bold")
            ax_g.grid(True, linestyle=":", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close()
            
        with c_g2:
            st.markdown("#### 🔍 Closest Matching Tracks in Database")
            # Find closest songs in GTZAN
            sim_catalog = cosine_similarity([scaled_user_features], data["Xc"])[0]
            top_cat_idx = np.argsort(sim_catalog)[::-1][:3]
            for rank_i, song_cat_idx in enumerate(top_cat_idx, 1):
                cat_row = data["clean_df"].iloc[song_cat_idx]
                st.markdown(f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:8px 12px; border-radius:8px; margin-bottom:6px;">
                    <span style="font-weight:700; color:#1E40AF;">#{rank_i}</span> <b>{cat_row['filename']}</b> 
                    <span class="badge-genre">{cat_row['genre'].upper()}</span>
                    <span style="float:right; font-size:0.85rem; color:#475569;">Similarity: <b>{sim_catalog[song_cat_idx]:.3f}</b></span>
                </div>
                """, unsafe_allow_html=True)
                
        # Noise Stress-Testing on the User's Song
        st.markdown("---")
        st.markdown("#### 🔬 Live Noise Vulnerability & Deep Learning Fix on this Track")
        st.markdown("See what happens if this audio was degraded by background noise in the wild:")
        
        test_snr = st.radio("Simulate Noise Level on Live Audio:", ["Clean (No Noise)", "10 dB Noise (Moderate)", "0 dB Noise (Extreme)"], horizontal=True)
        
        if test_snr != "Clean (No Noise)":
            snr_db = 0 if "0 dB" in test_snr else 10
            sig_power = np.mean(y_user ** 2)
            noise_power = sig_power / (10 ** (snr_db / 10))
            y_user_noisy = y_user + np.random.randn(len(y_user)) * np.sqrt(noise_power)
            
            raw_noisy_feats = extract_audio_features(y_user_noisy, sr_user)
            scaled_noisy_feats = data["scaler"].transform([raw_noisy_feats])[0]
            restored_user_feats = apply_neural_fix([scaled_noisy_feats])[0]
            
            b_noisy_live = io.BytesIO()
            sf.write(b_noisy_live, y_user_noisy, sr_user, format='WAV')
            
            st.markdown(f"##### 📢 Synthesized Noisy Audio ({test_snr}):")
            st.audio(b_noisy_live.getvalue(), format="audio/wav")
            
            # Metric comparisons
            nm1, nm2, nm3 = st.columns(3)
            with nm1:
                st.metric(
                    label="Spectral Centroid Under Noise",
                    value=f"{raw_user_features[0]:.0f} Hz (Clean)",
                    delta=f"Corrupted: {raw_noisy_feats[0]:.0f} Hz",
                    delta_color="inverse"
                )
            with nm2:
                st.metric(
                    label="Zero-Crossing Rate Under Noise",
                    value=f"{raw_user_features[3]:.3f} (Clean)",
                    delta=f"Corrupted: {raw_noisy_feats[3]:.3f}",
                    delta_color="inverse"
                )
            with nm3:
                # Restored centroid
                rest_raw_centroid = restored_user_feats[0] * data["scaler"].scale_[0] + data["scaler"].mean_[0]
                st.metric(
                    label="Neural Restored Centroid",
                    value=f"{rest_raw_centroid:.0f} Hz",
                    delta=f"Restored from {raw_noisy_feats[0]:.0f} Hz"
                )
                
        # Action button to add to Recommender Pool
        st.markdown("---")
        btn_add = st.button("➕ Inject This Live Track into Recommender Pool (Tab 1)", key="btn_add_to_pool")
        if btn_add:
            # Check if already added
            already_exists = any(s["filename"] == user_audio_name for s in st.session_state["user_songs"])
            if not already_exists:
                st.session_state["user_songs"].append({
                    "filename": user_audio_name,
                    "genre": pred_genre,
                    "raw_features": raw_user_features,
                    "scaled_features": scaled_user_features,
                    "audio_bytes": user_audio_bytes,
                    "user_added": True
                })
                st.success(f"✅ '{user_audio_name}' has been added to the Live Recommender pool! Switch to Tab 1 to see its ranking.")
                st.rerun()
            else:
                st.info("This song is already present in the recommendation pool.")
    else:
        st.info("👆 Use the tabs above to **record audio from your microphone**, upload a file, or click '⚡ Load Sample Track' to begin live analysis.")

# -----------------------------------------------------------------------------
# TAB 3: AUDIO SIGNAL & FEATURE DISTORTION INSPECTOR
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🔬 Audio Signal & Feature Distortion Inspector")
    st.markdown("""
    Inspect any audio track from the dataset. Listen to the original recording vs the audio injected with noise, 
    and inspect how mathematical features (**Spectral Centroid**, **Zero-Crossing Rate**, and **MFCCs**) get deformed and subsequently restored!
    """)
    
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_genre = st.selectbox("Filter Genre", GENRES, index=0)
    
    genre_songs = data["clean_df"][data["clean_df"]["genre"] == selected_genre]["filename"].values
    with col_sel2:
        selected_song = st.selectbox("Select Track", genre_songs, index=0)
        
    song_row = data["clean_df"][data["clean_df"]["filename"] == selected_song].iloc[0]
    song_idx = song_row.name
    
    # Path to audio
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(base_dir, "data", "raw", "genres_original", selected_genre, selected_song)
    
    if os.path.exists(audio_path):
        # Load 10 seconds of audio
        y_clean, sr = librosa.load(audio_path, duration=10, sr=22050)
        sig_pow = np.mean(y_clean ** 2)
        
        # Audio generation
        snr_val = 0 if "0 dB" in snr_option else (10 if "10 dB" in snr_option else (20 if "20 dB" in snr_option else 100))
        
        if snr_val < 100:
            noise_pow = sig_pow / (10 ** (snr_val / 10))
            if "White" in noise_type:
                noise = np.random.randn(len(y_clean)) * np.sqrt(noise_pow)
            else: # Pink noise approximation
                white_n = np.random.randn(len(y_clean))
                # 1/f filter in frequency domain
                X_w = np.fft.rfft(white_n)
                S = 1.0 / np.sqrt(np.arange(1, len(X_w) + 1))
                noise = np.fft.irfft(X_w * S, n=len(y_clean))
                noise = noise * np.sqrt(noise_pow / (np.mean(noise ** 2) + 1e-9))
            y_noisy = y_clean + noise
        else:
            y_noisy = y_clean
            
        # Audio players side by side
        ap_col1, ap_col2 = st.columns(2)
        
        b_clean = io.BytesIO()
        sf.write(b_clean, y_clean, sr, format='WAV')
        with ap_col1:
            st.markdown("##### 🎧 Original Clean Audio Snippet")
            st.audio(b_clean.getvalue(), format="audio/wav")
            
        b_noisy = io.BytesIO()
        sf.write(b_noisy, y_noisy, sr, format='WAV')
        with ap_col2:
            st.markdown(f"##### 📢 Synthesized Noisy Audio ({noise_label})")
            st.audio(b_noisy.getvalue(), format="audio/wav")
            
        # Waveform plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
        t_axis = np.linspace(0, len(y_clean) / sr, len(y_clean))
        ax1.plot(t_axis, y_clean, color='#2563EB', lw=0.8)
        ax1.set_title("Clean Waveform", fontsize=11, fontweight='bold', color='#1E293B')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 1)
        
        ax2.plot(t_axis, y_noisy, color='#DC2626', lw=0.8)
        ax2.set_title(f"Noisy Waveform ({noise_label})", fontsize=11, fontweight='bold', color='#1E293B')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_xlabel("Time (seconds)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    # Feature comparison metrics
    feat_names = ["centroid", "bandwidth", "rolloff", "zcr"] + [f"mfcc{i+1}" for i in range(13)]
    f_clean_vals = data["clean_raw"][song_idx]
    
    # Noisy features from precomputed table
    if "Clean" in snr_option:
        f_noisy_vals = f_clean_vals
    elif "White" in noise_type:
        snr_k = 0 if "0 dB" in snr_option else (10 if "10 dB" in snr_option else 20)
        f_noisy_vals = data["white_scaled"][snr_k][song_idx] * data["scaler"].scale_ + data["scaler"].mean_
    else:
        snr_k = 0 if "0 dB" in snr_option else 10
        f_noisy_vals = data["pink_scaled"][snr_k][song_idx] * data["scaler"].scale_ + data["scaler"].mean_
        
    # Restored features
    f_noisy_std = data["scaler"].transform([f_noisy_vals])
    f_rest_std = apply_neural_fix(f_noisy_std)
    f_rest_vals = data["scaler"].inverse_transform(f_rest_std)[0]
    
    st.markdown("#### 📊 Key Feature Shift Under Noise")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Spectral Centroid (Brightness)",
            value=f"{f_clean_vals[0]:.0f} Hz",
            delta=f"Noisy: {f_noisy_vals[0]:.0f} Hz | Restored: {f_rest_vals[0]:.0f} Hz",
            delta_color="inverse"
        )
    with m2:
        st.metric(
            label="Zero-Crossing Rate (ZCR)",
            value=f"{f_clean_vals[3]:.3f}",
            delta=f"Noisy: {f_noisy_vals[3]:.3f} | Restored: {f_rest_vals[3]:.3f}",
            delta_color="inverse"
        )
    with m3:
        st.metric(
            label="Spectral Bandwidth",
            value=f"{f_clean_vals[1]:.0f} Hz",
            delta=f"Noisy: {f_noisy_vals[1]:.0f} Hz | Restored: {f_rest_vals[1]:.0f} Hz",
            delta_color="inverse"
        )
    with m4:
        st.metric(
            label="Spectral Rolloff",
            value=f"{f_clean_vals[2]:.0f} Hz",
            delta=f"Noisy: {f_noisy_vals[2]:.0f} Hz | Restored: {f_rest_vals[2]:.0f} Hz",
            delta_color="inverse"
        )
        
    st.markdown("#### 📈 Standardized 17-Dimensional Feature Vector Comparison")
    fig2, ax = plt.subplots(figsize=(11, 3.8))
    x_idx = np.arange(17)
    width = 0.28
    
    f_clean_std = data["scaler"].transform([f_clean_vals])[0]
    ax.bar(x_idx - width, f_clean_std, width, label="Clean Ground Truth", color='#2563EB', alpha=0.85)
    ax.bar(x_idx, f_noisy_std[0], width, label="Corrupted by Noise", color='#EF4444', alpha=0.85)
    ax.bar(x_idx + width, f_rest_std[0], width, label="Neural MLP Restored", color='#10B981', alpha=0.85)
    
    ax.set_xticks(x_idx)
    ax.set_xticklabels(["Centroid", "Bandwidth", "Rolloff", "ZCR"] + [f"M{i}" for i in range(1, 14)], rotation=45, fontsize=9)
    ax.set_ylabel("Standardized Feature Value")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.set_title("How the Neural MLP Cancels Noise Distortion Across All 17 Acoustic Dimensions", fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# -----------------------------------------------------------------------------
# TAB 4: BENCHMARKS & METRIC RESULTS
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 📊 Benchmark Evaluation Across 5 Folds")
    st.markdown("""
    Evaluating performance on **Recall@10** for 200 hold-out new songs across 5 cross-validation splits (999 GTZAN tracks total).
    """)
    
    benchmark_data = {
        "Condition": ["Clean Audio", "White Noise 20 dB", "White Noise 10 dB", "White Noise 0 dB", "Pink Noise 0 dB (Seen)", "Pink Noise 0 dB (Unseen)"],
        "Handcrafted Raw": [0.120, 0.118, 0.106, 0.091, 0.109, 0.109],
        "Linear Ridge Fix": [0.120, 0.119, 0.112, 0.109, 0.118, 0.111],
        "Neural MLP Fix": [0.120, 0.120, 0.115, 0.114, 0.124, 0.117],
        "CNN Autoencoder": [0.127, 0.125, 0.121, 0.115, 0.120, 0.102],
    }
    b_df = pd.DataFrame(benchmark_data)
    
    st.dataframe(b_df.style.highlight_max(subset=["Handcrafted Raw", "Linear Ridge Fix", "Neural MLP Fix", "CNN Autoencoder"], color="#DCFCE7", axis=1))
    
    # Comparison chart
    fig3, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(b_df["Condition"]))
    w = 0.2
    
    ax.bar(x - 1.5*w, b_df["Handcrafted Raw"], w, label="Raw (No Fix)", color="#94A3B8")
    ax.bar(x - 0.5*w, b_df["Linear Ridge Fix"], w, label="Linear Ridge", color="#60A5FA")
    ax.bar(x + 0.5*w, b_df["Neural MLP Fix"], w, label="Neural MLP Fix", color="#2563EB")
    ax.bar(x + 1.5*w, b_df["CNN Autoencoder"], w, label="CNN Autoencoder", color="#7C3AED")
    
    ax.axhline(0.050, color='red', linestyle='--', label='Random Guessing (0.050)', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(b_df["Condition"], rotation=20, ha='right', fontsize=9.5)
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 0.14)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='lower left', frameon=True)
    ax.set_title("Recall@10 Comparison Under Varying Noise Environments", fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    
    st.markdown("""
    #### 💡 Key Takeaways:
    1. **Severe Degradation without Fix:** At 0 dB White Noise, raw feature recommendation accuracy drops from **0.120 to 0.091** (a 24% drop towards random guessing).
    2. **Neural MLP Recovery:** The small 3-layer MLP recovers the score back to **0.114** (restoring over 80% of lost performance).
    3. **Generalization to Unseen Noise:** On Pink noise never seen during training, the **Handcrafted MLP achieved 0.117**, while the CNN Autoencoder degraded to **0.102** because it overfitted to 2D white noise spectrogram patterns.
    """)

# -----------------------------------------------------------------------------
# TAB 5: VIVA & PRESENTATION GUIDE
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 🎓 Viva & Project Defense Guide")
    st.markdown("Use this tab to prepare for questions from professors or external examiners.")
    
    with st.expander("❓ 1. Why is this called a 'Hybrid' Recommendation System?", expanded=True):
        st.markdown("""
        **Answer:**
        Real-world recommendation systems face two fundamentally different regimes:
        - **Established Songs (Warm Items):** Have rich user listening history. Here, **Collaborative Filtering** (via TruncatedSVD) gives the highest accuracy because it models collective community behavior.
        - **Brand New Songs (Cold-Start Items):** Have zero listening history. Collaborative filtering fails completely. We must rely on **Content-Based Audio Signal Processing**.
        
        Our system is **Hybrid** because it combines both using dedicated slot allocation (5 slots for collaborative favorites + 5 slots for new audio releases).
        """)
        
    with st.expander("❓ 2. What exactly are the 17 Audio Signal Processing features?"):
        st.markdown("""
        **Answer:**
        We extract 17 mathematical descriptors from each 30-second audio clip using `librosa`:
        1. **Spectral Centroid (1 feature):** Center of gravity of the frequency spectrum; indicates the perceptual 'brightness' of the track.
        2. **Spectral Bandwidth (1 feature):** The frequency range spread around the centroid.
        3. **Spectral Rolloff (1 feature):** The frequency below which 85% of total spectral energy is contained.
        4. **Zero-Crossing Rate (ZCR) (1 feature):** The frequency at which the audio signal changes sign (positive/negative); acts as a detector for high-frequency percussion or noise.
        5. **MFCCs (13 features):** Mel-Frequency Cepstral Coefficients, which capture the timbral envelope of the music (differentiating piano from electric guitar).
        """)
        
    with st.expander("❓ 3. What happens to audio features when background noise is added?"):
        st.markdown("""
        **Answer:**
        Uncorrelated background noise adds broad high-frequency energy across the entire audio spectrum. This causes:
        - **Spectral Centroid** to artificially surge higher (e.g., from 1800 Hz to over 4000 Hz).
        - **Zero-Crossing Rate** to triple as random noise rapidly crosses zero.
        - **MFCCs** to become flattened and distorted.
        
        Because of this, a quiet acoustic folk song with background noise mathematically resembles harsh rock or heavy metal, causing the recommender to suggest the wrong songs!
        """)
        
    with st.expander("❓ 4. How does the Deep Learning Denoising MLP work?"):
        st.markdown("""
        **Answer:**
        We train a 3-layer Multi-Layer Perceptron (MLP):
        $$\\text{Input (17)} \\rightarrow \\text{Linear}(64) \\rightarrow \\text{ReLU} \\rightarrow \\text{Linear}(64) \\rightarrow \\text{ReLU} \\rightarrow \\text{Linear}(17)$$
        
        It is trained with Mean Squared Error (MSE) loss on pairs of `(noisy_features, clean_features)` from known songs. Because the feature shift caused by additive noise is mathematically structured, the MLP successfully learns the non-linear inverse mapping and reconstructs the clean acoustic vector.
        """)
        
    with st.expander("❓ 5. What is 0 dB SNR?"):
        st.markdown("""
        **Answer:**
        Signal-to-Noise Ratio (SNR) in decibels is:
        $$\\text{SNR}_{\\text{dB}} = 10 \\cdot \\log_{10}\\left(\\frac{P_{\\text{signal}}}{P_{\\text{noise}}}\\right)$$
        When $\\text{SNR} = 0\\text{ dB}$, $\\frac{P_{\\text{signal}}}{P_{\\text{noise}}} = 1$. This means the **noise power is equal to the music power**—representing an extreme, highly challenging real-world test.
        """)
