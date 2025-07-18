import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Styling dark mode ---
st.markdown("""
    <style>
    .main, .block-container {
        background-color: #121212;
        color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #1DB954;
    }
    button[kind="primary"] {
        background-color: #1DB954;
        color: #FFFFFF;
        border-radius: 20px;
        border: none;
        padding: 8px 20px;
    }
    button[kind="primary"]:hover {
        background-color: #1ed760;
    }
    .stTextInput>div>div>input {
        background-color: #222222;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 8px;
    }
    .music-card {
        background-color: #282828;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .music-card:hover {
        background-color: #333333;
    }
    .music-cover {
        width: 50px;
        height: 50px;
        color: #1DB954;
        font-size: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 6px;
        flex-shrink: 0;
    }
    .music-info {
        flex-grow: 1;
    }
    .music-title {
        font-weight: 600;
        font-size: 16px;
        margin: 0;
    }
    .music-artist {
        color: #b3b3b3;
        margin: 0;
        font-size: 14px;
    }
    .popularity {
        color: #1DB954;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("tracks.csv")
    df_clean = df.dropna(subset=['popularity', 'genre', 'tempo', 'duration_ms', 'energy', 'danceability'])
    
    low_thresh = df_clean['popularity'].quantile(0.33)
    high_thresh = df_clean['popularity'].quantile(0.66)

    def categorize_popularity(pop):
        if pop <= low_thresh:
            return 'Rendah'
        elif pop > high_thresh:
            return 'Tinggi'
        else:
            return np.nan

    df_clean['pop_category'] = df_clean['popularity'].apply(categorize_popularity)
    df_clean = df_clean.dropna(subset=['pop_category'])
    label_enc = LabelEncoder()
    df_clean['pop_encoded'] = label_enc.fit_transform(df_clean['pop_category'])
    return df, df_clean, label_enc

df, df_clean, label_enc = load_data()

# --- Train Model ---
@st.cache_resource
def train_model(df_clean):
    tfidf_genre = TfidfVectorizer()
    tfidf_title = TfidfVectorizer()
    tfidf_artist = TfidfVectorizer()
    tfidf_album = TfidfVectorizer()

    genre_tfidf = tfidf_genre.fit_transform(df_clean['genre'].fillna(''))
    title_tfidf = tfidf_title.fit_transform(df_clean['name'].fillna(''))
    artist_tfidf = tfidf_artist.fit_transform(df_clean['artists'].fillna(''))
    album_tfidf = tfidf_album.fit_transform(df_clean['album'].fillna(''))

    df_genre = pd.DataFrame(genre_tfidf.toarray(), index=df_clean.index)
    df_title = pd.DataFrame(title_tfidf.toarray(), index=df_clean.index)
    df_artist = pd.DataFrame(artist_tfidf.toarray(), index=df_clean.index)
    df_album = pd.DataFrame(album_tfidf.toarray(), index=df_clean.index)

    features_num = ['tempo', 'duration_ms', 'energy', 'danceability']
    scaler = MinMaxScaler()
    df_num_scaled = pd.DataFrame(scaler.fit_transform(df_clean[features_num]), columns=features_num, index=df_clean.index)

    X = pd.concat([df_genre, df_title, df_artist, df_album, df_num_scaled], axis=1)
    y = df_clean['pop_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, tfidf_genre, tfidf_title, tfidf_artist, tfidf_album, scaler, X_test, y_test, y_pred, title_tfidf

model, tfidf_genre, tfidf_title, tfidf_artist, tfidf_album, scaler, X_test, y_test, y_pred, title_tfidf = train_model(df_clean)

# --- Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'recommendation_table' not in st.session_state:
    st.session_state.recommendation_table = pd.DataFrame()

# --- Sidebar ---
with st.sidebar:
    st.markdown('<h2 style="color:#1DB954; margin-bottom: 15px;">\U0001F3B5 Dashboard</h2>', unsafe_allow_html=True)
    halaman = st.radio("", ["Beranda", "Rekomendasi Musik", "Histori"], index=0, key="page_select")

# --- Komponen UI Musik ---
def music_card(title, artist, popularity):
    st.markdown(f"""
    <div class="music-card">
        <div class="music-cover">🎵</div>
        <div class="music-info">
            <p class="music-title">{title}</p>
            <p class="music-artist">{artist}</p>
        </div>
        <div class="popularity">{int(popularity)}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Halaman Beranda ---
if halaman == "Beranda":
    st.header("Top 10 Musik Terpopuler")
    top10 = df.sort_values(by='popularity', ascending=False).head(10)
    for _, row in top10.iterrows():
        music_card(row['name'], row['artists'], row['popularity'])

    st.markdown("---")
    st.header("5 Musik Terpopuler dari Setiap Genre")
    genre_list = df['genre'].dropna().unique()
    for genre in genre_list:
        st.subheader(f"🎶 Genre: {genre}")
        top5_by_genre = df[df['genre'] == genre].sort_values(by='popularity', ascending=False).head(5)
        for _, row in top5_by_genre.iterrows():
            music_card(row['name'], row['artists'], row['popularity'])

# --- Halaman Histori ---
if halaman == "Histori":
    st.header("Riwayat Pencarian Rekomendasi")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(f"- **{h['Judul']}** oleh {h['Artis']} (Genre: {h['Genre']}, Prediksi: {h['Prediksi']})")
    else:
        st.info("Belum ada pencarian.")

    st.markdown("---")
    st.header("🎧 Hasil Rekomendasi")
    if not st.session_state.recommendation_table.empty:
        df_show = st.session_state.recommendation_table.sort_values(by='popularity', ascending=False)
        for _, row in df_show.iterrows():
            music_card(row['name'], row['artists'], row['popularity'])
    else:
        st.info("Belum ada rekomendasi genre ditampilkan.")

    if st.button("Reset Riwayat Pencarian"):
        st.session_state.history = []
        st.session_state.recommendation_table = pd.DataFrame()
        st.experimental_rerun()
        st.stop()

# --- Halaman Rekomendasi Musik ---
if halaman == "Rekomendasi Musik":
    st.header("Rekomendasi Musik Berdasarkan Judul")

    judul_list = df_clean['name'].dropna().unique()
    pilihan = st.selectbox("Pilih dari daftar judul musik", options=judul_list)
    manual_input = st.text_input("Atau ketik judul musik secara manual (opsional)")
    judul = manual_input if manual_input.strip() else pilihan

    if st.button("Rekomendasikan"):
        if not judul.strip():
            st.warning("Silakan masukkan judul musik terlebih dahulu.")
        else:
            judul_vector = tfidf_title.transform([judul])
            similarities = cosine_similarity(judul_vector, title_tfidf).flatten()
            top_index = similarities.argsort()[::-1][0]
            lagu = df_clean.iloc[[top_index]]

            fitur = lagu.iloc[0]
            genre = fitur['genre']
            tempo = fitur['tempo']
            duration_ms = fitur['duration_ms']
            energy = fitur['energy']
            danceability = fitur['danceability']
            artist = fitur['artists']
            album = fitur['album']
            judul_terdekat = fitur['name']

            X_input = np.hstack([
                tfidf_genre.transform([genre]).toarray(),
                tfidf_title.transform([judul_terdekat]).toarray(),
                tfidf_artist.transform([artist]).toarray(),
                tfidf_album.transform([album]).toarray(),
                scaler.transform([[tempo, duration_ms, energy, danceability]])
            ])

            pred = model.predict(X_input)[0]
            kategori = label_enc.inverse_transform([pred])[0]

            st.success(f"Input **'{judul}'** paling mirip dengan lagu **'{judul_terdekat}'** oleh **{artist}**.")
            st.info(f"Genre lagu tersebut adalah **{genre}**.")
            st.success(f"Musik ini diprediksi memiliki tingkat popularitas: **{kategori}**.")

            df_rekom_genre = df_clean[df_clean['genre'].str.lower() == genre.lower()].sort_values(by='popularity', ascending=False).head(5)
            st.subheader("🎧 Rekomendasi Berdasarkan Genre yang Sama")
            for _, row in df_rekom_genre.iterrows():
                music_card(row['name'], row['artists'], row['popularity'])

            top_indices = similarities.argsort()[::-1][1:6]
            df_rekom_judul = df_clean.iloc[top_indices]
            st.subheader("\U0001F3A7 Rekomendasi Berdasarkan Kemiripan Judul")
            for _, row in df_rekom_judul.sort_values(by='popularity', ascending=False).iterrows():
                music_card(row['name'], row['artists'], row['popularity'])

            df_rekomendasi = pd.concat([df_rekom_genre, df_rekom_judul]).drop_duplicates(subset='name')
            st.session_state.recommendation_table = df_rekomendasi

            st.session_state.history.append({
                'Judul': judul,
                'Artis': artist,
                'Genre': genre,
                'Prediksi': kategori,
                'Rekomendasi': ', '.join(df_rekomendasi['name'].head(3).tolist())
            })
