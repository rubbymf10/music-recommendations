import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
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
    zip_path = "spotify_songs.csv.zip"
    extract_path = "spotify_data"

    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    csv_path = os.path.join(extract_path, "spotify_songs.csv")
    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "track_name": "judul_musik",
        "track_artist": "artist",
        "track_popularity": "popularity",
        "track_album_name": "album",
        "playlist_genre": "genre",
        "playlist_subgenre": "subgenre"
    })
    df_clean = df.dropna(subset=["popularity", "genre", "subgenre", "tempo", "duration_ms", "energy", "danceability"])

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

@st.cache_resource
def train_model(df_clean):
    tfidf_genre = TfidfVectorizer()
    tfidf_subgenre = TfidfVectorizer()
    tfidf_title = TfidfVectorizer()
    tfidf_artist = TfidfVectorizer()
    tfidf_lyrics = TfidfVectorizer(max_features=500)
    tfidf_album = TfidfVectorizer()

    genre_tfidf = tfidf_genre.fit_transform(df_clean['genre'])
    subgenre_tfidf = tfidf_subgenre.fit_transform(df_clean['subgenre'])
    title_tfidf = tfidf_title.fit_transform(df_clean['judul_musik'])
    artist_tfidf = tfidf_artist.fit_transform(df_clean['artist'])
    lyrics_tfidf = tfidf_lyrics.fit_transform(df_clean['lyrics'].fillna(''))
    album_tfidf = tfidf_album.fit_transform(df_clean['album'])

    features_num = ['tempo', 'duration_ms', 'energy', 'danceability']
    scaler = MinMaxScaler()
    df_num_scaled = pd.DataFrame(scaler.fit_transform(df_clean[features_num]), columns=features_num, index=df_clean.index)

    X = pd.concat([
        pd.DataFrame(genre_tfidf.toarray(), index=df_clean.index),
        pd.DataFrame(subgenre_tfidf.toarray(), index=df_clean.index),
        pd.DataFrame(title_tfidf.toarray(), index=df_clean.index),
        pd.DataFrame(artist_tfidf.toarray(), index=df_clean.index),
        pd.DataFrame(lyrics_tfidf.toarray(), index=df_clean.index),
        pd.DataFrame(album_tfidf.toarray(), index=df_clean.index),
        df_num_scaled
    ], axis=1)

    y = df_clean['pop_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, tfidf_genre, tfidf_subgenre, tfidf_title, tfidf_artist, tfidf_lyrics, tfidf_album, scaler, title_tfidf

model, tfidf_genre, tfidf_subgenre, tfidf_title, tfidf_artist, tfidf_lyrics, tfidf_album, scaler, title_tfidf = train_model(df_clean)

def music_card(title, artist, popularity):
    st.markdown(f"""
    <div class=\"music-card\">
        <div class=\"music-cover\">🎵</div>
        <div class=\"music-info\">
            <p class=\"music-title\">{title}</p>
            <p class=\"music-artist\">{artist}</p>
        </div>
        <div class=\"popularity\">{int(popularity)}</div>
    </div>
    """, unsafe_allow_html=True)

st.title("\U0001F3B5 Sistem Rekomendasi Musik Spotify")
judul_list = df_clean['judul_musik'].dropna().unique()
pilihan = st.selectbox("Pilih judul lagu", options=judul_list)
manual_input = st.text_input("Atau masukkan manual judul lagu")
judul = manual_input if manual_input.strip() else pilihan

if st.button("Rekomendasikan"):
    judul_vector = tfidf_title.transform([judul])
    similarities = cosine_similarity(judul_vector, title_tfidf).flatten()
    top_index = similarities.argsort()[::-1][0]
    lagu = df_clean.iloc[top_index]

    X_input = np.hstack([
        tfidf_genre.transform([lagu['genre']]).toarray(),
        tfidf_subgenre.transform([lagu['subgenre']]).toarray(),
        tfidf_title.transform([lagu['judul_musik']]).toarray(),
        tfidf_artist.transform([lagu['artist']]).toarray(),
        tfidf_lyrics.transform([lagu['lyrics'] if pd.notna(lagu['lyrics']) else '']).toarray(),
        tfidf_album.transform([lagu['album']]).toarray(),
        scaler.transform([[lagu['tempo'], lagu['duration_ms'], lagu['energy'], lagu['danceability']]])
    ])

    pred = model.predict(X_input)[0]
    kategori = label_enc.inverse_transform([pred])[0]

    st.success(f"Judul terdekat: **{lagu['judul_musik']}** oleh **{lagu['artist']}**")
    st.info(f"Genre: **{lagu['genre']}** | Subgenre: **{lagu['subgenre']}**")
    st.success(f"Prediksi popularitas: **{kategori}**")

    df_rekom_genre = df_clean[df_clean['genre'].str.lower() == lagu['genre'].lower()].sort_values(by='popularity', ascending=False).head(5)
    st.subheader("\U0001F3A7 Rekomendasi Berdasarkan Genre")
    for _, row in df_rekom_genre.iterrows():
        music_card(row['judul_musik'], row['artist'], row['popularity'])

    top_indices = similarities.argsort()[::-1][1:6]
    df_rekom_judul = df_clean.iloc[top_indices]
    st.subheader("\U0001F4D6 Rekomendasi Berdasarkan Kemiripan Judul")
    for _, row in df_rekom_judul.iterrows():
        music_card(row['judul_musik'], row['artist'], row['popularity'])

    if pd.notna(lagu['lyrics']) and lagu['lyrics'].strip():
        lyric_vector = tfidf_lyrics.transform([lagu['lyrics']])
        lyric_similarities = cosine_similarity(lyric_vector, tfidf_lyrics.transform(df_clean['lyrics'].fillna(''))).flatten()
        top_lyric_indices = lyric_similarities.argsort()[::-1][1:6]
        df_rekom_lyrics = df_clean.iloc[top_lyric_indices]

        st.subheader("\U0001F3BC Rekomendasi Berdasarkan Lirik")
        for _, row in df_rekom_lyrics.iterrows():
            music_card(row['judul_musik'], row['artist'], row['popularity'])
    else:
        st.info("Lagu ini tidak memiliki lirik yang dapat dibandingkan.")
