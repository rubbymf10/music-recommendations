import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_songs.csv")
    return df

df = load_data()

# Features for recommendation
feature_cols = ['danceability', 'energy', 'valence', 'tempo',
                'acousticness', 'instrumentalness', 'speechiness', 'liveness']

# Track history
if 'history' not in st.session_state:
    st.session_state.history = []

# Page Navigation
page = st.sidebar.selectbox("Pilih Halaman", ["Home", "Rekomendasi", "Riwayat"])

# ------------------ PAGE 1: HOME ------------------
if page == "Home":
    st.title("Top Lagu Spotify")

    # Top 10 lagu terpopuler
    st.subheader("Top 10 Lagu Terpopuler")
    top10 = df.sort_values(by="track_popularity", ascending=False).drop_duplicates("track_name").head(10)
    st.table(top10[['track_name', 'track_artist', 'track_popularity']])

    # 5 lagu terpopuler per genre
    st.subheader("Top 5 Lagu per Genre")
    for genre in df['playlist_genre'].dropna().unique():
        st.markdown(f"### {genre.title()}")
        top5_genre = df[df['playlist_genre'] == genre]
        top5_genre = top5_genre.sort_values(by='track_popularity', ascending=False).drop_duplicates('track_name').head(5)
        st.table(top5_genre[['track_name', 'track_artist', 'track_popularity']])

# ------------------ PAGE 2: REKOMENDASI ------------------
elif page == "Rekomendasi":
    st.title("Rekomendasi Lagu Berdasarkan Input Anda")

    track_input = st.text_input("Masukkan judul lagu favorit Anda")

    if track_input:
        matched = df[df['track_name'].str.lower().str.contains(track_input.lower())]

        if not matched.empty:
            selected_track = matched.iloc[0]
            st.success(f"Lagu dipilih: {selected_track['track_name']} - {selected_track['track_artist']}")

            # Simpan ke history
            st.session_state.history.append({
                "input": f"{selected_track['track_name']} - {selected_track['track_artist']}",
                "output": []
            })

            # Cari rekomendasi berdasarkan kemiripan fitur
            input_features = selected_track[feature_cols].values.reshape(1, -1)
            all_features = df[feature_cols].values
            similarities = cosine_similarity(input_features, all_features)[0]
            df['similarity'] = similarities
            recommendations = df[df['track_id'] != selected_track['track_id']].sort_values(by='similarity', ascending=False)
            recommendations = recommendations.drop_duplicates('track_name').head(5)

            # Tampilkan
            st.subheader("Rekomendasi Lagu:")
            st.table(recommendations[['track_name', 'track_artist', 'playlist_genre', 'similarity']])

            # Simpan output ke history
            st.session_state.history[-1]['output'] = recommendations[['track_name', 'track_artist']].values.tolist()

        else:
            st.warning("Lagu tidak ditemukan dalam dataset.")

# ------------------ PAGE 3: HISTORY ------------------
elif page == "Riwayat":
    st.title("Riwayat Pencarian dan Rekomendasi")

    if st.session_state.history:
        for i, record in enumerate(st.session_state.history[::-1]):
            st.markdown(f"### Pencarian #{len(st.session_state.history) - i}")
            st.markdown(f"**Input Lagu:** {record['input']}")
            st.markdown("**Hasil Rekomendasi:**")
            for song in record['output']:
                st.write(f"- {song[0]} - {song[1]}")
    else:
        st.info("Belum ada pencarian dilakukan.")
