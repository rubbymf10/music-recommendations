import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
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
if 'liked_songs' not in st.session_state:
    st.session_state.liked_songs = []

# Page Navigation
page = st.sidebar.selectbox("Pilih Halaman", ["Home", "Rekomendasi & Simulasi", "Riwayat"])

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

# ------------------ PAGE 2: REKOMENDASI & SIMULASI ------------------
elif page == "Rekomendasi & Simulasi":
    st.title("Rekomendasi Lagu Berdasarkan Preferensi Anda")

    tab1, tab2 = st.tabs(["Rekomendasi Mirip Lagu", "Rekomendasi Personal (RF)"])

    # Tab 1: Content-Based Recommendation
    with tab1:
        st.subheader("Rekomendasi Lagu Berdasarkan Judul dan Genre")

        track_input = st.text_input("🎧 Masukkan judul lagu (opsional)")
        selected_genre = st.selectbox("🎼 Pilih Genre:", ["Semua"] + sorted(df['playlist_genre'].dropna().unique()))

        filtered_df = df.copy()
        if selected_genre != "Semua":
            filtered_df = filtered_df[filtered_df['playlist_genre'] == selected_genre]

        if track_input:
            filtered_df = filtered_df[filtered_df['track_name'].str.lower().str.contains(track_input.lower())]

        if not filtered_df.empty:
            selected_track = filtered_df.iloc[0]
            st.success(f"Lagu dipilih: {selected_track['track_name']} - {selected_track['track_artist']}")

            st.session_state.history.append({
                "input": f"{selected_track['track_name']} - {selected_track['track_artist']}",
                "output": []
            })

            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(df[feature_cols])
            input_features = scaler.transform(selected_track[feature_cols].values.reshape(1, -1))

            similarities = cosine_similarity(input_features, scaled_features)[0]
            df['similarity'] = similarities
            recommendations = df[df['track_id'] != selected_track['track_id']].sort_values(by='similarity', ascending=False)
            recommendations = recommendations.drop_duplicates('track_name').head(5)

            st.markdown("## 🎯 Rekomendasi Teratas")
            top_reco = recommendations.iloc[0]
            st.markdown(f"**🎵 {top_reco['track_name']}**  ")
            st.markdown(f"*{top_reco['track_artist']}*  ")
            st.markdown(f"Genre: `{top_reco['playlist_genre']}` | Kecocokan: `{top_reco['similarity']:.2f}`")

            st.markdown("---")
            st.markdown("## 🔁 More Like This")
            for i in range(1, len(recommendations)):
                row = recommendations.iloc[i]
                st.markdown(f"- **{row['track_name']}** — *{row['track_artist']}* (🎧 {row['playlist_genre']})")

            st.session_state.history[-1]['output'] = recommendations[['track_name', 'track_artist']].values.tolist()
        else:
            st.info("Tidak ditemukan lagu dengan kriteria tersebut.")

    # Tab 2: Personalized RF Recommendation by Genre
    with tab2:
        st.subheader("Rekomendasi Berdasarkan Genre dan Urutan")

        genre_choice = st.selectbox("🎼 Pilih Genre:", sorted(df['playlist_genre'].dropna().unique()))
        sort_by = st.radio("Urutkan Berdasarkan:", ["track_popularity", "track_album_release_date"])

        genre_df = df[df['playlist_genre'] == genre_choice]
        if sort_by == "track_album_release_date":
            genre_df = genre_df.sort_values(by=sort_by, ascending=True)
        else:
            genre_df = genre_df.sort_values(by=sort_by, ascending=False)

        st.markdown(f"### 🔝 Lagu Terpopuler dalam Genre `{genre_choice}`")
        for idx, row in genre_df.drop_duplicates('track_name').head(10).iterrows():
            st.markdown(f"- **{row['track_name']}** — *{row['track_artist']}* ({row['track_album_release_date']})")

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
