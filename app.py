import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
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
page = st.sidebar.selectbox("Pilih Halaman", ["Home", "Rekomendasi", "Riwayat", "Simulasi RF"])

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
    st.title("Rekomendasi Lagu Mirip dengan Input Anda")

    track_input = st.text_input("🎧 Masukkan judul lagu favorit Anda")

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

            # Tampilkan hasil seperti tampilan Spotify
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

# ------------------ PAGE 4: SIMULASI RF ------------------
elif page == "Simulasi RF":
    st.title("Simulasi Rekomendasi Personal (Random Forest)")

    st.subheader("1. Pilih Lagu Favorit")
    track_options = df[['track_name', 'track_artist']].drop_duplicates()
    selected_tracks = st.multiselect("Pilih 3-10 lagu favorit:",
                                     track_options.apply(lambda x: f"{x['track_name']} - {x['track_artist']}", axis=1))

    if st.button("Latih Model dan Rekomendasikan") and selected_tracks:
        liked_ids = []
        for track_str in selected_tracks:
            name, artist = track_str.split(" - ", 1)
            result = df[(df['track_name'] == name) & (df['track_artist'] == artist)]
            if not result.empty:
                liked_ids.append(result.iloc[0]['track_id'])

        df['liked'] = df['track_id'].apply(lambda x: 1 if x in liked_ids else 0)

        X = df[feature_cols]
        y = df['liked']

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        df['like_prob'] = rf.predict_proba(X)[:, 1]
        rekomendasi = df[~df['track_id'].isin(liked_ids)].sort_values(by='like_prob', ascending=False).drop_duplicates('track_name').head(10)

        st.subheader("Rekomendasi Berdasarkan Model Anda:")
        for idx, row in rekomendasi.iterrows():
            st.markdown(f"- **{row['track_name']}** — *{row['track_artist']}* (🎧 {row['playlist_genre']}) | Probabilitas Suka: `{row['like_prob']:.2f}`")
    elif selected_tracks:
        st.info("Klik tombol untuk melatih model dan melihat rekomendasi.")
