# ==========================
# Imports
# ==========================
import os
import json
import zipfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# seaborn hanya untuk EDA opsional (tidak wajib)
import seaborn as sns  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Konstanta & Path
# ==========================
KAGGLE_DATASET = "amitanshjoshi/spotify-1million-tracks"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_ZIP = DATA_DIR / "spotify_1m_raw.zip"
RAW_CSV_CANDIDATES = [
    "spotify_data.csv",  # terlihat di notebook dan artikel terkait
    "spotify_1mtracks.csv",
    "Spotify_1Million_Tracks.csv",
]
CLEAN_CACHE_PARQUET = DATA_DIR / "spotify_1m_clean.parquet"
META_CACHE_JSON = DATA_DIR / "spotify_1m_meta.json"

# ==========================
# Styling (dark theme)
# ==========================
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# ==========================
# Utility: Kaggle API download
# ==========================
def kaggle_auth_available() -> bool:
    """Cek apakah kredensial Kaggle ada di environment atau file ~/.kaggle/kaggle.json."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def kaggle_download_if_needed(dataset: str = KAGGLE_DATASET, out_zip: Path = RAW_ZIP) -> Path | None:
    """Download dataset Kaggle jika belum ada.

    Menggunakan KaggleApi jika kredensial tersedia. Jika gagal, kembalikan None.
    """
    if out_zip.exists():
        return out_zip

    if not kaggle_auth_available():
        st.warning("Kredensial Kaggle tidak ditemukan. Silakan unggah data manual atau set environment KAGGLE_USERNAME dan KAGGLE_KEY.")
        return None

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # pragma: no cover
        st.error(f"Gagal import KaggleApi: {e}")
        return None

    try:
        api = KaggleApi()
        api.authenticate()
        st.info("Mengunduh dataset dari Kaggle, mohon tunggu...")
        api.dataset_download_files(dataset, path=str(DATA_DIR), force=False, quiet=False)
    except Exception as e:  # pragma: no cover
        st.error(f"Gagal download dari Kaggle: {e}")
        return None

    # Kaggle API menaruh file <slug>.zip di folder path
    # Cari file zip terbaru di DATA_DIR
    zips = sorted(DATA_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if zips:
        dl_zip = zips[0]
        if dl_zip != out_zip:
            dl_zip.rename(out_zip)
        return out_zip
    else:
        st.error("Zip dataset Kaggle tidak ditemukan setelah download.")
        return None


# ==========================
# Utility: ekstrak zip & temukan CSV
# ==========================

def extract_and_find_csv(zip_path: Path) -> Path | None:
    """Ekstrak zip dan cari CSV utama.
    Mengembalikan path ke CSV jika ada.
    """
    if not zip_path or not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    # Cari kandidat CSV berdasarkan nama umum
    for name in RAW_CSV_CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            return p
    # Jika tidak ketemu, cari file CSV terbesar di folder
    csvs = list(DATA_DIR.glob("*.csv"))
    if csvs:
        return max(csvs, key=lambda p: p.stat().st_size)
    return None


# ==========================
# Utility: pemetaan kolom dataset Kaggle -> kolom app
# ==========================
KAGGLE_TO_APP_MAP_PRIORITY = {
    "judul_musik": ["track_name", "name", "title", "song", "track"],
    "artist": ["artist_name", "artists", "track_artist", "artist"],
    "album": ["album_name", "track_album_name", "album"],
    "genre": ["genre", "track_genre", "playlist_genre", "artist_genres"],
    "subgenre": ["playlist_subgenre", "subgenre", "sub_genre"],
    "popularity": ["popularity", "track_popularity"],
    "tempo": ["tempo"],
    "duration_ms": ["duration_ms", "duration"],
    "energy": ["energy"],
    "danceability": ["danceability"],
    # lyrics tidak umum di dataset ini
}


def pick_col(cols_lower: list[str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand.lower() in cols_lower:
            # temukan nama asli case-sensitive
            idx = cols_lower.index(cand.lower())
            return idx  # index, caller will map to actual name
    return None


def map_columns_auto(df: pd.DataFrame) -> pd.DataFrame:
    """Coba memetakan kolom Kaggle ke skema app.
    Mengembalikan DataFrame baru dengan kolom target.
    Kolom yang tidak ditemukan akan diisi nilai default.
    """
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]

    mapping = {}
    for target, cands in KAGGLE_TO_APP_MAP_PRIORITY.items():
        found_idx = pick_col(cols_lower, cands)
        if found_idx is not None:
            mapping[target] = cols[found_idx]

    out = pd.DataFrame()

    # judul_musik
    if "judul_musik" in mapping:
        out["judul_musik"] = df[mapping["judul_musik"]].astype(str)
    else:
        out["judul_musik"] = "(unknown title)"

    # artist
    if "artist" in mapping:
        # bisa berupa list atau string bergabung
        out["artist"] = df[mapping["artist"]].astype(str)
    else:
        out["artist"] = "(unknown artist)"

    # album
    if "album" in mapping:
        out["album"] = df[mapping["album"]].astype(str)
    else:
        out["album"] = ""

    # popularity
    if "popularity" in mapping:
        out["popularity"] = pd.to_numeric(df[mapping["popularity"]], errors="coerce")
    else:
        out["popularity"] = np.nan

    # tempo
    if "tempo" in mapping:
        out["tempo"] = pd.to_numeric(df[mapping["tempo"]], errors="coerce")
    else:
        out["tempo"] = np.nan

    # duration_ms
    if "duration_ms" in mapping:
        out["duration_ms"] = pd.to_numeric(df[mapping["duration_ms"]], errors="coerce")
    else:
        out["duration_ms"] = np.nan

    # energy
    if "energy" in mapping:
        out["energy"] = pd.to_numeric(df[mapping["energy"]], errors="coerce")
    else:
        out["energy"] = np.nan

    # danceability
    if "danceability" in mapping:
        out["danceability"] = pd.to_numeric(df[mapping["danceability"]], errors="coerce")
    else:
        out["danceability"] = np.nan

    # genre & subgenre lebih tricky karena bisa list
    raw_genre = None
    if "genre" in mapping:
        raw_genre = df[mapping["genre"]]
    elif "subgenre" in mapping:  # fallback
        raw_genre = df[mapping["subgenre"]]
    else:
        raw_genre = None

    def _split_genres(x):
        if pd.isna(x):
            return (np.nan, np.nan)
        if isinstance(x, (list, tuple)):
            g = [str(i) for i in x if str(i).strip()]
        else:
            # coba split koma atau titik dua
            s = str(x)
            # hapus bracket
            s = s.strip("[](){}")
            g = [p.strip().strip("'\"") for p in s.split(",") if p.strip()]
        if not g:
            return (np.nan, np.nan)
        g1 = g[0]
        g2 = g[1] if len(g) > 1 else g1
        return (g1, g2)

    if raw_genre is not None:
        genres, subgenres = zip(*raw_genre.apply(_split_genres))
        out["genre"] = genres
        out["subgenre"] = subgenres
    else:
        out["genre"] = np.nan
        out["subgenre"] = np.nan

    # lyrics tidak tersedia di dataset ini
    out["lyrics"] = ""

    return out


# ==========================
# Load & Clean Data (cached)
# ==========================
@st.cache_data(show_spinner=False)
def load_and_prepare_data(sample_size: int | None = None, random_state: int = 42):
    """Muat dataset dari cache parquet jika ada, jika tidak maka unduh Kaggle.
    Opsi sampling untuk mengurangi ukuran saat training.
    """
    # Jika cache parquet ada, gunakan
    if CLEAN_CACHE_PARQUET.exists():
        df_clean_full = pd.read_parquet(CLEAN_CACHE_PARQUET)
        if sample_size is not None and sample_size < len(df_clean_full):
            df_clean_full = df_clean_full.sample(n=sample_size, random_state=random_state)
        df_orig = df_clean_full.copy()  # catatan: sudah clean, jadi df & df_clean sama
        label_enc = LabelEncoder()
        label_enc.fit(["Rendah", "Tinggi"])  # placeholder, nanti ulang di bawah
        # Rebuild kategori dari popularity
        low_thresh = df_clean_full["popularity"].quantile(0.33)
        high_thresh = df_clean_full["popularity"].quantile(0.66)
        def categorize_popularity(pop):
            if pop <= low_thresh:
                return "Rendah"
            elif pop > high_thresh:
                return "Tinggi"
            return np.nan
        df_clean_full["pop_category"] = df_clean_full["popularity"].apply(categorize_popularity)
        df_clean_full = df_clean_full.dropna(subset=["pop_category"])
        df_clean_full["pop_encoded"] = LabelEncoder().fit_transform(df_clean_full["pop_category"])
        return df_orig, df_clean_full, label_enc

    # Kalau belum ada cache, coba unduh Kaggle
    zip_path = kaggle_download_if_needed()
    if zip_path is None:
        st.stop()

    csv_path = extract_and_find_csv(zip_path)
    if csv_path is None:
        st.error("CSV utama tidak ditemukan di arsip Kaggle.")
        st.stop()

    # Muat CSV (gunakan low_memory=False agar tipe lebih stabil)
    st.info("Membaca CSV besar, mohon tunggu...")
    df_raw = pd.read_csv(csv_path, low_memory=False)

    # Pemetaan kolom
    df_app = map_columns_auto(df_raw)

    # Sampling awal sebelum pembersihan untuk hemat memori
    if sample_size is not None and sample_size < len(df_app):
        df_app = df_app.sample(n=sample_size, random_state=random_state)

    # Bersih minimal sesuai kolom yang diperlukan model
    needed = ["popularity", "genre", "subgenre", "tempo", "duration_ms", "energy", "danceability"]
    df_clean = df_app.dropna(subset=needed)

    # Binning popularitas menjadi rendah dan tinggi (abaikan tengah)
    low_thresh = df_clean["popularity"].quantile(0.33)
    high_thresh = df_clean["popularity"].quantile(0.66)

    def categorize_popularity(pop):
        if pop <= low_thresh:
            return "Rendah"
        elif pop > high_thresh:
            return "Tinggi"
        return np.nan

    df_clean["pop_category"] = df_clean["popularity"].apply(categorize_popularity)
    df_clean = df_clean.dropna(subset=["pop_category"])

    label_enc = LabelEncoder()
    df_clean["pop_encoded"] = label_enc.fit_transform(df_clean["pop_category"])

    # Simpan cache parquet agar cepat pada run berikut
    try:
        df_clean.to_parquet(CLEAN_CACHE_PARQUET, index=False)
        META_CACHE_JSON.write_text(json.dumps({"rows": int(len(df_clean))}))
    except Exception:  # pragma: no cover
        pass

    return df_app, df_clean, label_enc


# ==========================
# Sidebar controls
# ==========================
with st.sidebar:
    st.markdown('<h2 style="color:#1DB954; margin-bottom: 15px;">🎵 Dashboard</h2>', unsafe_allow_html=True)
    halaman = st.radio("", ["Beranda", "Rekomendasi Musik", "Histori"], index=0, key="page_select")
    st.markdown("---")
    st.subheader("Pengaturan Data")
    max_rows_choice = st.select_slider(
        "Jumlah baris untuk dimuat (sampling)",
        options=[10_000, 25_000, 50_000, 100_000, 250_000, 500_000],
        value=50_000,
        help="Semakin besar semakin akurat namun lebih lambat dan berat memori."
    )
    st.caption("Jika cache sudah dibuat, muatan akan lebih cepat.")

# ==========================
# Load data berdasar pilihan sidebar
# ==========================
df, df_clean, label_enc = load_and_prepare_data(sample_size=max_rows_choice)

# ==========================
# Train model (cached resource)
# ==========================
@st.cache_resource(show_spinner=False)
def train_model(df_clean):
    # Vectorizer
    tfidf_genre = TfidfVectorizer()
    tfidf_subgenre = TfidfVectorizer()
    tfidf_title = TfidfVectorizer()
    tfidf_artist = TfidfVectorizer()
    tfidf_lyrics = TfidfVectorizer(max_features=500)
    tfidf_album = TfidfVectorizer()

    # Pastikan kolom ada
    for col in ["genre", "subgenre", "judul_musik", "artist", "lyrics", "album"]:
        if col not in df_clean.columns:
            df_clean[col] = ""

    genre_tfidf = tfidf_genre.fit_transform(df_clean["genre"].fillna(""))
    subgenre_tfidf = tfidf_subgenre.fit_transform(df_clean["subgenre"].fillna(""))
    title_tfidf = tfidf_title.fit_transform(df_clean["judul_musik"].fillna(""))
    artist_tfidf = tfidf_artist.fit_transform(df_clean["artist"].fillna(""))
    lyrics_tfidf = tfidf_lyrics.fit_transform(df_clean["lyrics"].fillna(""))
    album_tfidf = tfidf_album.fit_transform(df_clean["album"].fillna(""))

    df_genre = pd.DataFrame(genre_tfidf.toarray(), columns=tfidf_genre.get_feature_names_out(), index=df_clean.index)
    df_subgenre = pd.DataFrame(subgenre_tfidf.toarray(), columns=tfidf_subgenre.get_feature_names_out(), index=df_clean.index)
    df_title = pd.DataFrame(title_tfidf.toarray(), columns=tfidf_title.get_feature_names_out(), index=df_clean.index)
    df_artist = pd.DataFrame(artist_tfidf.toarray(), columns=tfidf_artist.get_feature_names_out(), index=df_clean.index)
    df_lyrics = pd.DataFrame(lyrics_tfidf.toarray(), columns=tfidf_lyrics.get_feature_names_out(), index=df_clean.index)
    df_album = pd.DataFrame(album_tfidf.toarray(), columns=tfidf_album.get_feature_names_out(), index=df_clean.index)

    features_num = ["tempo", "duration_ms", "energy", "danceability"]
    scaler = MinMaxScaler()
    df_num_scaled = pd.DataFrame(
        scaler.fit_transform(df_clean[features_num]),
        columns=features_num,
        index=df_clean.index,
    )

    X = pd.concat([df_genre, df_subgenre, df_title, df_artist, df_lyrics, df_album, df_num_scaled], axis=1)
    y = df_clean["pop_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return (
        model,
        tfidf_genre,
        tfidf_subgenre,
        tfidf_title,
        tfidf_artist,
        tfidf_lyrics,
        tfidf_album,
        scaler,
        X_test,
        y_test,
        y_pred,
        title_tfidf,
    )


model, tfidf_genre, tfidf_subgenre, tfidf_title, tfidf_artist, tfidf_lyrics, tfidf_album, scaler, X_test, y_test, y_pred, title_tfidf = train_model(df_clean)

# ==========================
# Session State
# ==========================
if "history" not in st.session_state:
    st.session_state.history = []
if "recommendation_table" not in st.session_state:
    st.session_state.recommendation_table = pd.DataFrame()

# ==========================
# Komponen UI Musik
# ==========================

def music_card(title, artist, popularity):
    st.markdown(
        f"""
        <div class="music-card">
            <div class="music-cover">🎵</div>
            <div class="music-info">
                <p class="music-title">{title}</p>
                <p class="music-artist">{artist}</p>
            </div>
            <div class="popularity">{int(popularity) if pd.notna(popularity) else '-'}"""
        + "</div></div>",
        unsafe_allow_html=True,
    )


# ==========================
# Halaman Beranda
# ==========================
if halaman == "Beranda":
    st.header("Top 10 Musik Terpopuler")
    if "popularity" in df.columns:
        top10 = df.sort_values(by="popularity", ascending=False).head(10)
        for _, row in top10.iterrows():
            music_card(row.get("judul_musik", "?"), row.get("artist", "?"), row.get("popularity", 0))
    else:
        st.info("Kolom popularity tidak tersedia.")

    st.markdown("---")
    st.header("5 Musik Terpopuler dari Setiap Genre")
    if "genre" in df.columns:
        genre_list = df["genre"].dropna().unique()[:20]  # batasi supaya UI tidak panjang
        for genre in genre_list:
            st.subheader(f"🎶 Genre: {genre}")
            top5_by_genre = (
                df[df["genre"] == genre]
                .sort_values(by="popularity", ascending=False)
                .head(5)
            )
            for _, row in top5_by_genre.iterrows():
                music_card(row.get("judul_musik", "?"), row.get("artist", "?"), row.get("popularity", 0))
    else:
        st.info("Kolom genre tidak tersedia.")

# ==========================
# Halaman Histori
# ==========================
if halaman == "Histori":
    st.header("Riwayat Pencarian Rekomendasi")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-20:]):
            st.markdown(
                f"- **{h['Judul']}** oleh {h['Artis']} (Genre: {h['Genre']}, Prediksi: {h['Prediksi']})"
            )
    else:
        st.info("Belum ada pencarian.")

    st.markdown("---")
    st.header("🎧 Hasil Rekomendasi")
    if not st.session_state.recommendation_table.empty:
        df_show = st.session_state.recommendation_table.sort_values(
            by="popularity", ascending=False
        )
        for _, row in df_show.iterrows():
            music_card(row.get("judul_musik", "?"), row.get("artist", "?"), row.get("popularity", 0))
    else:
        st.info("Belum ada rekomendasi genre ditampilkan.")

    if st.button("Reset Riwayat Pencarian"):
        st.session_state.history = []
        st.session_state.recommendation_table = pd.DataFrame()
        st.experimental_rerun()
        st.stop()

# ==========================
# Halaman Rekomendasi Musik
# ==========================
if halaman == "Rekomendasi Musik":
    st.header("Rekomendasi Musik Berdasarkan Judul")

    judul_list = df_clean["judul_musik"].dropna().unique()
    pilihan = st.selectbox("Pilih dari daftar judul musik", options=judul_list)
    manual_input = st.text_input("Atau ketik judul musik secara manual (opsional)")
    judul = manual_input if manual_input.strip() else pilihan

    if st.button("Rekomendasikan"):
        if not judul.strip():
            st.warning("Silakan masukkan judul musik terlebih dahulu.")
        else:
            # Vector judul input
            judul_vector = tfidf_title.transform([judul])
            similarities = cosine_similarity(judul_vector, title_tfidf).flatten()
            top_index = similarities.argsort()[::-1][0]
            lagu = df_clean.iloc[[top_index]]

            fitur = lagu.iloc[0]
            genre = fitur.get("genre", "?")
            subgenre = fitur.get("subgenre", "?")
            tempo = fitur.get("tempo", np.nan)
            duration_ms = fitur.get("duration_ms", np.nan)
            energy = fitur.get("energy", np.nan)
            danceability = fitur.get("danceability", np.nan)
            artist = fitur.get("artist", "?")
            album = fitur.get("album", "")
            lyrics = fitur.get("lyrics", "") if pd.notna(fitur.get("lyrics", "")) else ""
            judul_terdekat = fitur.get("judul_musik", judul)

            X_input = np.hstack([
                tfidf_genre.transform([str(genre)]).toarray(),
                tfidf_subgenre.transform([str(subgenre)]).toarray(),
                tfidf_title.transform([str(judul_terdekat)]).toarray(),
                tfidf_artist.transform([str(artist)]).toarray(),
                tfidf_lyrics.transform([str(lyrics)]).toarray(),
                tfidf_album.transform([str(album)]).toarray(),
                scaler.transform([[tempo, duration_ms, energy, danceability]]),
            ])

            pred = model.predict(X_input)[0]
            kategori = label_enc.inverse_transform([pred])[0]

            st.success(
                f"Input '{judul}' paling mirip dengan lagu '{judul_terdekat}' oleh {artist}."
            )
            st.info(f"Genre lagu tersebut adalah {genre}.")
            st.success(f"Musik ini diprediksi memiliki tingkat popularitas: {kategori}.")

            # Rekom berdasarkan genre
            if "genre" in df_clean.columns:
                df_rekom_genre = (
                    df_clean[df_clean["genre"].str.lower() == str(genre).lower()]
                    .sort_values(by="popularity", ascending=False)
                    .head(5)
                )
            else:
                df_rekom_genre = pd.DataFrame()

            st.subheader("🎧 Rekomendasi Berdasarkan Genre yang Sama")
            if not df_rekom_genre.empty:
                for _, row in df_rekom_genre.iterrows():
                    music_card(row["judul_musik"], row["artist"], row["popularity"])
                    st.caption(f"Genre: {row['genre']}")
            else:
                st.info("Tidak ada rekomendasi genre.")

            # Rekom berdasarkan kemiripan judul
            top_indices = similarities.argsort()[::-1][1:6]
            df_rekom_judul = df_clean.iloc[top_indices]
            st.subheader("🎧 Rekomendasi Berdasarkan Kemiripan Judul")
            for _, row in df_rekom_judul.sort_values(by="popularity", ascending=False).iterrows():
                music_card(row["judul_musik"], row["artist"], row["popularity"])

            # Rekom berdasarkan lirik jika ada
            if lyrics.strip():
                lyric_vector = tfidf_lyrics.transform([lyrics])
                lyric_similarities = cosine_similarity(
                    lyric_vector, tfidf_lyrics.transform(df_clean["lyrics"].fillna(""))
                ).flatten()
                top_lyric_indices = lyric_similarities.argsort()[::-1][1:6]
                df_rekom_lyrics = df_clean.iloc[top_lyric_indices]

                st.subheader("🎧 Rekomendasi Berdasarkan Kemiripan Lirik")
                for _, row in df_rekom_lyrics.sort_values(by="popularity", ascending=False).iterrows():
                    music_card(row["judul_musik"], row["artist"], row["popularity"])
            else:
                st.subheader("🎧 Rekomendasi Berdasarkan Kemiripan Lirik")
                st.info("Lagu tidak memiliki lirik untuk dibandingkan.")
                df_rekom_lyrics = pd.DataFrame()

            # Gabungkan semua rekomendasi untuk histori
            df_rekomendasi = pd.concat(
                [df_rekom_genre, df_rekom_judul, df_rekom_lyrics],
                ignore_index=True,
            ).drop_duplicates(subset="judul_musik")
            st.session_state.recommendation_table = df_rekomendasi

            st.session_state.history.append(
                {
                    "Judul": judul,
                    "Artis": artist,
                    "Genre": genre,
                    "Subgenre": subgenre,
                    "Prediksi": kategori,
                    "Rekomendasi": ", ".join(
                        df_rekomendasi["judul_musik"].head(3).tolist()
                    ),
                }
            )

# ==========================
# Info samping: cara menyiapkan kredensial Kaggle (ditampilkan di expander)
# ==========================
with st.expander("Cara menyiapkan kredensial Kaggle"):
    st.write(
        "1. Buka halaman akun Kaggle, klik Create New API Token."
        " 2. Akan terunduh file kaggle.json."
        " 3. Simpan di ~/.kaggle/kaggle.json atau set environment KAGGLE_USERNAME dan KAGGLE_KEY."
        " 4. Jalankan ulang aplikasi."
    )
    st.code(
        """# Contoh set environment sebelum run streamlit\nexport KAGGLE_USERNAME=namakamu\nexport KAGGLE_KEY=apitokenkamu\nstreamlit run app.py\n""",
        language="bash",
    )

# ==========================
# Debug opsional
# ==========================
if st.sidebar.checkbox("Tampilkan contoh data", value=False):
    st.subheader("Contoh Data Mentah (5 baris)")
    st.dataframe(df.head())
    st.subheader("Contoh Data Bersih (5 baris)")
    st.dataframe(df_clean.head())

