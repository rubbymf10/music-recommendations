import os
import gdown
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set judul aplikasi
st.set_page_config(page_title="Sistem Rekomendasi Musik", layout="wide")
st.title("🎵 Sistem Rekomendasi Musik Berdasarkan Popularitas (Random Forest)")

# -----------------------------
# FUNGSI MEMUAT DATASET
# -----------------------------
@st.cache_data
def load_data():
    # ID dan URL file Google Drive
    file_id = "1yzq3kWl2TiFghG0DoPNswJEjPdfGYB66"
    file_url = f"https://drive.google.com/uc?id={file_id}"
    local_file = "musik.csv"

    # Unduh file jika belum tersedia
    if not os.path.exists(local_file):
        with st.spinner("📥 Mengunduh dataset dari Google Drive..."):
            gdown.download(file_url, local_file, quiet=False)

    # Load data
    df = pd.read_csv(local_file)

    # Drop data kosong
    df_clean = df.dropna(subset=['popularity', 'genre', 'subgenre', 'tempo', 'duration_ms', 'energy', 'danceability'])

    # Kategori popularitas berdasarkan kuartil
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

    # Encoding
    label_enc = LabelEncoder()
    df_clean['pop_encoded'] = label_enc.fit_transform(df_clean['pop_category'])

    return df, df_clean, label_enc

# -----------------------------
# PREPROCESSING & MODELING
# -----------------------------
def process_and_train(df_clean):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_clean['lyrics'].fillna(""))

    numerik = df_clean[['tempo', 'duration_ms', 'energy', 'danceability']]
    scaler = MinMaxScaler()
    numerik_scaled = scaler.fit_transform(numerik)

    # Gabungkan fitur
    from scipy.sparse import hstack
    X = hstack([tfidf_matrix, numerik_scaled])
    y = df_clean['pop_encoded']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, cm, report

# -----------------------------
# MAIN
# -----------------------------
df, df_clean, label_enc = load_data()

st.subheader("Contoh Data")
st.dataframe(df_clean.head())

# Training model
model, cm, report = process_and_train(df_clean)

# Visualisasi Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)

# Tampilkan classification report
st.subheader("Classification Report")
st.json(report)

# Fitur input rekomendasi
st.subheader("Coba Prediksi Popularitas Lagu")
judul = st.text_input("Judul Lagu")
lyrics = st.text_area("Lirik Lagu")
tempo = st.slider("Tempo", 50, 250, 120)
duration = st.slider("Durasi (ms)", 30000, 400000, 180000)
energy = st.slider("Energi", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)

if st.button("Prediksi"):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_clean['lyrics'].fillna("")).toarray()

    # Buat vektor baru
    tfidf_new = TfidfVectorizer(stop_words='english', vocabulary=tfidf.vocabulary_)
    vektor_lirik = tfidf_new.fit_transform([lyrics])

    numerik = np.array([[tempo, duration, energy, danceability]])
    scaler = MinMaxScaler()
    scaler.fit(df_clean[['tempo', 'duration_ms', 'energy', 'danceability']])
    numerik_scaled = scaler.transform(numerik)

    from scipy.sparse import hstack
    X_new = hstack([vektor_lirik, numerik_scaled])

    pred = model.predict(X_new)
    kategori = label_enc.inverse_transform(pred)[0]

    st.success(f"🎧 Lagu diprediksi memiliki tingkat popularitas: **{kategori}**")

