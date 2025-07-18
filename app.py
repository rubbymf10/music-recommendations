import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Fungsi memuat dataset dari Google Drive
@st.cache_data
def load_data():
    file_id = "1yzq3kWl2TiFghG0DoPNswJEjPdfGYB66"
    file_url = f"https://drive.google.com/uc?id={file_id}"
    local_file = "musik.csv"

    if not os.path.exists(local_file):
        with st.spinner("Mengunduh dataset dari Google Drive..."):
            gdown.download(file_url, local_file, quiet=False)

    df = pd.read_csv(local_file)

    # Hapus data kosong pada kolom penting
    df_clean = df.dropna(subset=[
        'popularity', 'genre', 'subgenre', 'tempo',
        'duration_ms', 'energy', 'danceability'
    ])

    # Kategorisasi popularitas
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

    # Encoding target
    label_enc = LabelEncoder()
    df_clean['pop_encoded'] = label_enc.fit_transform(df_clean['pop_category'])

    return df, df_clean, label_enc

# -------------------------------
# Fungsi training model
def train_model(df_clean):
    tfidf = TfidfVectorizer()
    text_features = tfidf.fit_transform(
        df_clean['judul_musik'].astype(str) + " " +
        df_clean['artist'].astype(str) + " " +
        df_clean['genre'].astype(str)
    )

    numeric_features = df_clean[['tempo', 'duration_ms', 'energy', 'danceability']].values
    X = np.hstack((text_features.toarray(), numeric_features))
    y = df_clean['pop_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# -------------------------------
# Aplikasi Streamlit
st.title("🎵 Sistem Rekomendasi Musik Berdasarkan Popularitas")
st.write("Menggunakan TF-IDF dan Random Forest")

df_raw, df_clean, label_encoder = load_data()

if st.checkbox("Tampilkan data mentah"):
    st.dataframe(df_raw)

model, X_test, y_test = train_model(df_clean)
y_pred = model.predict(X_test)

# Evaluasi
st.subheader("📊 Evaluasi Model")
st.text("Confusion Matrix:")
st.text(confusion_matrix(y_test, y_pred))
st.text("\nClassification Report:")
st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
