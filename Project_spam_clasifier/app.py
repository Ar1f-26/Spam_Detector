import streamlit as st
import joblib

# Load Model & Vectorizer yang sudah ditraining
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Judul Web UI
st.title("🛡️ Detektor Pesan Spam AI")
st.write("Masukkan pesan di bawah ini untuk memeriksa apakah pesan tersebut termasuk **Spam** atau **Bukan (Ham)**.")

# Input text dari pengguna
user_text = st.text_area("Teks Pesan:", placeholder="Contoh: Selamat! Anda terpilih mendapatkan hadiah uang tunai 10 juta rupiah...")

# Tombol prediksi
if st.button("Identifikasi Pesan", type="primary"):
    if user_text.strip():
        # Ubah teks menjadi vektor TF-IDF
        text_vectorized = vectorizer.transform([user_text])
        
        # Prediksi
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Tampilkan Hasil
        if prediction == 1:
            confidence = round(probabilities[1] * 100, 2)
            st.error(f"🚨 **Hasil Prediksi: SPAM** (Tingkat Keyakinan: {confidence}%)")
        else:
            confidence = round(probabilities[0] * 100, 2)
            st.success(f"✅ **Hasil Prediksi: HAM / Bukan Spam** (Tingkat Keyakinan: {confidence}%)")
    else:
        st.warning("⚠️ Mohon masukkan teks pesan terlebih dahulu!")
