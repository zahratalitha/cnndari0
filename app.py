import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# -----------------------------
# Konfigurasi
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 180, 180

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="zahratalitha/anjingkucing",
        filename="best_model.h5"
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# -----------------------------
# Preprocessing Gambar
# -----------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(
    page_title="Klasifikasi Anjing vs Kucing",
    page_icon="🐾",
    layout="centered"
)

st.title("🐶🐱 Klasifikasi Anjing vs Kucing")
st.write("Upload gambar hewanmu, dan model akan memprediksi apakah itu Anjing atau Kucing.")

uploaded_file = st.file_uploader("📤 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)

    # Preprocess & prediksi
    img_array = preprocess_image(uploaded_file)
    pred = model.predict(img_array)

    # Tentukan label dan confidence
    if pred.shape[1] == 1:
        prob = float(pred[0][0])
        label = "🐱 Kucing" if prob < 0.5 else "🐶 Anjing"
        confidence = 100 * (1 - prob) if prob < 0.5 else 100 * prob
    else:
        score = tf.nn.softmax(pred[0]).numpy()
        label = ["🐱 Kucing", "🐶 Anjing"][np.argmax(score)]
        confidence = 100 * np.max(score)

    # -----------------------------
    # Tampilkan hasil
    # -----------------------------
    st.markdown("---")
    st.subheader("📌 Hasil Prediksi")
    
    # Label prediksi ditampilkan besar dan berwarna hijau
    st.markdown(
        f"<h2 style='text-align:center; color:#4CAF50;'>{label}</h2>",
        unsafe_allow_html=True
    )

    # Confidence ditampilkan dengan progress bar
    st.write(f"Tingkat keyakinan model:")
    st.progress(int(confidence))
    st.write(f"**{confidence:.2f}%**")
