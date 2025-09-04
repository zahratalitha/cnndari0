import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


img_height, img_width = 180, 180

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="zahratalitha/anjingkucing",
        filename="best_model.h5"
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🐶🐱 Klasifikasi Anjing vs Kucing")
st.write("Upload gambar:")

uploaded_file = st.file_uploader("Upload gambar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)

    img_array = preprocess_image(uploaded_file)

    pred = model.predict(img_array)

    if pred.shape[1] == 1:
        prob = float(pred[0][0])
        label = "Kucing 🐱" if prob < 0.5 else "Anjing 🐶"
        confidence = 100 * (1 - prob) if prob < 0.5 else 100 * prob
    else:
        score = tf.nn.softmax(pred[0]).numpy()
        label = ["Kucing 🐱", "Anjing 🐶"][np.argmax(score)]
        confidence = 100 * np.max(score)

    st.subheader("📌 Hasil Prediksi")
    st.write(f"Model memprediksi gambar ini adalah **{label}** dengan probabilitas **{confidence:.2f}%**")
