import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "xray_multilabel_model.keras"
DISEASES = [
    "Cardiomegaly",
    "Edema",
    "Effusion",
    "Pneumonia",
    "Atelectasis",
    "Consolidation",
    "No Finding"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("Chest X-ray Differential Diagnosis (Multilabel)")
st.write("Upload an X-ray image to get predictions.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    model = load_model()
    img = preprocess(image)

    preds = model.predict(img)[0]

    st.subheader("Predictions")
    for disease, score in zip(DISEASES, preds):
        st.write(f"{disease}: {score:.3f}")
