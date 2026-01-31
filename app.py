import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
import onnxruntime as ort

MODEL_PATH = "xray_model.onnx"
DRIVE_LINK = "https://drive.google.com/uc?export=download&id=1xXM-jVyHdMYhkH-p36qw9nHknuGTA5RK"

DISEASES = [
    "Cardiomegaly",
    "Edema",
    "Effusion",
    "Pneumonia",
    "Atelectasis",
    "Consolidation",
    "No Finding"
]

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("📥 Downloading model…")
        gdown.download(DRIVE_LINK, MODEL_PATH, quiet=False)
        st.write("✅ Model downloaded")

@st.cache_resource
def load_model():
    download_model()
    return ort.InferenceSession(MODEL_PATH)

def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

st.title("🫁 Chest X-ray Differential Diagnosis (Multilabel)")
st.write("Upload a chest X-ray image to get predictions.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    model = load_model()
    img = preprocess(image)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    preds = model.run([output_name], {input_name: img})[0][0]

    st.subheader("Predictions")
    for disease, score in zip(DISEASES, preds):
        st.write(f"{disease}: **{score:.3f}**")


