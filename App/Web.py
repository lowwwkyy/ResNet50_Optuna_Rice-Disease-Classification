import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

st.set_page_config(page_title="Rice Leaf Disease Classifier", layout="centered")

st.title("🌾 Rice Leaf Disease Classification")
st.write("Upload a rice leaf image and let the model predict the disease class.")

@st.cache_resource
def load_trained_model():
    model_path = "Z:/Okky/BINUSIAN/Semester 5/Deep Learning/FINAL PROJECT/Outputs/best_resnet50_final.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return load_model(model_path)

model = load_trained_model()

class_names = ['Bacterial Leaf Blight', "Healthy _leaf", 'Rice', 'Rice Blast', 'Tungro']

def predict_image(image):
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(preds)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(preds))
    
    return predicted_class, confidence

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        predicted_class, confidence = predict_image(uploaded_file)

    st.success("Prediction Complete!")
    st.markdown(f"### 🧠 Predicted Class: `{predicted_class}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2%}`")
