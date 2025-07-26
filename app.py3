import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import json

# Load model
model = load_model("leaf_disease_model.h5")

# Load class names and disease info
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___healthy']  # Replace with yours
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# App UI
st.title("🌿 Leaf Disease Detection AI")
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    st.success(f"✅ Prediction: {predicted_class}")
    st.info(f"📊 Confidence: {confidence}%")

    # Show disease info
    if predicted_class in disease_info:
        st.markdown(f"### 🦠 Description:\n{disease_info[predicted_class]['description']}")
        st.markdown(f"### 🌍 Spread:\n{disease_info[predicted_class]['spread']}")
        st.markdown(f"### 💊 Cure:\n{disease_info[predicted_class]['cure']}")