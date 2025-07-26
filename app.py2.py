import gradio as gr
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import json

# ✅ Load trained model
model = load_model("leaf_disease_model.h5")

# ✅ Load disease information JSON
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# ✅ Get class labels from JSON keys
class_names = list(disease_info.keys())

# ✅ Define prediction function
def predict_disease(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = round(100 * np.max(prediction), 2)

    # Disease details
    description = disease_info[predicted_label]["description"]
    spread = disease_info[predicted_label]["spread"]
    cure = disease_info[predicted_label]["cure"]

    result = f"✅ Prediction: {predicted_label}\n📊 Confidence: {confidence}%\n\n🩺 Description: {description}\n🌿 Spread: {spread}\n💊 Cure: {cure}"
    return result

# ✅ Create Gradio UI
input_image = gr.Image(type="pil", label="Upload Leaf Image")
output_text = gr.Textbox(label="Disease Report")

gr.Interface(
    fn=predict_disease,
    inputs=input_image,
    outputs=output_text,
    title="🌱 Leaf Disease Detection AI Tool",
    description="Upload a leaf image to predict plant disease. You'll get its name, description, spreading condition & cure."
).launch()