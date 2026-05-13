from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

# Reduce TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# Image size
IMG_SIZE = (128, 128)

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# Load disease information
with open("disease_info.json") as f:
    disease_info = json.load(f)

# Load model once
model = None

def load_model_once():
    global model

    if model is None:
        print("Loading TensorFlow model...")
        
        model = tf.keras.models.load_model(
            "final_model.h5",
            compile=False
        )

        print("Model loaded successfully")


# Prediction function
def predict_image(img):

    load_model_once()

    # Image preprocessing
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    # Get top 3 predictions
    top3_idx = prediction.argsort()[-3:][::-1]

    results = []

    for i in top3_idx:

        label = class_names[i]

        display_label = (
            label.replace("___", " : ")
                 .replace("_", " ")
        )

        results.append({
            "class": display_label,
            "confidence": round(float(prediction[i]) * 100, 2),
            "info": disease_info.get(
                label,
                "No disease information available."
            )
        })

    return results


# Home route
@app.route("/", methods=["GET", "POST"])
def index():

    results = None

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":

            results = [{
                "class": "No File Selected",
                "confidence": 0,
                "info": "Please upload an image."
            }]

            return render_template(
                "index.html",
                results=results
            )

        try:

            image = Image.open(file.stream)

            results = predict_image(image)

        except Exception as e:

            print("Prediction Error:", str(e))

            results = [{
                "class": "Error",
                "confidence": 0,
                "info": str(e)
            }]

    return render_template(
        "index.html",
        results=results
    )


# Run app
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )