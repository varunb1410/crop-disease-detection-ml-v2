from cProfile import label

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
from PIL import Image

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# Load model
model = None

def load_model_once():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(
            "final_model.h5",
            compile=False
        )

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

with open("disease_info.json") as f:
    disease_info = json.load(f)

IMG_SIZE = (128, 128)

def predict_image(img):
    try:
        load_model_once()
        print("Model loaded")

        img = img.convert("RGB")
        img = img.resize((128, 128))
        print("Image processed")

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        print("Prediction done")

        top3_idx = prediction.argsort()[-3:][::-1]

        results = []
        for i in top3_idx:
            label = class_names[i]
            display_label = label.replace("___", " : ").replace("_", " ")

            results.append({
                "class": display_label,
                "confidence": round(float(prediction[i]) * 100, 2),
                "info": disease_info.get(label, "No info available")
            })

        return results

    except Exception as e:
        print("ERROR:", str(e))
        return [{
            "class": "Error",
            "confidence": 0,
            "info": str(e)
        }]

@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("index.html", results=[{
                "class": "No file selected",
                "confidence": 0,
                "info": "Please upload an image"
            }])

        try:
            image = Image.open(file.stream)
            results = predict_image(image)
        except Exception as e:
            results = [{
                "class": "Upload Error",
                "confidence": 0,
                "info": str(e)
            }]

    return render_template("index.html", results=results)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)