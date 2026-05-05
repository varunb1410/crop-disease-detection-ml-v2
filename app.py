from cProfile import label

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("plant_disease_model_final.keras")

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

with open("disease_info.json") as f:
    disease_info = json.load(f)

IMG_SIZE = (128, 128)

def predict_image(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    top3_idx = prediction.argsort()[-3:][::-1]

    results = []
    for i in top3_idx:
        label = class_names[i]
        display_label = label.replace("___", " : ")

        info = disease_info.get(label, "No info available")

        results.append({
            "class": display_label,
            "confidence": round(float(prediction[i]) * 100, 2),
            "info": info
        })

    return results

@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            image = Image.open(file)
            results = predict_image(image)

    return render_template("index.html", results=results)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)