from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)

MODEL_PATH = "models/best_model.h5"
model = load_model(MODEL_PATH)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def preprocess_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    file_path = "temp.jpg"
    file.save(file_path)

    try:
        img = preprocess_img(file_path)
        preds = model.predict(img)[0]

        os.remove(file_path)

        pred_index = np.argmax(preds)
        confidence = round(float(preds[pred_index] * 100), 2)

        output = CLASS_NAMES[pred_index]

        return jsonify({
            "prediction": output,
            "confidence": confidence,
            "probabilities": {
                "glioma": round(float(preds[0] * 100), 2),
                "meningioma": round(float(preds[1] * 100), 2),
                "notumor": round(float(preds[2] * 100), 2),
                "pituitary": round(float(preds[3] * 100), 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
