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
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array, verbose=0)[0]
    
    conf = predictions * 100
    pred_class = classes[np.argmax(predictions)]
    
    probs = {
        'glioma': round(float(conf[0]), 2),
        'meningioma': round(float(conf[1]), 2),
        'notumor': round(float(conf[2]), 2),
        'pituitary': round(float(conf[3]), 2)
    }
    
    return pred_class, round(max(conf), 2), probs


            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})


