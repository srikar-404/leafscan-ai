"""
app.py - Flask Web Application for Plant Disease Detection
Run: python app.py
Then open: http://localhost:5000
"""

import os
import json
import uuid
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
from predict import get_disease_info, predict as run_predict

# ─────────────────────────────────────────
#  APP CONFIGURATION
# ─────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max
app.config["UPLOAD_FOLDER"] = "./static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ─────────────────────────────────────────
#  LOAD MODEL (at startup)
# ─────────────────────────────────────────
MODEL = None
CLASS_LABELS = None
MODEL_PATH = "./model/plant_disease_model.h5"
LABELS_PATH = "./model/class_labels.json"


def load_resources():
    global MODEL, CLASS_LABELS
    if Path(MODEL_PATH).exists() and Path(LABELS_PATH).exists():
        print("📦 Loading model into Flask app...")
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            CLASS_LABELS = json.load(f)
        print(f"✅ Model ready | {len(CLASS_LABELS)} disease classes")
    else:
        print("⚠️  Model not found. Using demo mode.")
        print("   Run train_model.py to train the model first.")


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def demo_prediction():
    """Returns a mock prediction when model isn't loaded."""
    import random
    demo_diseases = [
        "Tomato___Early_blight",
        "Apple___Apple_scab",
        "Corn_(maize)___Common_rust_",
        "Tomato___healthy",
        "Potato___Late_blight",
    ]
    top = random.choice(demo_diseases)
    results = [{"class": top, "confidence": round(random.uniform(70, 98), 2), "index": 0}]
    for _ in range(4):
        results.append({
            "class": random.choice(demo_diseases),
            "confidence": round(random.uniform(1, 30), 2),
            "index": 1
        })
    return sorted(results, key=lambda x: x["confidence"], reverse=True)


# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, or JPEG."}), 400

    # Save file
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Predict
    if MODEL is not None and CLASS_LABELS is not None:
        img_array = preprocess_image(filepath)
        results = run_predict(MODEL, img_array, CLASS_LABELS, top_k=5)
    else:
        results = demo_prediction()

    # Get disease info
    top = results[0]
    disease_info = get_disease_info(top["class"])
    is_healthy = "healthy" in top["class"].lower()

    response = {
        "status": "success",
        "image_url": f"/static/uploads/{filename}",
        "top_prediction": {
            "class": top["class"],
            "display_name": top["class"].replace("___", " → ").replace("_", " "),
            "confidence": round(top["confidence"], 2),
            "is_healthy": is_healthy,
        },
        "all_predictions": [
            {
                "class": r["class"],
                "display_name": r["class"].replace("___", " → ").replace("_", " "),
                "confidence": round(r["confidence"], 2),
            }
            for r in results
        ],
        "disease_info": {
            "severity": disease_info.get("severity", "Unknown"),
            "cause": disease_info.get("cause", "Unknown pathogen"),
            "symptoms": disease_info.get("symptoms", "Visible abnormalities on leaves."),
            "treatment": disease_info.get("treatment", disease_info.get("description", "")),
            "prevention": disease_info.get("prevention", "Regular monitoring and crop hygiene."),
        },
        "demo_mode": MODEL is None,
    }

    return jsonify(response)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "classes": len(CLASS_LABELS) if CLASS_LABELS else 0,
    })


@app.route("/classes")
def get_classes():
    if CLASS_LABELS:
        return jsonify({"classes": list(CLASS_LABELS.values()), "count": len(CLASS_LABELS)})
    return jsonify({"classes": [], "count": 0, "note": "Model not loaded"})


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    load_resources()
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
