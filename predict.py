"""
predict.py - Standalone prediction utility
Usage: python predict.py --image path/to/leaf.jpg
"""

import argparse
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Disease info database
DISEASE_INFO = {
    "healthy": {
        "severity": "None",
        "description": "The plant appears healthy with no visible signs of disease.",
        "treatment": "No treatment needed. Continue regular care.",
        "color": "#27ae60"
    },
    "default": {
        "severity": "Moderate",
        "description": "Disease detected. Consult the treatment guide below.",
        "treatment": "Apply appropriate fungicide/pesticide. Remove affected leaves. Improve air circulation.",
        "color": "#e74c3c"
    }
}

DISEASE_TREATMENTS = {
    "Apple___Apple_scab": {
        "severity": "High",
        "cause": "Fungus: Venturia inaequalis",
        "symptoms": "Dark, scabby lesions on leaves and fruit.",
        "treatment": "Apply fungicides (captan, mancozeb). Remove fallen leaves. Plant resistant varieties.",
        "prevention": "Prune for air circulation. Avoid overhead irrigation."
    },
    "Apple___Black_rot": {
        "severity": "High",
        "cause": "Fungus: Botryosphaeria obtusa",
        "symptoms": "Brown lesions with purple borders, mummified fruit.",
        "treatment": "Prune infected branches. Apply copper-based fungicide. Remove mummified fruit.",
        "prevention": "Maintain tree vigor. Remove dead wood promptly."
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Moderate",
        "cause": "Fungus: Puccinia sorghi",
        "symptoms": "Brick-red pustules scattered across both leaf surfaces.",
        "treatment": "Apply fungicides (azoxystrobin, propiconazole) at early infection.",
        "prevention": "Plant resistant hybrids. Early planting to avoid peak rust periods."
    },
    "Tomato___Early_blight": {
        "severity": "Moderate",
        "cause": "Fungus: Alternaria solani",
        "symptoms": "Dark spots with concentric rings (target-board appearance).",
        "treatment": "Apply chlorothalonil or copper fungicide every 7-10 days.",
        "prevention": "Crop rotation. Remove infected plant debris. Mulch around plants."
    },
    "Tomato___Late_blight": {
        "severity": "Very High",
        "cause": "Oomycete: Phytophthora infestans",
        "symptoms": "Water-soaked lesions, white mold on undersides, rapid plant death.",
        "treatment": "Apply mancozeb or metalaxyl immediately. Remove infected plants.",
        "prevention": "Avoid overhead watering. Use resistant varieties. Monitor during cool, wet weather."
    },
    "Potato___Late_blight": {
        "severity": "Very High",
        "cause": "Oomycete: Phytophthora infestans",
        "symptoms": "Dark, water-soaked lesions on leaves and tubers.",
        "treatment": "Apply fungicides (mefenoxam, cymoxanil). Destroy infected tubers.",
        "prevention": "Use certified seed potatoes. Hill up soil around plants."
    },
    "Grape___Black_rot": {
        "severity": "High",
        "cause": "Fungus: Guignardia bidwellii",
        "symptoms": "Tan lesions with dark borders, shriveled mummified berries.",
        "treatment": "Apply mancozeb or myclobutanil at early season.",
        "prevention": "Remove mummified berries. Prune for air circulation."
    },
}


def load_model_and_labels(model_path="./model/plant_disease_model.h5",
                          labels_path="./model/class_labels.json"):
    """Load trained model and class labels."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first!"
        )
    print("📦 Loading model...")
    model = tf.keras.models.load_model(model_path)

    with open(labels_path) as f:
        class_labels = json.load(f)

    print(f"✅ Model loaded | {len(class_labels)} classes")
    return model, class_labels


def preprocess_image(img_path, img_size=(224, 224)):
    """Load and preprocess image for prediction."""
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, img


def predict(model, img_array, class_labels, top_k=5):
    """Run prediction and return top-k results."""
    preds = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:top_k]

    results = []
    for idx in top_indices:
        label = class_labels[str(idx)]
        confidence = float(preds[idx]) * 100
        results.append({"class": label, "confidence": confidence, "index": int(idx)})

    return results


def get_disease_info(class_name):
    """Get treatment info for a disease."""
    if "healthy" in class_name.lower():
        return DISEASE_INFO["healthy"]
    info = DISEASE_TREATMENTS.get(class_name, {
        "severity": "Moderate",
        "cause": "Fungal/Bacterial pathogen",
        "symptoms": "Visible lesions, discoloration, or spots on leaves.",
        "treatment": "Apply broad-spectrum fungicide. Remove infected leaves. Ensure proper drainage.",
        "prevention": "Regular monitoring. Crop rotation. Remove plant debris."
    })
    return info


def display_results(img_path, results):
    """Display prediction results with visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    # Left: Original image
    ax1 = axes[0]
    img = plt.imread(img_path)
    ax1.imshow(img)
    ax1.set_title("Input Leaf Image", color="white", fontsize=13, pad=10)
    ax1.axis("off")

    top_result = results[0]
    is_healthy = "healthy" in top_result["class"].lower()
    border_color = "#27ae60" if is_healthy else "#e74c3c"

    for spine in ax1.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)

    # Right: Top-5 predictions bar chart
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")
    labels = [r["class"].replace("___", "\n").replace("_", " ") for r in results]
    confs = [r["confidence"] for r in results]
    colors = ["#27ae60" if "healthy" in l.lower() else "#e74c3c" for l in labels]
    colors[0] = "#f39c12"  # Top prediction in gold

    bars = ax2.barh(labels[::-1], confs[::-1], color=colors[::-1], height=0.5)
    for bar, conf in zip(bars, confs[::-1]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{conf:.1f}%", va="center", ha="left", color="white", fontsize=10)

    ax2.set_xlim(0, 115)
    ax2.set_title("Top-5 Predictions", color="white", fontsize=13, pad=10)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.set_xlabel("Confidence (%)", color="#aaa")
    ax2.spines[:].set_color("#333")

    # Title
    disease_name = top_result["class"].replace("___", " - ").replace("_", " ")
    status = "✅ HEALTHY" if is_healthy else "⚠️ DISEASE DETECTED"
    fig.suptitle(
        f"{status}: {disease_name}\nConfidence: {top_result['confidence']:.2f}%",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    output_path = Path(img_path).stem + "_prediction.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"\n📊 Visualization saved: {output_path}")
    plt.show()


def print_report(results):
    """Print formatted prediction report to console."""
    print("\n" + "=" * 60)
    print("  🌿 PLANT DISEASE DETECTION REPORT")
    print("=" * 60)

    top = results[0]
    info = get_disease_info(top["class"])
    disease_name = top["class"].replace("___", " → ").replace("_", " ")

    print(f"\n🔍 PRIMARY DIAGNOSIS: {disease_name}")
    print(f"📊 Confidence       : {top['confidence']:.2f}%")

    if isinstance(info, dict):
        print(f"⚠️  Severity         : {info.get('severity', 'Unknown')}")
        print(f"\n📋 DETAILS:")
        for key in ["cause", "symptoms", "treatment", "prevention", "description"]:
            if key in info:
                print(f"   {key.capitalize():12}: {info[key]}")

    print(f"\n📊 TOP-5 PREDICTIONS:")
    for i, r in enumerate(results):
        bar = "█" * int(r["confidence"] / 5)
        print(f"   {i+1}. {r['class']:40} {r['confidence']:6.2f}%  {bar}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Predictor")
    parser.add_argument("--image", required=True, help="Path to leaf image")
    parser.add_argument("--model", default="./model/plant_disease_model.h5")
    parser.add_argument("--labels", default="./model/class_labels.json")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    model, class_labels = load_model_and_labels(args.model, args.labels)
    img_array, _ = preprocess_image(args.image)
    results = predict(model, img_array, class_labels, args.top_k)
    print_report(results)
    display_results(args.image, results)
