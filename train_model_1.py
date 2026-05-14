"""
============================================================
  🌿 Plant Leaf Disease Detection - Model Training Script
============================================================
  Uses Transfer Learning with MobileNetV2 on PlantVillage Dataset
  
  STEPS TO RUN:
  1. pip install -r requirements.txt
  2. Download PlantVillage dataset from Kaggle:
     https://www.kaggle.com/datasets/emmarex/plantdisease
  3. Extract to: ./data/PlantVillage/
  4. Run: python train_model.py
============================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
CONFIG = {
    "data_dir": r"C:\mini project\PlantVillage",
    "model_save_path": "./model/plant_disease_model.h5",
    "labels_path":     "./model/class_labels.json",
    "img_size":        (224, 224),
    "batch_size":      32,
    "epochs":          25,
    "learning_rate":   1e-4,
    "validation_split": 0.2,
    "test_split":      0.1,
    "seed":   j         42,
}

os.makedirs("./model", exist_ok=True)
os.makedirs("./logs", exist_ok=True)


# ─────────────────────────────────────────
#  1. DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────
def create_data_generators():
    print("\n📂 Loading dataset from:", CONFIG["data_dir"])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=CONFIG["validation_split"],
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=CONFIG["validation_split"],
    )

    train_gen = train_datagen.flow_from_directory(
        CONFIG["data_dir"],
        target_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        subset="training",
        seed=CONFIG["seed"],
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        CONFIG["data_dir"],
        target_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        subset="validation",
        seed=CONFIG["seed"],
        shuffle=False,
    )

    # Save class labels
    class_labels = {v: k for k, v in train_gen.class_indices.items()}
    with open(CONFIG["labels_path"], "w") as f:
        json.dump(class_labels, f, indent=2)
    print(f"✅ Found {train_gen.num_classes} classes | {train_gen.samples} training samples")

    return train_gen, val_gen, class_labels


# ─────────────────────────────────────────
#  2. MODEL ARCHITECTURE (Transfer Learning)
# ─────────────────────────────────────────
def build_model(num_classes):
    print("\n🏗️  Building MobileNetV2 Transfer Learning Model...")

    # Load pretrained MobileNetV2 (ImageNet weights)
    base_model = MobileNetV2(
        input_shape=(*CONFIG["img_size"], 3),
        include_top=False,
        weights="imagenet",
    )

    # Phase 1: Freeze base model
    base_model.trainable = False

    # Custom classification head
    inputs = tf.keras.Input(shape=(*CONFIG["img_size"], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    model.summary()
    return model, base_model


# ─────────────────────────────────────────
#  3. CALLBACKS
# ─────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            CONFIG["model_save_path"], monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        TensorBoard(log_dir="./logs", histogram_freq=1),
    ]


# ─────────────────────────────────────────
#  4. FINE-TUNING (Phase 2)
# ─────────────────────────────────────────
def fine_tune(model, base_model, train_gen, val_gen):
    print("\n🔧 Phase 2: Fine-tuning top layers of base model...")

    # Unfreeze top 40 layers
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["learning_rate"] / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    history_fine = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=get_callbacks(),
    )
    return history_fine


# ─────────────────────────────────────────
#  5. TRAINING
# ─────────────────────────────────────────
def train(model, train_gen, val_gen):
    print("\n🚀 Starting Training - Phase 1 (Frozen base)...")
    history = model.fit(
        train_gen,
        epochs=CONFIG["epochs"],
        validation_data=val_gen,
        callbacks=get_callbacks(),
    )
    return history


# ─────────────────────────────────────────
#  6. PLOT RESULTS
# ─────────────────────────────────────────
def plot_history(history, fine_history=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History - Plant Disease Detection", fontsize=14)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    if fine_history:
        acc += fine_history.history["accuracy"]
        val_acc += fine_history.history["val_accuracy"]
        loss += fine_history.history["loss"]
        val_loss += fine_history.history["val_loss"]

    epochs_range = range(len(acc))

    axes[0].plot(epochs_range, acc, label="Train Accuracy", color="#2ecc71")
    axes[0].plot(epochs_range, val_acc, label="Val Accuracy", color="#e74c3c")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, loss, label="Train Loss", color="#2ecc71")
    axes[1].plot(epochs_range, val_loss, label="Val Loss", color="#e74c3c")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./model/training_history.png", dpi=150)
    print("📊 Training plot saved to ./model/training_history.png")
    plt.show()


# ─────────────────────────────────────────
#  7. EVALUATE
# ─────────────────────────────────────────
def evaluate(model, val_gen, class_labels):
    print("\n📊 Evaluating model on validation set...")
    loss, acc, top3 = model.evaluate(val_gen, verbose=1)
    print(f"\n✅ Validation Accuracy : {acc*100:.2f}%")
    print(f"✅ Top-3 Accuracy       : {top3*100:.2f}%")
    print(f"✅ Validation Loss      : {loss:.4f}")

    # Confusion Matrix
    print("\n🔍 Generating predictions for confusion matrix...")
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes
    labels = [class_labels[str(i)] for i in range(len(class_labels))]

    report = classification_report(y_true, y_pred, target_names=labels)
    print("\n📋 Classification Report:\n", report)

    with open("./model/classification_report.txt", "w") as f:
        f.write(report)

    # Plot confusion matrix (top 10 classes for visibility)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig("./model/confusion_matrix.png", dpi=120)
    print("📊 Confusion matrix saved to ./model/confusion_matrix.png")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  🌿 Plant Leaf Disease Detection - Training Pipeline")
    print("=" * 60)

    if not Path(CONFIG["data_dir"]).exists():
        print(f"\n❌ Dataset not found at: {CONFIG['data_dir']}")
        print("   Please download PlantVillage from:")
        print("   https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("   And extract to: ./data/PlantVillage/")
        exit(1)

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n💻 GPU available: {len(gpus) > 0} ({len(gpus)} device(s))")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # Train
    train_gen, val_gen, class_labels = create_data_generators()
    model, base_model = build_model(num_classes=len(class_labels))
    history = train(model, train_gen, val_gen)
    fine_history = fine_tune(model, base_model, train_gen, val_gen)
    plot_history(history, fine_history)
    evaluate(model, val_gen, class_labels)

    print("\n✅ Training complete! Model saved to:", CONFIG["model_save_path"])
    print("🚀 Now run: python app.py to start the web application")
