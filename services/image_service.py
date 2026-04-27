import io
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from sqlalchemy.orm import Session

from models import ImageDB, Prediction

BASE_DIR = Path(__file__).resolve().parent.parent
TIF_SUPPORT = True

print("Загрузка модели...")
model = tf.keras.models.load_model(BASE_DIR / "model" / "model.keras")
print("Модель загружена")

def preprocess_image(image_data: bytes, filename: str = ""):
    if TIF_SUPPORT and filename.lower().endswith((".tif", ".tiff")):
        img_array = imageio.imread(io.BytesIO(image_data))
    else:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(image)

    img_array = cv2.resize(img_array, (128, 128))
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif len(img_array.shape) == 3 and img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array.astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_earlier_conv_layer_name(model):
    conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    if not conv_layers:
        raise ValueError("Нет Conv2D слоев")
    return conv_layers[min(1, len(conv_layers) - 1)].name

def build_grad_model(model):
    try:
        last_conv_layer_name = get_earlier_conv_layer_name(model)
        target_conv_layer = model.get_layer(last_conv_layer_name)
        input_tensor = layers.Input(shape=(128, 128, 3))
        x = input_tensor
        conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == target_conv_layer.name:
                conv_output = x
                break
        if conv_output is None:
            return None
        return Model(inputs=input_tensor, outputs=[conv_output, x])
    except Exception:
        return None

grad_model = build_grad_model(model)

def make_gradcam_heatmap(img_array, grad_model):
    if grad_model is None:
        return None
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_outputs[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-7)
    return heatmap.numpy()

def save_history(db: Session, file_bytes: bytes, filename: str, probability: float, label: str, user_id=None):
    uploads_dir = BASE_DIR / "output" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(filename).suffix.lower() or ".png"
    safe_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
    file_path = uploads_dir / safe_name

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    img_row = ImageDB(user_id=user_id, file_path=str(file_path))
    db.add(img_row)
    db.commit()
    db.refresh(img_row)

    pred_row = Prediction(image_id=img_row.id, probability=probability, result=label)
    db.add(pred_row)
    db.commit()

    return img_row, pred_row

def save_gradcam_result(image_path: str, heatmap):
    output_path = BASE_DIR / "output" / "gradcam_result.png"
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    original = Image.open(image_path).convert("RGB")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return "/output/gradcam_result.png"