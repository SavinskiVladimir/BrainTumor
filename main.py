from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import tensorflow as tf
import io
from pathlib import Path
import cv2
from tensorflow.keras import layers, Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import traceback
import imageio.v3 as imageio

TIF_SUPPORT = True


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()

# Создаем папки
os.makedirs("output", exist_ok=True)

print("Загружается модель...")
model = tf.keras.models.load_model(BASE_DIR / "model" / "model.keras")
print("Модель загружена")

def make_gradcam_heatmap(img_array, grad_model):
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

def get_earlier_conv_layer_name(model):
    conv_layers = []
    print("Архитектура модели:")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.Conv2D):
            conv_layers.append(layer)
            try:
                config = layer.get_config()
                print(f"  Conv2D #{len(conv_layers)}: {layer.name}, filters={config['filters']}")
            except:
                print(f"  Conv2D #{len(conv_layers)}: {layer.name}")

    if not conv_layers:
        raise ValueError("Нет Conv2D слоев!")
    if len(conv_layers) == 1:
        print("Используется единственный Conv2D")
        return conv_layers[0].name
    print(f"Используется 2-й Conv2D: {conv_layers[1].name}")
    return conv_layers[1].name

print("Инициализация Grad-CAM...")
grad_model = None
try:
    last_conv_layer_name = get_earlier_conv_layer_name(model)
    target_conv_layer = model.get_layer(last_conv_layer_name)

    input_tensor = layers.Input(shape=(128, 128, 3))
    x = input_tensor
    for layer in model.layers:
        x = layer(x)
        if layer.name == target_conv_layer.name:
            conv_output = x
            break

    grad_model = Model(inputs=input_tensor, outputs=[conv_output, x])
    print("Grad-CAM модель создана")

    test_img = np.random.random((1, 128, 128, 3))
    test_heatmap = make_gradcam_heatmap(test_img, grad_model)
    print(f"Тест Grad-CAM: {test_heatmap.shape}")

except Exception as e:
    print(f"Ошибка Grad-CAM: {e}")
    grad_model = None

def preprocess_image(image_data: bytes, filename: str = ""):
    """УНИВЕРСАЛЬНАЯ предобработка JPG/PNG/TIF"""
    try:
        if TIF_SUPPORT and filename.lower().endswith(('.tif', '.tiff')):
            print("IF через imageio...")
            img_array = imageio.imread(io.BytesIO(image_data))
        else:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            img_array = np.array(image)

        img_array = cv2.resize(img_array, (128, 128))

        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        img_array = img_array.astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Ошибка предобработки: {e}")

app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/")
def frontend():
    html_path = BASE_DIR / "front.html"
    if not html_path.exists():
        return JSONResponse({"error": "front.html не найден!"})
    return FileResponse(str(html_path), media_type="text/html")

@app.post("/preview_tiff")
async def preview_tiff(file: UploadFile = File(...)):
    try:
        print(f"Предпросмотр TIF: {file.filename}")
        image_data = await file.read()

        if not TIF_SUPPORT:
            return JSONResponse({"error": "Установите: pip install imageio"}, status_code=400)

        img_array = imageio.imread(io.BytesIO(image_data))
        print(f"TIF загружен: {img_array.shape}, dtype={img_array.dtype}")

        # Изменяем размер для предпросмотра
        img_array = cv2.resize(img_array, (512, 512))

        # Нормализация каналов
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        if img_array.dtype != np.uint8:
            if img_array.max() > 1.0:
                img_array = np.clip(img_array / img_array.max(), 0, 1) * 255
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        else:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img_array)
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
        output_path = f"output/preview_{safe_filename.rsplit('.', 1)[0]}.png"
        pil_img.save(output_path)
        print(f"TIF предпросмотр сохранен: {output_path}")

        return JSONResponse({"preview_url": f"/output/{os.path.basename(output_path)}"})
    except Exception as e:
        print(f"TIF предпросмотр ошибка: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        processed_image = preprocess_image(image_data, file.filename)
        prediction = model.predict(processed_image, verbose=0)
        tumor_prob = float(prediction[0][0])
        return JSONResponse({"tumor_probability": tumor_prob})
    except Exception as e:
        print(f"Predict ошибка: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/predict_with_gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    if grad_model is None:
        return JSONResponse({"error": "Grad-CAM не доступен"}, status_code=500)

    try:
        print(f"🔄 Grad-CAM: {file.filename}")
        image_data = await file.read()
        processed_image = preprocess_image(image_data, file.filename)

        prediction = model.predict(processed_image, verbose=0)
        tumor_prob = float(prediction[0][0])
        print(f"Предсказание: {tumor_prob:.3f}")

        heatmap = make_gradcam_heatmap(processed_image, grad_model)

        original_processed = processed_image[0]  # Убираем batch dimension

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_processed)
        plt.title("Исходное изображение", fontsize=14, fontweight='bold')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title("Карта признаков", fontsize=14, fontweight='bold')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("output/gradcam_result.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        label = "Подозрение на опухоль" if tumor_prob > 0.5 else "Нет подозрений на опухоль"

        return JSONResponse({
            "tumor_probability": tumor_prob,
            "label": label,
            "gradcam_image": "/output/gradcam_result.png"
        })
    except Exception as e:
        print(f"Grad-CAM ошибка: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=400)
