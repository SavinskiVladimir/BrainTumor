from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()

# Загружаем модель
model = tf.keras.models.load_model(BASE_DIR / "model" / "model.keras")

def preprocess(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return np.expand_dims(image, axis=0)

@app.get("/")
def frontend():
    html_path = BASE_DIR / "front.html"
    if not html_path.exists():
        return JSONResponse({"error": "front.html не найден!"})
    return FileResponse(str(html_path), media_type="text/html")

def preprocess(image):
    image = image.resize((128, 128))  # Модель ожидает 128x128
    image = np.array(image) / 255.0
    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]  # Убираем альфа-канал
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        processed_image = preprocess(image)
        prediction = model.predict(processed_image, verbose=0)
        tumor_prob = float(prediction[0][0])
        return JSONResponse(content={"tumor_probability": tumor_prob})
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=400
        )