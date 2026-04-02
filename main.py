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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    tumor_prob = float(prediction[0][0])
    return JSONResponse(content={"tumor_probability": tumor_prob})