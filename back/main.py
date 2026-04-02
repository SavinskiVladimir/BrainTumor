from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

model = tf.keras.models.load_model("./model/model.keras")

def preprocess(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess(image)
    pred = model.predict(img)[0][0]

    return {
        "probability": float(pred),
        "class": "tumor" if pred > 0.5 else "healthy"
    }