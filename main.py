from fastapi import FastAPI, UploadFile, File, Depends, Response, Cookie
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
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
import imageio.v3 as imageio

from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session

SECRET_KEY = "SECRET123"
ALGORITHM = "HS256"
DATABASE_URL = "sqlite:///./database.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

TIF_SUPPORT = True
BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password = Column(String)

class ImageDB(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_path = Column(String)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    probability = Column(Float)
    result = Column(String)

Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"])

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)

def create_token(user_id):
    payload = {"sub": str(user_id), "exp": datetime.utcnow() + timedelta(hours=24)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(access_token: str = Cookie(None)):
    if not access_token:
        return None
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
        db = SessionLocal()
        user = db.query(User).get(user_id)
        db.close()
        return user
    except Exception:
        return None

os.makedirs("output", exist_ok=True)

print("Загрузка модели...")
model = tf.keras.models.load_model(BASE_DIR / "model" / "model.keras")
print("Модель загружена")

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

def get_earlier_conv_layer_name(model):
    conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    if not conv_layers:
        raise ValueError("Нет Conv2D слоев")
    return conv_layers[min(1, len(conv_layers)-1)].name

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
except Exception as e:
    grad_model = None
    print(f"Grad-CAM ошибка: {e}")

def preprocess_image(image_data: bytes, filename: str = ""):
    try:
        if TIF_SUPPORT and filename.lower().endswith(('.tif', '.tiff')):
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
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/login")

@app.get("/login", response_class=HTMLResponse)
async def get_login_page():
    return FileResponse("static/login.html")

@app.get("/register", response_class=HTMLResponse)
async def get_register_page():
    return FileResponse("static/register.html")

@app.get("/app", response_class=HTMLResponse)
async def frontend_app(user=Depends(get_current_user)):
    if not user:
        return RedirectResponse("/login")
    return FileResponse("front.html")

@app.post("/api/register")
def api_register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        return JSONResponse({"error": "Пользователь существует"}, status_code=400)
    hashed_password = hash_password(user.password)
    new_user = User(username=user.username, email=user.email, password=hashed_password)
    db.add(new_user)
    db.commit()
    return {"message": "Зарегистрирован"}

@app.post("/api/login")
def api_login(user: UserLogin, response: Response, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        return JSONResponse({"error": "Неверные данные"}, status_code=400)
    token = create_token(db_user.id)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=24*3600,
        samesite="lax"
    )
    return {"message": "Вход успешен"}

@app.post("/api/logout")
def api_logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Вы вышли"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    processed_image = preprocess_image(image_data, file.filename or "")
    prediction = model.predict(processed_image, verbose=0)
    tumor_prob = float(prediction[0][0])
    label = "Подозрение на опухоль" if tumor_prob > 0.5 else "Нет подозрений на опухоль"
    return JSONResponse({"tumor_probability": tumor_prob, "label": label})

@app.post("/predict_with_gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    if grad_model is None:
        return JSONResponse({"error": "Grad-CAM недоступен"}, status_code=500)
    image_data = await file.read()
    processed_image = preprocess_image(image_data, file.filename or "")
    prediction = model.predict(processed_image, verbose=0)
    tumor_prob = float(prediction[0][0])
    heatmap = make_gradcam_heatmap(processed_image, grad_model)
    original_processed = processed_image[0]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_processed)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("output/gradcam_result.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    label = "Подозрение на опухоль" if tumor_prob > 0.5 else "Нет подозрений на опухоль"
    return JSONResponse({"tumor_probability": tumor_prob, "label": label, "gradcam_image": "/output/gradcam_result.png"})