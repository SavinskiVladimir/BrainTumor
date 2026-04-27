from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from database import get_db
from auth import get_current_user
from services.image_service import preprocess_image, model, grad_model, make_gradcam_heatmap, save_history
from services.image_service import save_gradcam_result

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db), user=Depends(get_current_user)):
    image_data = await file.read()
    processed_image = preprocess_image(image_data, file.filename or "")
    prediction = model.predict(processed_image, verbose=0)
    tumor_prob = float(prediction[0][0])
    label = "Подозрение на опухоль" if tumor_prob > 0.5 else "Нет подозрений на опухоль"

    save_history(db, image_data, file.filename or "image.png", tumor_prob, label, user_id=user.id if user else None)
    return JSONResponse({"tumor_probability": tumor_prob, "label": label})

@router.post("/predict_with_gradcam")
async def predict_with_gradcam(file: UploadFile = File(...), db: Session = Depends(get_db), user=Depends(get_current_user)):
    if grad_model is None:
        return JSONResponse({"error": "Grad-CAM недоступен"}, status_code=500)

    image_data = await file.read()
    processed_image = preprocess_image(image_data, file.filename or "")
    prediction = model.predict(processed_image, verbose=0)
    tumor_prob = float(prediction[0][0])
    heatmap = make_gradcam_heatmap(processed_image, grad_model)
    label = "Подозрение на опухоль" if tumor_prob > 0.5 else "Нет подозрений на опухоль"

    img_row, _ = save_history(db, image_data, file.filename or "image.png", tumor_prob, label, user_id=user.id if user else None)
    gradcam_url = save_gradcam_result(img_row.file_path, heatmap)

    return JSONResponse({
        "tumor_probability": tumor_prob,
        "label": label,
        "gradcam_image": gradcam_url
    })