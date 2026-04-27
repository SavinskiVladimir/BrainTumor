from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from database import get_db
from auth import get_current_user
from models import ImageDB, Prediction

router = APIRouter(prefix="/api")

@router.get("/history")
def api_history(db: Session = Depends(get_db), user=Depends(get_current_user)):
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)

    rows = (
        db.query(ImageDB, Prediction)
        .join(Prediction, Prediction.image_id == ImageDB.id)
        .filter(ImageDB.user_id == user.id)
        .order_by(desc(Prediction.id))
        .all()
    )

    return [
        {
            "image": "/" + img.file_path.replace("\\", "/"),
            "result": pred.result,
            "probability": pred.probability
        }
        for img, pred in rows
    ]