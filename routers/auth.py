from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from database import get_db
from models import User
from schemas import UserCreate, UserLogin
from auth import hash_password, verify_password, create_token

router = APIRouter(prefix="/api")

@router.post("/register")
def api_register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        return JSONResponse({"error": "Пользователь существует"}, status_code=400)
    hashed_password = hash_password(user.password)
    new_user = User(username=user.username, email=user.email, password=hashed_password)
    db.add(new_user)
    db.commit()
    return {"message": "Зарегистрирован"}

@router.post("/login")
def api_login(user: UserLogin, response: Response, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        return JSONResponse({"error": "Неверные данные"}, status_code=400)
    token = create_token(db_user.id)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=24 * 3600,
        samesite="lax"
    )
    return {"message": "Вход успешен"}

@router.post("/logout")
def api_logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("access_token")
    return response