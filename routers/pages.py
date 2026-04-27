from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from auth import get_current_user

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/login")

@router.get("/login", response_class=HTMLResponse)
async def get_login_page():
    return FileResponse("static/login.html")

@router.get("/register", response_class=HTMLResponse)
async def get_register_page():
    return FileResponse("static/register.html")

@router.get("/app", response_class=HTMLResponse)
async def frontend_app(user=Depends(get_current_user)):
    if not user:
        return RedirectResponse("/login")
    return FileResponse("front.html")

@router.get("/history", response_class=HTMLResponse)
async def history_page(user=Depends(get_current_user)):
    if not user:
        return RedirectResponse("/login")
    return FileResponse("static/history.html")