from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from database import Base, engine
from routers.pages import router as pages_router
from routers.auth import router as auth_router
from routers.predict import router as predict_router
from routers.history import router as history_router

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")

app.include_router(pages_router)
app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(history_router)