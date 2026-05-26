from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
from utils import analyze, load_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()    
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "Skin Lesion Medical Assistant API"}

@app.post("/analyze")
async def analyze_lesion(image: UploadFile = File(...),
                         symptoms: str = Form(...)):
    img_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(img_bytes))
    result = analyze(pil_img, symptoms)
    return result