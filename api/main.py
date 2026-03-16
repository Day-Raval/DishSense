import os
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile # type: ignore
from fastapi.responses import JSONResponse # type: ignore

from api.schemas import HealthResponse, RecommendResponse
from src.config import CHROMA_PATH, COLLECTION_NAME
from src.pipeline import recommend_from_photo

import chromadb # type: ignore

app = FastAPI(
    title="FridgeRAG API",
    description=(
        "Upload a fridge photo → detect ingredients via YOLOv8 + DETR + CLIP "
        "→ retrieve and rank recipes via RAG + GPT-4o-mini."
    ),
    version="1.0.0",
)

UPLOAD_DIR    = "/tmp/fridgerag_uploads"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(
    photo:        UploadFile = File(...,
                      description="Fridge photo — JPEG, PNG, or WEBP"),
    preferences:  str = Form("",
                      description="Dietary preferences e.g. 'high protein, low carb'"),
    max_calories: int = Form(0,
                      description="Max calories per serving — 0 means no limit"),
    max_minutes:  int = Form(0,
                      description="Max cook time in minutes — 0 means no limit"),
    top_n:        int = Form(5, ge=1, le=10,
                      description="Number of recipes to return"),
):
    if photo.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{photo.content_type}'. "
                "Please upload a JPEG, PNG, or WEBP image."
            ),
        )

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(photo.file, f)

    try:
        result = recommend_from_photo(
            image_path=tmp_path,
            user_preferences=preferences,
            max_calories=float(max_calories) if max_calories > 0 else None,
            max_minutes=int(max_minutes)     if max_minutes  > 0 else None,
            top_n=top_n,
        )
        return JSONResponse(content=result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        c     = chromadb.PersistentClient(path=CHROMA_PATH)
        col   = c.get_collection(COLLECTION_NAME)
        count = col.count()
        ready = count > 0
    except Exception:
        count = 0
        ready = False

    return {
        "status":        "ok",
        "models_loaded": ["YOLOv8", "DETR", "CLIP (ViT-B-32)", "all-MiniLM-L6-v2"],
        "db_ready":      ready,
        "recipe_count":  count,
    }

# ── Run ───────────────────────────────────────────────────────────────────────
# uvicorn api.main:app --reload --port 8000