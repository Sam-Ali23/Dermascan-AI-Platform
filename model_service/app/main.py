from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2

from app.model_loader import ModelLoader
from app.inference import run_inference

app = FastAPI(
    title="Dermascan AI Model Service",
    version="1.0.0",
    description="AI inference API for dermoscopic skin lesion analysis",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    predicted_class: int
    label: str
    confidence: float
    lesion_area_ratio: float
    mask_base64: str
    overlay_base64: str


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "skin_multitask_ai.pth"

loader = ModelLoader(str(MODEL_PATH))
model = loader.get_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        image = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode the image.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = run_inference(model, image)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")