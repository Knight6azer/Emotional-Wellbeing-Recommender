from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import numpy as np
import cv2
import io
import shutil
import os
from pydantic import BaseModel

# Import our modules
from backend.models.vision_model import build_vision_model
from backend.models.text_model import TextEmotionModel
from backend.models.fusion_engine import FusionEngine
from backend.preprocessing.vision_data import preprocess_face
from backend.personalization import PersonalizationEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WBRM_Backend")

app = FastAPI(
    title="WBRM Multimodal Emotion System",
    description="Research-grade backend for Vision, Audio, and Text emotion recognition.",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Components
logger.info("Initializing System Components...")

# 1. Vision Model
try:
    vision_model = build_vision_model()
    # In a real scenario, we would load weights here:
    # vision_model.load_weights("backend/weights/vision_weights.h5")
    logger.info("Vision Model initialized (Untrained).")
except Exception as e:
    logger.error(f"Vision Model Init Failed: {e}")
    vision_model = None

# 2. Text Model
try:
    text_model = TextEmotionModel()
except Exception as e:
    logger.error(f"Text Model Init Failed: {e}")
    text_model = None

# Audio model is skipped for this MVP integration to save memory/complexity, 
# but the architecture is defined in models/audio_model.py

# 3. Fusion & Personalization
fusion_engine = FusionEngine()
personalization = PersonalizationEngine()

EMOTIONS = ["sad", "angry", "disgust", "fear", "happy", "neutral", "surprise"]

class TextRequest(BaseModel):
    text: str
    user_id: str = "guest"

@app.get("/")
async def root():
    return {"status": "online", "message": "WBRM Emotion System Ready"}

@app.post("/analyze/vision")
async def analyze_vision(file: UploadFile = File(...)):
    """
    Analyzes a video frame for facial expression.
    """
    try:
        if not vision_model:
            return {"error": "Vision model not active"}

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess
        # Face detection would normally happen here or on client. 
        # For this research backend, we assume client sends a cropped face OR we detect it.
        # Let's add simple detection for robustness:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
        else:
            face_roi = frame # Fallback to full frame
            
        processed_face = preprocess_face(face_roi)
        
        # Inference
        preds = vision_model.predict(processed_face, verbose=0)[0]
        # preds is array of 7 floats
        
        scores = {e: float(p) for e, p in zip(EMOTIONS, preds)}
        dominant = max(scores, key=scores.get)
        
        return {"emotion": dominant, "scores": scores, "status": "success"}
        
    except Exception as e:
        logger.error(f"Vision Analysis Error: {e}")
        return {"error": str(e)}

@app.post("/analyze/text")
async def analyze_text(request: TextRequest):
    """
    Analyzes text semantics for emotion using Transformers.
    """
    try:
        if not text_model:
             return {"error": "Text model not active"}
             
        scores = text_model.predict(request.text)
        if not scores:
            return {"emotion": "neutral", "scores": {}, "status": "failed"}
            
        # Map HF scores to our EMOTIONS set if needed (handled in wrapper but let's ensure)
        # For simplicity, we assume wrapper returns compatible dict or we handle it here.
        
        # Standardize keys to lowercase
        standard_scores = {}
        for k, v in scores.items():
            # Basic mapping
            key = k.lower()
            if "joy" in key: key = "happy"
            if "sadness" in key: key = "sad"
            if key in EMOTIONS:
                standard_scores[key] = v
        
        # Fill missing with 0
        for e in EMOTIONS:
            if e not in standard_scores:
                standard_scores[e] = 0.0
                
        dominant = max(standard_scores, key=standard_scores.get) if standard_scores else "neutral"
        
        # Personalization Hook
        personalization.update_user_emotion(request.user_id, dominant)
        rec = personalization.get_recommendation(request.user_id, dominant)
        
        return {
            "emotion": dominant, 
            "scores": standard_scores, 
            "recommendation": rec,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Text Analysis Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
